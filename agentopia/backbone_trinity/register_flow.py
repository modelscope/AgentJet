import os
import uuid
import hydra
import openai
import numpy as np
import asyncio, uuid, copy
import threading

from typing import Dict, List, Optional, Union
from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow
from trinity.common.workflows.agentscope.react.templates import TEMPLATE_MAP
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Literal, Callable, Union
from loguru import logger
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from beast_logger import register_logger, print_dict, print_listofdict
from agentopia.schema.task import Task
from agentopia.utils.utils import run_async_coro__no_matter_what_the_fuck
from agentopia.parallel_env import DynamicRollout
from agentopia.schema.logprob import TokenAndProb
from agentopia.schema.task import Task
from agentopia.schema.trajectory import Sample
from omegaconf import OmegaConf

class TrinityCompatWorkflow(DynamicRollout):

    def __init__(self, task, llm_handle, tokenizer, config, llm_mode="trinity", **kwargs):

        self.task = task
        self.trinity_llm_model_client = llm_handle
        self.tokenizer = tokenizer
        self.config = config
        self.llm_mode = "trinity"

        super().__init__(
            config=self.config,
            async_rollout_manager=None,
            max_parallel=1,
            max_llm_retries = 1,
            tokenizer=tokenizer,
            llm_mode=llm_mode,
            **kwargs
        )

    def convert_task(self, task):
        main_query = task.raw_task.get('main_query', "[not defined]")
        task_id = task.raw_task.get('task_selector', str(uuid.uuid4().hex))
        env_type = task.raw_task.get('env_type', "[not defined]")
        metadata = task.raw_task.get('metadata', {})
        init_messages = task.raw_task.get('init_messages', [])

        return Task(
            main_query=main_query,
            task_id=task_id,
            env_type=env_type,
            metadata=metadata,
            init_messages=init_messages,
        )

    def thread_worker(self):
        obs_window = {
            'stop': [False],
            'step': [0],
            'token': [0],
        }
        agentopia_task = self.convert_task(self.task)
        return self.rollout_env_worker(
            task=agentopia_task,
            task_batch_index=0,
            task_tag=f"T{agentopia_task.task_id}#R?",
            mode="sample",
            task_thread_index=0,
            obs_window=obs_window
        )

    def run_in_new_thread(self):
        # begin self.thread_worker in a new thread
        # then wait for it to finish, and get the result

        result_holder = {}
        exc_holder = {}

        def _target():
            try:
                result_holder["result"] = self.thread_worker()
            except Exception as e:
                exc_holder["exc"] = e

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join()

        if "exc" in exc_holder:
            raise exc_holder["exc"]

        return result_holder.get("result", None)


def read_astune_config(yaml_fp):
    from hydra import initialize, compose
    from omegaconf import DictConfig

    def load_hydra_config(config_path: str, config_name: str) -> DictConfig:
        with initialize(config_path=config_path, version_base=None):
            cfg = compose(config_name=config_name, overrides=[])
            return cfg

    dir_path = os.path.dirname(yaml_fp)
    file_name = os.path.basename(yaml_fp)
    return load_hydra_config(config_path=dir_path, config_name=file_name)


@WORKFLOWS.register_module("agentopia_workflow")
class AgentopiatWorkflowWrap(Workflow):
    is_async: bool = True
    def __init__(
        self,
        config,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.config = config
        self.task = task

        # 模拟openai的异步客户端
        self.model_client = model.get_openai_async_client()
        # task_type 用于获取奖励函数
        # extract the query and the answer from the task
        self.query = task.raw_task.get(task.format_args.prompt_key)  # type: ignore [index]
        self.answer = task.raw_task.get(task.format_args.response_key)  # type: ignore [index]
        self.task.workflow_args = {
            "env_type": "appworld",
            "task_id": self.task.task_id,
            "instance_id": uuid.uuid4().hex,
        }

    async def run_async(self):

        yaml_path = os.environ.get('ASTUNE_CONFIG_REDIRECT', None)
        if yaml_path is None:
            raise ValueError("ASTUNE_CONFIG_REDIRECT is not set in environment variables")

        cmt = TrinityCompatWorkflow(
            task=self.task,
            llm_handle=self.model_client,
            tokenizer=AutoTokenizer.from_pretrained(self.model_client.model_path),
            config=read_astune_config(os.path.relpath(yaml_path, os.path.dirname(__file__))),
        ).run_in_new_thread()

        sample_final = []
        try:
            sample_arr = cmt.group_tokenize()
        except Exception as e:
            cmt.generate_log(global_step=-1)
            raise e
        cmt.generate_log(global_step=-1)
        sample_final += sample_arr


        exps = []
        for index, sample in enumerate(sample_final):
            sample: Sample
            input_ids = sample.input_ids
            prompt_ids = sample.prompt_ids
            response_ids = sample.response_ids
            attention_mask = sample.attention_mask
            prompt_attention_mask = sample.prompt_attention_mask
            response_attention_mask = sample.response_attention_mask
            loss_mask = sample.loss_mask
            prompt_loss_mask = sample.prompt_loss_mask
            response_loss_mask = sample.response_loss_mask
            position_ids = sample.position_ids
            prompt_position_ids = sample.prompt_position_ids
            response_position_ids = sample.response_position_ids
            # cmt_tokenized["step_reward"] = self.reward_structure.step_reward[index]

            logprobs = sample.response_logprobs
            try:
                reward = cmt.reward_structure.step_reward
                if isinstance(reward, list):
                    reward = reward[0]
            except Exception as e:
                reward = cmt.reward_structure.raw_reward
            if not isinstance(reward, (float, int)): # if reward is still not a float or int, set it to 0.0
                reward = cmt.reward_structure.raw_reward

            if len(response_ids) + len(prompt_ids) == len(input_ids) and len(logprobs) == len(response_ids) and len(logprobs) > 0:
                exp = Experience(
                    # eid=uuid.uuid4().hex,
                    tokens = input_ids,     # [seq_length] prompt + response
                    prompt_length = len(prompt_ids),  # Length of the prompt in tokens, used for generating attention masks
                    logprobs = logprobs,   # [resp_length]
                    reward = reward,  #
                    # advantages=None,
                    # returns=None,
                    info = {},
                    metrics = {},   # for wandb logging (must be string:float)
                    response_text = "", # optional
                    prompt_text = "", # optional
                    #### for multi-turn experiences
                    action_mask = response_loss_mask,  # 1 是训练
                    messages=sample.messages,    #
                    # tools,
                    #### for dpo experiences
                    # chosen,
                    # rejected,
                    # chosen_messages,
                    # rejected_messages,
                    #### for multi-modal data
                    # multi_modal_inputs
                )
                exps += [exp]
            else:
                from vsdb import bp
                bp("BUGX")
        return exps
