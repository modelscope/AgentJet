import os
import uuid
import openai
import threading

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow
from typing import List, Literal, Optional
from loguru import logger
from astune.schema.task import Task
from transformers import AutoTokenizer
from astune.task_rollout.native_parallel_worker import DynamicRollout
from astune.schema.trajectory import Sample
from astune.utils.config_utils import read_astune_config
from astune.context_manager.cmt_base_attr import CMTBaseAttr

class TrinityCompatWorkflow(DynamicRollout):

    def __init__(
        self,
        task,
        llm_handle,
        tokenizer,
        config,
        llm_mode: Literal['local', 'remote', 'trinity'] = "trinity",
        **kwargs
    ):

        self.task = task
        self.trinity_llm_model_client = llm_handle
        self.tokenizer = tokenizer
        self.config = config
        self.llm_mode = llm_mode

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
        astune_task = self.convert_task(self.task)
        return self.rollout_env_worker(
            task=astune_task,
            task_batch_index=0,
            task_tag=f"T{astune_task.task_id}#R?",
            mode="sample",
            task_thread_index=0,
            obs_window=obs_window
        )

    def run_in_new_thread(self) -> CMTBaseAttr:
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

        thread_conclusion: CMTBaseAttr = result_holder.get("result", None)  # type: ignore
        return thread_conclusion



@WORKFLOWS.register_module("astune_workflow")
class ASTunetWorkflowWrap(Workflow):
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

        self.model_client = model.get_openai_async_client()
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
            config=read_astune_config(yaml_path),
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

            if len(response_ids) + len(prompt_ids) == len(input_ids) and \
                len(logprobs) == len(response_ids) and len(logprobs) > 0:
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
                logger.exception(f"Data length mismatch when converting sample to experience.")
        return exps
