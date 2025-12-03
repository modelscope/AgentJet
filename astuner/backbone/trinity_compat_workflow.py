import asyncio
import os

import datasets
import openai
from loguru import logger
from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS
from trinity.common.workflows.workflow import Task as TrinityTask
from trinity.common.workflows.workflow import Workflow

try:
    from trinity.buffer.reader import READER
    from trinity.buffer.reader.file_reader import TaskFileReader, _HFBatchReader
    from trinity.buffer.schema.formatter import FORMATTER

    logger.success("[New Trinity] Trinity imports successful.")
except ImportError:
    logger.success("[Old Trinity] Using old trinity.")

from typing import List, Literal, Optional, cast

from transformers import AutoTokenizer

from astuner.backbone.common_warm_up import warm_up_process
from astuner.context_tracker.agentscope_tracker.multiagent_tracking import (
    MultiAgentContextTracker,
)
from astuner.schema.trajectory import Sample
from astuner.task_rollout.native_parallel_worker import DynamicRolloutManager
from astuner.utils.config_utils import read_astune_config


class TrinityCompatWorkflow(DynamicRolloutManager):
    def __init__(
        self,
        is_eval,
        task,
        llm_handle,
        tokenizer,
        config,
        llm_mode: Literal["local", "remote", "trinity"] = "trinity",
        **kwargs,
    ):
        self.is_eval = is_eval
        self.task = task
        self.tokenizer = tokenizer
        self.config = config
        self.llm_mode = llm_mode

        super().__init__(
            config=self.config,
            async_rollout_manager=llm_handle,
            max_parallel=1,
            max_llm_retries=1,
            tokenizer=tokenizer,
            llm_mode=llm_mode,
            **kwargs,
        )

    def convert_task(self, task: TrinityTask):
        from astuner.schema.task import Task

        d = {}
        for vip_key in ["main_query", "task_id", "env_type", "metadata", "init_messages"]:
            if vip_key not in task.raw_task:
                raise ValueError(f"Key {vip_key} not found in task.raw_task")
            d[vip_key] = task.raw_task[vip_key]
        return Task(**d)

    def thread_worker(self):
        obs_window = {
            "stop": [False],
            "step": [0],
            "token": [0],
        }
        astune_task = self.convert_task(self.task)
        return self.rollout_env_worker(
            task=astune_task,
            task_batch_index=0,
            task_tag=f"T{astune_task.task_id}#R",
            mode="sample" if not self.is_eval else "validate",
            task_thread_index=0,
            obs_window=obs_window,
        )

    async def run_in_new_thread(self) -> MultiAgentContextTracker:
        return cast(
            MultiAgentContextTracker,
            await asyncio.to_thread(self.thread_worker),
        )


@WORKFLOWS.register_module("astuner_workflow")
class ASTunerWorkflowWrap(Workflow):
    is_async: bool = True

    def __init__(
        self,
        model: ModelWrapper,
        task: TrinityTask,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.task = task
        self.model_client = model.get_openai_async_client()
        self.is_eval = task.is_eval
        # extract the query and the answer from the task
        self.query = task.raw_task.get(task.format_args.prompt_key)  # type: ignore [index]
        self.answer = task.raw_task.get(task.format_args.response_key)  # type: ignore [index]

    async def run_async(self):
        yaml_path = os.environ.get("ASTUNER_CONFIG_REDIRECT", None)
        if yaml_path is None:
            raise ValueError("ASTUNER_CONFIG_REDIRECT is not set in environment variables")
        astune_config = read_astune_config(yaml_path)
        warm_up_process(astune_config)
        tracker = await TrinityCompatWorkflow(
            is_eval=self.is_eval,
            task=self.task,
            llm_handle=self.model_client,
            tokenizer=AutoTokenizer.from_pretrained(self.model_client.model_path),
            config=astune_config,
        ).run_in_new_thread()

        sample_final = []
        try:
            sample_arr = tracker.group_tokenize()
        except Exception as e:
            raise e
        finally:
            tracker.generate_log(global_step="NA")
        sample_final += sample_arr

        exps = []
        for _, sample in enumerate(sample_final):
            sample: Sample
            input_ids = sample.input_ids
            prompt_ids = sample.prompt_ids
            response_ids = sample.response_ids
            # attention_mask = sample.attention_mask
            # prompt_attention_mask = sample.prompt_attention_mask
            # response_attention_mask = sample.response_attention_mask
            # loss_mask = sample.loss_mask
            # prompt_loss_mask = sample.prompt_loss_mask
            response_loss_mask = sample.response_loss_mask
            # position_ids = sample.position_ids
            # prompt_position_ids = sample.prompt_position_ids
            # response_position_ids = sample.response_position_ids
            # tracker_tokenized["step_reward"] = self.reward_structure.step_reward[index]

            logprobs = sample.response_logprobs
            try:
                reward = tracker.reward_structure.step_reward
                if isinstance(reward, list):
                    reward = reward[0]
            except Exception:
                reward = tracker.reward_structure.raw_reward
            if not isinstance(
                reward, (float, int)
            ):  # if reward is still not a float or int, set it to 0.0
                reward = tracker.reward_structure.raw_reward

            if (
                len(response_ids) + len(prompt_ids) == len(input_ids)
                and len(logprobs) == len(response_ids)
                and len(logprobs) > 0
            ):
                exp = Experience(
                    # eid=uuid.uuid4().hex,
                    tokens=input_ids,  # [seq_length] prompt + response
                    prompt_length=len(
                        prompt_ids
                    ),  # Length of the prompt in tokens, used for generating attention masks
                    logprobs=logprobs,  # [resp_length]
                    reward=reward,  #
                    # advantages=None,
                    # returns=None,
                    info={},
                    metrics={},  # for wandb logging (must be string:float)
                    response_text="",  # optional
                    prompt_text="",  # optional
                    #### for multi-turn experiences
                    action_mask=response_loss_mask,  # 1 stands for training, 0 stands for ignoring
                    messages=sample.messages,  #
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
                logger.exception("Data length mismatch when converting sample to experience.")
        return exps


try:

    @READER.register_module("astuner")
    class ASTunerTaskReader(TaskFileReader):
        def __init__(self, config):
            self.config = config
            self.read_batch_size = config.batch_size
            self.split = config.split

            yaml_path = os.environ.get("ASTUNER_CONFIG_REDIRECT", None)
            if yaml_path is None:
                raise ValueError("ASTUNER_CONFIG_REDIRECT is not set in environment variables")
            astune_config = read_astune_config(yaml_path)

            from astuner.task_reader import TaskReaderRouter, task_to_standard_dataset

            task_reader = TaskReaderRouter(astune_config)

            dataset_segments = []
            if "train" in self.split:
                dataset_segments.append(task_to_standard_dataset(task_reader.get_training_tasks()))
            if "val" in self.split:
                dataset_segments.append(
                    task_to_standard_dataset(task_reader.get_validation_tasks())
                )
            if not dataset_segments:
                raise ValueError(
                    f"Unsupported split '{self.split}'. Expected to contain 'train' or 'val'."
                )

            concatenated_dataset = (
                dataset_segments[0]
                if len(dataset_segments) == 1
                else datasets.concatenate_datasets(dataset_segments)
            )

            self.dataset = _HFBatchReader(
                concatenated_dataset,
                name=self.config.name,
                default_batch_size=self.read_batch_size,
                total_epochs=self.config.total_epochs if not self.config.is_eval else 1,
                offset=self.config.index,
                drop_last=not self.config.is_eval,
                total_steps=self.config.total_steps,
                enable_progress_bar=self.config.enable_progress_bar,
            )
            self.formatter = FORMATTER.get("task")(self.config)

        def read(self, batch_size: Optional[int] = None) -> List:
            batch_size = batch_size or self.read_batch_size
            tasks = []
            samples, indices = self.dataset.read_batch(batch_size)
            for sample in samples:
                task = self.formatter.format(sample)
                tasks.append(task)
            return tasks

except Exception:
    pass
