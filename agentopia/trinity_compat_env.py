import os
import time
import numpy as np
import asyncio, uuid, copy
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
        return Task(task_id=task.raw_task['task_selector'],
            env_type=task.workflow_args['env_type'],
            metadata={},
            query=""
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
        import threading

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
