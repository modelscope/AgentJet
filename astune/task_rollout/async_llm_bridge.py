import os
import copy
import time
import numpy as np
import torch
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Literal, Callable, Any
from loguru import logger
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from astune.workflow_controller.classic_agentflow import AgentFlow
from astune.workflow_controller.classic_agentflow import BaseAgentFlow
from astune.task_rollout.env_worker import EnvWorker
from astune.schema.task import Task, TaskLaunchCoreArgument
from astune.schema.trajectory import Sample
from astune.context_manager.cmt_linear import CMTLinear, CMTBaseAttr
from beast_logger import register_logger, print_dict, print_listofdict
from astune.workflow_controller.agentscope_flow import AgentScopeWorkflow
from astune.utils.utils import run_async_coro__no_matter_what
from astune.schema.logprob import TokenAndProb

class AsyncLlmBridge(object):


    def get_llm_chat_fn(self, sampling_params: dict = {}) -> Callable:
        def llm_chat(
            messages: List[Dict[str, str]],
            custom_sampling_params: dict = {},
            tools = [],
            request_id: str = ""
        ) -> dict:

            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            tools = messages[-1].get("tools", None)
            for msg in messages: msg.pop("tools", None)

            input_messages = copy.deepcopy(messages)
            request_id = uuid.uuid4().hex
            if tools is not None:
                prompt_ids = self.tokenizer.apply_chat_template(input_messages, add_generation_prompt=True, tokenize=True, tools=tools)
            else:
                prompt_ids = self.tokenizer.apply_chat_template(input_messages, add_generation_prompt=True, tokenize=True)

            final_res = run_async_coro__no_matter_what(self.async_rollout_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=updated_sampling_params,
                )
            )

            if self.config.astune.rollout.name == 'vllm':
                token_array = final_res.outputs[0].token_ids
            elif self.config.astune.rollout.name == 'sglang':
                token_array = final_res

            decoded_text = self.tokenizer.decode(token_array) # type: ignore

            if decoded_text.endswith('<|im_end|>'):
                decoded_text = decoded_text[:-len('<|im_end|>')]

            return {
                "role": "assistant",
                "request_id": request_id,
                "content": decoded_text,
                "tokens": [
                    TokenAndProb(
                        token_id=token,
                        logprob=-1,
                        decoded_string=self.tokenizer.decode(token)
                    )
                    for token in token_array    # type: ignore
                ]
            }

        def llm_chat_remote(
            messages: List[Dict[str, str]],
            custom_sampling_params: dict = {},
            tools = [],
            request_id: str = ""
        ) -> dict:

            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)
            updated_sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})
            input_messages = copy.deepcopy(messages)
            for i in range(self.max_llm_retries):
                try:
                    # this function is defined in `astune/main_vllm.py`
                    output_message = self.async_rollout_manager.submit_chat_completions(
                        messages=input_messages,
                        sampling_params=updated_sampling_params,
                        tools=tools,
                        request_id=request_id
                    )
                    break
                except Exception as e:
                    logger.bind(exception=True).exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)
            return output_message[-1]   # type: ignore



        def llm_chat_trinity(
            messages: List[Dict[str, str]],
            custom_sampling_params: dict = {},
            tools = [],
            request_id: str = ""
        ) -> dict:

            async def main(model_client):
                updated_sampling_params = {}
                if sampling_params:
                    updated_sampling_params.update(sampling_params)
                if custom_sampling_params:
                    updated_sampling_params.update(custom_sampling_params)
                updated_sampling_params.pop('min_tokens')

                if tools:
                    response = await model_client.chat.completions.create(
                        model=model_client.model_path,
                        messages=messages,
                        logprobs=True,
                        tools=tools,
                        top_logprobs=0,
                        **updated_sampling_params
                    )
                else:
                    response = await model_client.chat.completions.create(
                        model=model_client.model_path,
                        messages=messages,
                        logprobs=True,
                        top_logprobs=0,
                        **updated_sampling_params
                    )
                return response

            assert hasattr(self, 'trinity_llm_model_client'), "trinity_llm_model_client is not set in AsyncLlmBridge"
            response = run_async_coro__no_matter_what(main(self.trinity_llm_model_client)) # type: ignore

            content = response.choices[0].message.content
            message = response.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)

            if content is None:
                content = ""

            return {
                "role": "assistant",
                "request_id": response.id,
                "content": content,
                "tool_calls": message.get("tool_calls", None),
                "tokens": [
                    TokenAndProb(
                        token_id=token,
                        logprob=tokenlogprob.logprob,
                        decoded_string=tokenlogprob.token
                    )
                    for tokenlogprob, token in zip(
                        response.choices[0].logprobs.content,
                        response.choices[0].token_ids
                    )
                ]
            }

        if self.llm_mode == "remote":
            return llm_chat_remote
        if self.llm_mode == "trinity":
            return llm_chat_trinity
        else:
            return llm_chat


