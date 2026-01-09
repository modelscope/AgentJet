import asyncio
import copy
import json
import time
import uuid
from typing import Any, Callable, Dict, List, Literal, Type, Union



from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseModel
from transformers.tokenization_utils import PreTrainedTokenizer
from vllm.entrypoints.openai.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
from vllm.outputs import RequestOutput as VerlVllmRequestOutput

from agentscope.model import ChatResponse as AgentScopeChatResponse
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion

ChatResponse = Union[OpenAIChatCompletion, AgentScopeChatResponse]

from ajet.context_tracker.multiagent_tracking import (
    MultiAgentContextTracker,
)
from ajet.schema.convertion import convert_llm_proxy_response_to_oai_response
from ajet.schema.convertion import convert_llm_proxy_response_to_agentscope_response
from ajet.schema.logprob import TokenAndProb
from ajet.utils.async_utils import run_async_coroutine_with_timeout
from ajet.utils.testing_utils import _mock_if_test_mode, _test_if_test_mode
from ajet.utils.tokenizer import ajet_apply_chat_template


class AjetStandardLlmBridgeRequest(BaseModel):
    messages: List[Dict[str, str]]
    custom_sampling_params: dict = {}
    tools: List = []
    request_id: str = ""

class AjetStandardLlmBridgeResponse(BaseModel):
    role: str = "assistant"
    request_id: str = ""
    content: str = ""
    tool_calls: List[Dict] = []
    tokens: List[TokenAndProb] = []


# -------------------------------------------------------------------------------------
# ------------------------ Unify LLM for Verl + Trinity + Vllm ------------------------
# -------------------------------------------------------------------------------------

class AsyncLlmBridge(object):
    def __init__(
        self,
        config: DictConfig,
        async_rollout_manager: Any,
        tokenizer: Any,
        llm_mode: Literal["local", "remote", "trinity"] = "local",
        max_llm_retries: int = 3,
    ):
        self.config = config
        self.async_rollout_manager = async_rollout_manager
        self.tokenizer = tokenizer
        self.llm_mode = llm_mode
        self.max_llm_retries = max_llm_retries

    def get_llm_inference_fn(self, sampling_params: dict = {}) -> Callable:  # noqa: C901

        def llm_chat_verl(
            messages: List[Dict[str, str]],
            custom_sampling_params: dict = {},
            tools=[],
            request_id: str = "",
        ) -> dict:
            request_id = uuid.uuid4().hex

            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            input_messages = copy.deepcopy(messages)
            prompt_text = ajet_apply_chat_template(
                tokenizer=self.tokenizer,
                conversation=input_messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_ids = self.tokenizer(prompt_text)["input_ids"]

            if self.config.ajet.execute_test:
                _test_if_test_mode("prompt_text", prompt_text, self.config)

            final_res = run_async_coroutine_with_timeout(
                self.async_rollout_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=updated_sampling_params,
                ),
                timeout=1800,
            )

            if self.config.ajet.rollout.name == "vllm":
                final_res: VerlVllmRequestOutput
                token_array = final_res.outputs[0].token_ids
                logprob_array = final_res.outputs[0].logprobs
            elif self.config.ajet.rollout.name == "sglang":
                token_array = final_res

            decoded_text = self.tokenizer.decode(token_array)  # type: ignore
            if self.config.ajet.execute_test:
                decoded_text = _mock_if_test_mode("mock_decoded_text", decoded_text, self.config)

            if decoded_text.endswith("<|im_end|>"):
                decoded_text = decoded_text[: -len("<|im_end|>")]

            # if tool call
            tool_calls = None
            if (
                ("<tool_call>" in decoded_text)
                and ("</tool_call>" in decoded_text)
                and (not self.config.ajet.rollout.force_disable_toolcalls)
            ):
                tool_parser = Hermes2ProToolParser(self.tokenizer)
                parsed_tool_calls = tool_parser.extract_tool_calls(decoded_text, None)  # type: ignore
                parsed_tool_calls = parsed_tool_calls.model_dump()
                if self.config.ajet.execute_test:
                    _test_if_test_mode(
                        "parsed_tool_calls", parsed_tool_calls["tool_calls"], self.config
                    )
                model_called = parsed_tool_calls["tools_called"]
                if model_called:
                    tool_calls = parsed_tool_calls["tool_calls"]
                    is_bad_toolcall = False
                    for i in range(len(tool_calls)):
                        if "function" in tool_calls[i] and "arguments" in tool_calls[i]["function"]:
                            expect_dict = json.loads(tool_calls[i]["function"]["arguments"])
                            if not isinstance(expect_dict, dict):
                                is_bad_toolcall = True
                    if is_bad_toolcall:
                        tool_calls = None
                        decoded_text = decoded_text
                    else:
                        decoded_text = parsed_tool_calls["content"]
                        if decoded_text is None:
                            decoded_text = ""

            return {
                "role": "assistant",
                "request_id": request_id,
                "content": decoded_text,
                "tool_calls": tool_calls,
                "tokens": [
                    TokenAndProb(
                        token_id=token_id,
                        logprob=logprob[token_id].logprob,    # Warning: vllm logprob does not participant training (not reliable enough), for log only.
                        decoded_string=logprob[token_id].decoded_token,
                    )
                    for token_id, logprob in zip(token_array, logprob_array)  # type: ignore
                ],
            }


        def llm_chat_remote(
            messages: List[Dict[str, str]],
            custom_sampling_params: dict = {},
            tools=[],
            request_id: str = "",
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
                    # this function is defined in `ajet/backbone/main_vllm.py`
                    output_message = self.async_rollout_manager.submit_chat_completions(
                        messages=input_messages,
                        sampling_params=updated_sampling_params,
                        tools=tools,
                        request_id=request_id,
                    )
                    break
                except Exception as e:
                    logger.bind(exception=True).exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)
            return output_message[-1]  # type: ignore


        def llm_chat_trinity(
            messages: List[Dict[str, str]],
            custom_sampling_params: dict = {},
            tools=[],
            request_id: str = "",
        ) -> dict:
            async def main():
                updated_sampling_params = {}
                if sampling_params:
                    updated_sampling_params.update(sampling_params)
                if custom_sampling_params:
                    updated_sampling_params.update(custom_sampling_params)
                updated_sampling_params.pop("min_tokens")

                if tools:
                    response = await self.async_rollout_manager.chat.completions.create(
                        model=self.async_rollout_manager.model_path,
                        messages=messages,
                        logprobs=True,
                        tools=tools,
                        top_logprobs=0,
                        **updated_sampling_params,
                    )
                else:
                    response = await self.async_rollout_manager.chat.completions.create(
                        model=self.async_rollout_manager.model_path,
                        messages=messages,
                        logprobs=True,
                        top_logprobs=0,
                        **updated_sampling_params,
                    )
                return response

            response = run_async_coroutine_with_timeout(main(), timeout=1800)  # type: ignore
            prompt_text = self.tokenizer.decode(response.model_extra["prompt_token_ids"])
            prompt_token_ids = response.model_extra["prompt_token_ids"]
            content = response.choices[0].message.content
            message = response.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)

            if content is None:
                content = ""

            if ("<tool_call>" in content) and (not message.get("tool_calls", None)):
                # logger.bind(exception=True).exception(f"Bad toolcall discovered \n\nprompt_text:\n{prompt_text}\n\nrepsonse:\n{content}")
                logger.warning(f"Bad toolcall discovered: {content}")

            return {
                "role": "assistant",
                "request_id": response.id,
                "content": content,
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt_token_ids,
                "tool_calls": message.get("tool_calls", []),
                "tokens": [
                    TokenAndProb(
                        token_id=token,
                        logprob=tokenlogprob.logprob, # Warning: vllm logprob does not participant training, for log only.
                        decoded_string=tokenlogprob.token,
                    )
                    for tokenlogprob, token in zip(
                        response.choices[0].logprobs.content,
                        response.choices[0].token_ids,
                    )
                ],
            }

        if self.llm_mode == "remote":
            return llm_chat_remote
        if self.llm_mode == "trinity":
            return llm_chat_trinity
        else:
            return llm_chat_verl




# ----------------------------------------------------------------------------------------------
# ------------------------ call async llm with context tracker (OpenAI) ------------------------
# ----------------------------------------------------------------------------------------------

class OpenaiLlmProxyWithTracker(object):
    """
    An essential wrapper to connect AsyncLlmBridge with AgentScope

    User_user_workflow <-> AsyncLlmBridge <-> Context Tracker.
    """

    def __init__(
        self,
        llm_inference_fn: Callable, # Callable[AjetStandardLlmBridgeRequest, AjetStandardLlmBridgeResponse]
        context_tracker: MultiAgentContextTracker,
        config,
    ) -> None:
        self.context_tracker = context_tracker
        self.llm_inference_fn = llm_inference_fn
        self.config = config


    async def __call__(
        self,
        messages: List[dict],
        tools: List = [],
        tool_choice: str = "auto",
        structured_model=None,
        **kwargs,
    ) -> ChatResponse:
        llm_output = await self.run_infer(messages, tools, tool_choice, structured_model, **kwargs)
        return convert_llm_proxy_response_to_oai_response(llm_output)


    async def run_infer(
        self,
        messages: List[dict],
        tools: List = [],
        tool_choice: str = "auto",      # always auto
        structured_model=None,          # this is for AgentScope only
        **kwargs,
    ):
        # generate timeline uuid
        timeline_uuid = uuid.uuid4().hex

        # prepare context tracker, check context safety
        (
            context_safe,
            token_overflow,
            info,
            converted_message,
            custom_sampling_params,
            tools,
        ) = self.context_tracker.step_prepare(messages, tools, timeline_uuid=timeline_uuid)

        # if context not safe to infer further
        if not context_safe:
            logger.warning(f"[{info}] detected.")
            self.context_tracker.context_overflow = True
            if token_overflow:
                # ajet_action_when_overflow = self.config.ajet.rollout.ajet_action_when_overflow
                # cannot proceed due to context overflow
                return self.construct_overflow_response()
            # else:
            #     otherwise, for abnormal output, can still proceed, but we do not track output anymore

        # run llm inference ✨
        llm_output = await asyncio.wait_for(
            asyncio.to_thread(
                self.llm_inference_fn, converted_message, custom_sampling_params, tools
            ),
            timeout=1800,
        )

        # begin context tracking
        self.context_tracker.step_track(llm_output, context_safe, converted_message, tools, timeline_uuid=timeline_uuid)
        return llm_output


    def construct_overflow_response(self):
        return {
            "role": "assistant",
            "request_id": "overflow_response",
            "content": "ajet_proxy: Exceeded max model context length.",
            "tool_calls": None,
            "tokens": [],
        }






# ----------------------------------------------------------------------------------------------
# ------------------------ call async llm with context tracker (AgentScope) --------------------
# ----------------------------------------------------------------------------------------------

class AgentScopeLlmProxyWithTracker(OpenaiLlmProxyWithTracker):

    async def __call__(
        self,
        messages: List[dict],
        tools: List = [],
        tool_choice: str = "auto",
        structured_model=None,
        **kwargs,
    ) -> AgentScopeChatResponse:

        llm_output = await self.run_infer(messages, tools, tool_choice, structured_model)
        response = convert_llm_proxy_response_to_agentscope_response(llm_output, structured_model=structured_model)
        return response
