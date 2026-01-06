# EXPERIMENTAL:
# This is a global store for cross-machine tuner request proxy
# unless your workflow have to be executed across multiple machines
# you probably don't need this at all.

global CROSS_MACHINE_TUNER_REQUEST_PROXY_STORE
CROSS_MACHINE_TUNER_REQUEST_PROXY_STORE = dict()

import asyncio
from typing import Optional, Union, Annotated
from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from http import HTTPStatus
from vllm.entrypoints.openai.protocol import (CompletionLogProbs,
                                              CompletionRequest,
                                              CompletionResponse,
                                              CompletionResponseChoice,
                                              CompletionResponseStreamChoice,
                                              CompletionStreamResponse,
                                              ErrorResponse,
                                              PromptTokenUsageInfo,
                                              RequestResponseMetadata,
                                              UsageInfo)
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.entrypoints.openai.serving_engine import (
    TextTokensPrompt,
    EmbedsPrompt,
    AnyRequest,
)

router = APIRouter()

async def _normalize_prompt_text_to_input(
    request: AnyRequest,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]],
    add_special_tokens: bool,
) -> TextTokensPrompt:

    def _get_async_tokenizer(tokenizer: PreTrainedTokenizer):
        async def async_tokenizer(
            prompt: str,
            add_special_tokens: bool = True,
            truncation: bool = False,
        ):
            return await asyncio.to_thread(
                tokenizer.__call__,
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
            )
        return async_tokenizer

    assert truncate_prompt_tokens is None
    async_tokenizer = _get_async_tokenizer(tokenizer)
    encoded = await async_tokenizer(prompt, add_special_tokens=add_special_tokens)

    input_ids = encoded.input_ids
    input_text = prompt

    return self._validate_input(request, input_ids, input_text)


async def _tokenize_prompt_input_or_inputs_async(
    request: AnyRequest,
    tokenizer: PreTrainedTokenizer,
    input_or_inputs: Optional[Union[str, list[str], list[int],
                                    list[list[int]]]],
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None,
    add_special_tokens: bool = True,
) -> tuple[list[TextTokensPrompt], list[EmbedsPrompt]]:

    inputs_text = list[TextTokensPrompt]()

    # Parse and batch the input prompts
    batch_inputs = parse_and_batch_prompt(input_or_inputs)

    # Process each input in the batch concurrently
    tasks = []
    for prompt_input in batch_inputs:
        assert not prompt_input["is_tokens"]
        task = _normalize_prompt_text_to_input(
            request,
            tokenizer,
            prompt_input["content"],
            truncate_prompt_tokens=truncate_prompt_tokens,
            add_special_tokens=add_special_tokens)

        tasks.append(task)

    # Wait for all tokenization tasks to complete
    results = await asyncio.gather(*tasks)
    inputs_text.extend(results)

    return inputs_text, []

async def _preprocess_completion(
    request,
    tokenizer,
    input_or_inputs: Optional[Union[str, list[str], list[int],
                                    list[list[int]]]],
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None,
    add_special_tokens: bool = True,
):

    (request_prompts_text, request_prompts_embeds
        ) = await _tokenize_prompt_input_or_inputs_async(
            request,
            tokenizer,
            input_or_inputs,
            truncate_prompt_tokens=truncate_prompt_tokens,
            add_special_tokens=add_special_tokens,
        )

    request_prompts = request_prompts_text
    return request_prompts, None

async def create_completion(
    request: CompletionRequest,
    raw_request: Optional[Request] = None,
) -> Union[AsyncGenerator[str, None], CompletionResponse, ErrorResponse]:

    tokenizer = await self.engine_client.get_tokenizer(lora_request)

    request_prompts, engine_prompts = await _preprocess_completion(
        request,
        tokenizer,
        request.prompt,
        truncate_prompt_tokens=request.truncate_prompt_tokens,
        add_special_tokens=request.add_special_tokens,
    )
    return request_prompts



@router.post("/v1/completions",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.OK.value: {
                     "content": {
                         "text/event-stream": {}
                     }
                 },
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.NOT_FOUND.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
async def create_completion(request: CompletionRequest, raw_request: Request):
    """The idea is simple: parse request, get apply chat template and tokenize

    Args:
        request (CompletionRequest): _description_
        raw_request (Request): _description_

    Returns:
        _type_: _description_
    """


    # from api key or url, read `agent_name`, `uuid` etc

    session_uuid = ...
    CROSS_MACHINE_TUNER_REQUEST_PROXY_STORE[session_uuid] = request

    # convert to prompt
    request_prompts = await create_completion(request, raw_request)


    return StreamingResponse(content=generator, media_type="text/event-stream")
