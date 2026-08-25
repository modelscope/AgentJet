"""
A shadow FastAPI server for serving as interchange endpoint between Tuner and Workflow.

- This functionality is experimental.
- The code is very async, considering extreme efficiency for handling many concurrent requests,
  therefore, it may be hard to read.

---------------------------------------------------------------------------------------------

"""

import asyncio
import threading
import uuid
import time

import base64
import json
import os
import zmq
import uvicorn
import atexit
import httpx
from datetime import datetime

from loguru import logger
from pydantic import BaseModel
from functools import lru_cache
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from multiprocessing import Manager, Process
from concurrent.futures import ThreadPoolExecutor
from typing import Coroutine, Optional, Tuple

try:
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
except ModuleNotFoundError:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

from ajet.utils.networking import get_host_ip
from ajet.utils.message_utils import log_empty_content_messages
from ajet.tuner_lib.experimental.interchange_utils import EpisodeStatus
from ajet.tuner_lib.experimental.interchange_utils import DEBUG, VERBOSE, API_KEY_PREFIX


class InterchangeCompletionRequest(BaseModel):
    completion_request: ChatCompletionRequest
    agent_name: str
    target_tag: str
    episode_uuid: str
    timeline_uuid: str
    preserve_sampling_params: bool = False

class HealthCheckRequest(BaseModel):
    agent_name: str
    target_tag: str
    episode_uuid: str
    timeline_uuid: str
    health_check: bool = True

# Create FastAPI app
SERVER_SHUTDOWN_EVENT = threading.Event()

context = zmq.Context()
atexit.register(context.term)

@lru_cache(maxsize=128)
def ep_key(episode_uuid: str) -> str:
    return f"episodes-{episode_uuid}"

# ---------------------------------------------------------------------------
# Streaming keepalive for the non-incremental SSE proxy
# ---------------------------------------------------------------------------
#
# The ZMQ worker pipeline (oai_model_client.py) is NON-incremental: each request
# returns exactly one complete ChatCompletion, only AFTER the full generation
# finishes (see the 20-minute recv budget in _begin_handle_chat_completion). The
# three HTTP endpoints translate that single result into OpenAI / Anthropic SSE
# event sequences. The naive version -- await the worker, THEN return a
# StreamingResponse -- sends no bytes to the client for the entire generation, so
# streaming harnesses (Claude Code over /v1/messages, Codex over /v1/responses)
# trip their stream-inactivity / first-byte timeout and abort. The worker cannot
# stream real tokens without re-architecting the rollout, so instead we open the
# SSE response immediately and hold it open with periodic keepalive frames until
# the result is ready. SSE comment lines (": ...") and the Anthropic `ping` event
# are ignored by every compliant SSE parser, so they keep the connection alive
# without corrupting the event stream.
STREAM_KEEPALIVE_INTERVAL_SEC = 5.0


def _reqmon(line: str) -> None:
    """Request-lifecycle monitoring log (second-precision wall clock inline,
    because the project's loguru format only carries HH:MM)."""
    logger.info(f"[reqmon {datetime.now().strftime('%H:%M:%S')}] {line}")


async def _drain_result_with_keepalives(executor, fn, fn_args, keepalive_sse, log_tag: str = ""):
    """Run ``fn(*fn_args)`` in ``executor``; emit ``keepalive_sse`` immediately and
    every ``STREAM_KEEPALIVE_INTERVAL_SEC`` while it is pending, then yield the
    ChatCompletion result as the final item.

    Yields ``keepalive_sse`` (str) one or more times, then a single ChatCompletion.
    If the background call raises after the stream has already opened (so we can no
    longer return a clean HTTP error), the failure is logged and the generator ends
    -- the caller treats this as "drained without a result" and closes the stream.

    ``asyncio.shield`` lets the executor thread run to completion if the client
    disconnects, matching the pre-existing behaviour (the ZMQ recv cannot be
    cancelled mid-flight).
    """
    loop = asyncio.get_running_loop()
    t0 = time.time()
    pings = 0
    finished = False
    fut = loop.run_in_executor(executor, fn, *fn_args)
    # Immediate first byte so short time-to-first-byte timeouts are satisfied.
    yield keepalive_sse
    pings += 1
    while True:
        try:
            result = await asyncio.wait_for(asyncio.shield(fut), timeout=STREAM_KEEPALIVE_INTERVAL_SEC)
            if log_tag:
                _reqmon(f"GEN_DONE {log_tag} dur={time.time()-t0:.1f}s pings={pings}")
            finished = True
            yield result
            return
        except asyncio.TimeoutError:
            yield keepalive_sse
            pings += 1
        except Exception as e:  # pragma: no cover - rare mid-stream worker failure
            if log_tag:
                _reqmon(f"GEN_ERR {log_tag} dur={time.time()-t0:.1f}s pings={pings} err={e!r}")
            logger.exception(f"[stream-proxy] background LLM call failed: {e!r}; closing SSE stream early.")
            return
        except (GeneratorExit, asyncio.CancelledError):
            # Client (or a middle layer) closed the connection while the
            # generation was still running. The shielded executor future keeps
            # generating server-side -- this is exactly the ghost-sample path.
            if log_tag and not finished:
                # `finished=True` here is the post-success finalization of this
                # generator (consumer broke out of the async-for after the
                # result; GC's aclose() throws GeneratorExit at the yield) --
                # NOT a client disconnect. Only an unfinished generator that
                # gets closed/cancelled is a real CLIENT_GONE.
                _reqmon(f"CLIENT_GONE {log_tag} dur={time.time()-t0:.1f}s pings={pings}")
            raise

def get_app(max_fastapi_threads: int = 512, enable_swarm_mode=False, shared_mem_dict=None, shared_mem_dict_lock=None) -> Tuple[FastAPI, Optional[Coroutine]]:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        SERVER_SHUTDOWN_EVENT.clear()
        app.state.executor = ThreadPoolExecutor(max_workers=max_fastapi_threads)
        yield
        # Shutdown
        SERVER_SHUTDOWN_EVENT.set()
        app.state.executor.shutdown(wait=False, cancel_futures=True)


    app = FastAPI(title="AJet Interchange Endpoint", lifespan=lifespan)


    def _begin_handle_chat_completion(episode_address, int_req: InterchangeCompletionRequest, episode_uuid):
        """ run this in thread to avoid blocking main event loop
        """
        if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | Received new chat completion request (inside thread)")

        socket = context.socket(zmq.REQ)
        timeout_recv_ms = 6*1000 # 6 second recv timeout
        socket.setsockopt(zmq.RCVTIMEO, timeout_recv_ms)
        socket.connect(f"{episode_address}")
        if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | connect done")

        # <send to>
        #   <to_sourcefile>: ajet/tuner_lib/experimental/oai_model_client.py
        #   <to_code>: message = self.socket.recv_string()
        socket.send_string(int_req.model_dump_json())

        if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | send_string")

        result_str = ""
        timeout_sec = 3600 * 1000   # max 1 hour wait = 3600 * 1000 ms
        for _ in range(timeout_sec//timeout_recv_ms):

            if enable_swarm_mode:
                assert shared_mem_dict is not None
                assert shared_mem_dict_lock is not None
                if ep_key(episode_uuid) not in shared_mem_dict:
                    raise HTTPException(status_code=404, detail=f"Episode {episode_uuid} not found.")

                # update activate timestamp and increment llm call counter
                with shared_mem_dict_lock:
                    es:EpisodeStatus = shared_mem_dict[ep_key(episode_uuid)]
                    es.latest_activity_timestamp = time.time()
                    episode_status = es.episode_status
                    shared_mem_dict[ep_key(episode_uuid)] = es

                if episode_status != "claimed":
                    raise HTTPException(status_code=404, detail=f"The episode {episode_uuid} is not claimed, cannot accept new requests.")

            try:
                if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | recv_string begin.")

                # <wait for>:
                #   <from_sourcefile>: ajet/tuner_lib/experimental/oai_model_client.py
                #   <from_code>: self.socket.send_string(result)
                #   <expect>: ChatCompletion object in JSON string format
                result_str = socket.recv_string()

                break
            except zmq.Again as e:
                # check whether server is still in rolling status
                if enable_swarm_mode:
                    assert shared_mem_dict is not None
                    if shared_mem_dict['engine_status'] not in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]:
                        raise HTTPException(status_code=404, detail="The server is not in ENGINE.ROLLING status, cannot accept new requests.")

                if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | recv_string timeout, retrying.")
                continue

        if not result_str:
            raise RuntimeError(f"Failed to get response from episode_address: {episode_address} after {timeout_sec // 1000} seconds, consider decrease `max_response_length_in_one_turn`.")
        else:
            if DEBUG: logger.success(f"[server] episode_uuid: {episode_uuid} | recv_string done.")
        result_object = ChatCompletion(**json.loads(result_str))
        return result_object


    async def mock_as_stream_response(result: ChatCompletion):
        """
        Convert a non-streaming ChatCompletion result to streaming format.

        Args:
            result: ChatCompletion object to convert to streaming format

        Yields:
            Server-sent events formatted as streaming chat completion chunks
        """
        content = result.choices[0].message.content if result.choices else ""
        role = result.choices[0].message.role if result.choices else "assistant"
        result_id = result.id if result.id else uuid.uuid4().hex
        result.id = "chatcmpl-" + result_id if not result_id.startswith("chatcmpl-") else result_id
        # try:
        #     thinking = result.choices[0].message.reasoning_content
        # except:
        #     thinking = None
        tool_calls = result.choices[0].message.tool_calls if result.choices and result.choices[0].message.tool_calls else None
        delta_tool_calls = [] # tool_calls: Optional[List[ChoiceDeltaToolCall]] = None
        finish_reason = result.choices[0].finish_reason
        usage = result.usage
        if tool_calls:
            delta_tool_calls = [ChoiceDeltaToolCall(
                index=index,
                id=tc.id,
                function=ChoiceDeltaToolCallFunction(
                    name = tc.function.name,
                    arguments = tc.function.arguments,
                ),
                type=tc.type
            ) for index, tc in enumerate(tool_calls)]

        def dump_chunk(chunk: ChatCompletionChunk) -> str:
            dump = chunk.model_dump()
            dump.pop("service_tier", None)
            dump.pop("system_fingerprint", None)
            if "usage" in dump and dump["usage"] is None:
                dump.pop("usage", None)
            # for each choice delta, if field (such as tool_calls) is empty, remove it from the delta to avoid confusion
            for key in list(dump["choices"][0]["delta"].keys()):
                if not dump["choices"][0]["delta"][key] and key != "content":  # keep content even if it's empty
                    dump["choices"][0]["delta"].pop(key, None)
            return f"data: {json.dumps(dump)}\n\n"

        # First chunk with role
        first_chunk = ChatCompletionChunk(
            id=result.id,
            model=result.model,
            created=result.created,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role=role, content=""),
                    finish_reason=None
                )
            ]
        )
        yield dump_chunk(first_chunk)

        # Content chunk
        content_chunk = ChatCompletionChunk(
            id=result.id,
            model=result.model,
            created=result.created,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=content, tool_calls=delta_tool_calls),
                    finish_reason=None
                )
            ]
        )
        yield dump_chunk(content_chunk)
        # Final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id=result.id,
            model=result.model,
            created=result.created,
            object="chat.completion.chunk",
            usage=usage,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=""),
                    finish_reason=finish_reason,
                )
            ]
        )
        yield dump_chunk(final_chunk)
        yield "data: [DONE]\n\n"


    @app.get("/health")
    async def health():
        return {"status": "ok"}


    def _parse_authorization_header(authorization):
        """Parse the AgentJet authorization header.

        Returns (agent_name, target_tag, episode_uuid, episode_address).
        Raises HTTPException on any failure so callers can `return` it directly.
        """
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing authorization header")

        try:
            auth_token = authorization.replace("Bearer ", "").replace("bearer ", "").replace(API_KEY_PREFIX, "")
            decoded = base64.b64decode(auth_token).decode('utf-8')
            auth_data = json.loads(decoded)

            agent_name = auth_data.get("agent_name")
            target_tag = auth_data.get("target_tag")
            episode_uuid = auth_data.get("episode_uuid")
            episode_address = auth_data.get("episode_address")

            if not all([agent_name, target_tag, episode_uuid]):
                raise HTTPException(status_code=401, detail="Invalid authorization data")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid authorization header: {str(e)}")

        return agent_name, target_tag, episode_uuid, episode_address

    def _check_swarm_episode_and_refresh(episode_uuid: str):
        """Verify the engine is ROLLING and the episode is claimed; refresh activity.

        Returns `preserve_sampling_params` (True iff the episode is an eval episode).
        Mirrors the body of the same name that lived inline in `chat_completions`,
        so the behaviour of both endpoints stays identical.
        """
        from ajet.tuner_lib.experimental.swarm_server import ep_key
        from ajet.tuner_lib.experimental.interchange_utils import _refresh_client_activity
        assert shared_mem_dict is not None
        assert shared_mem_dict_lock is not None

        if shared_mem_dict['engine_status'] not in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]:
            logger.error(f"The server is not in ENGINE.ROLLING status (current status: [{shared_mem_dict['engine_status']}]), cannot accept new requests.")
            raise HTTPException(status_code=404, detail="The server is not in ENGINE.ROLLING status, cannot accept new requests.")

        if ep_key(episode_uuid) not in shared_mem_dict:
            raise HTTPException(status_code=404, detail=f"Episode {episode_uuid} not found.")

        preserve_sampling_params = False
        with shared_mem_dict_lock:
            es: EpisodeStatus = shared_mem_dict[ep_key(episode_uuid)]
            es.latest_activity_timestamp = time.time()
            es.llm_call_count += 1
            shared_mem_dict[ep_key(episode_uuid)] = es
        if es.episode_type == "eval":
            preserve_sampling_params = True
        # An LLM call counts as activity for keeping the owning client in the
        # swarm-server active list (no-op if it's not active yet).
        _refresh_client_activity(es.client_uuid, shared_mem_dict)
        return preserve_sampling_params

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: str = Header(None)):
        """
        OpenAI-compatible chat completions endpoint.
        Receives ChatCompletionRequest and returns ChatCompletion.
        """

        # Parse authorization header (base64 encoded JSON)
        agent_name, target_tag, episode_uuid, episode_address = _parse_authorization_header(authorization)

        if VERBOSE: logger.info(f"Running [{episode_uuid}]: /v1/chat/completions")

        # Parse request body
        body = await request.json()
        new_req = ChatCompletionRequest.model_validate(body)

        # Check if the first message is a system message, if not, add a default one
        if new_req.messages:
            first_msg = new_req.messages[0]
            if first_msg.get("role") != "system":
                logger.warning(f"First message role is '{first_msg.get('role')}', expected 'system'. Adding default system prompt.")
                new_req.messages.insert(0, {"role": "system", "content": "You are a helpful assistant, your name is AgentJet."})

        # Detect empty-content messages in the inbound request
        log_empty_content_messages(new_req.messages, episode_uuid=episode_uuid)

        # Create timeline UUID
        timeline_uuid = uuid.uuid4().hex

        # if training, ignore all sampling parameters from request
        preserve_sampling_params = False

        # enable_swarm_mode
        if enable_swarm_mode:
            preserve_sampling_params = _check_swarm_episode_and_refresh(episode_uuid)

        # For streaming, we process as non-streaming but return in streaming format
        original_stream = new_req.stream
        if original_stream:
            new_req.stream = False
            new_req.stream_options = None

        # Add to received queue
        int_req = InterchangeCompletionRequest(
            completion_request = new_req,
            agent_name = agent_name,
            target_tag = target_tag,
            episode_uuid = episode_uuid,
            timeline_uuid = timeline_uuid,
            preserve_sampling_params = preserve_sampling_params,
        )
        if DEBUG: logger.info(f"episode_uuid: {episode_uuid} | Received new chat completion request (outside thread)")
        loop = asyncio.get_running_loop()
        executor = request.app.state.executor

        if original_stream:
            # Open the SSE stream IMMEDIATELY and hold it open with comment
            # keepalives while the non-incremental worker computes the full
            # response (see _drain_result_with_keepalives). Awaiting the worker
            # before opening the stream would send no bytes for the whole
            # generation and trip streaming clients' first-byte / inactivity timeout.
            async def _stream_chat_completions():
                result = None
                async for item in _drain_result_with_keepalives(
                    executor, _begin_handle_chat_completion,
                    (episode_address, int_req, episode_uuid), ": keepalive\n\n", log_tag=f"ep={episode_uuid} tl={timeline_uuid} endpoint=/v1/chat/completions",
                ):
                    if isinstance(item, ChatCompletion):
                        result = item
                        break
                    yield item
                if result is None:
                    # close the open block so the client's parser ends sanely
                    yield _anthropic_sse("content_block_stop", {"index": 0})
                    return
                if enable_swarm_mode:
                    assert shared_mem_dict is not None
                    shared_mem_dict["latest_llm_call"] = {"input": body, "output": result}
                result.model = "unknown_model" if not new_req.model else new_req.model
                async for chunk in mock_as_stream_response(result):
                    yield chunk

            return StreamingResponse(
                _stream_chat_completions(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = await loop.run_in_executor(executor, _begin_handle_chat_completion, episode_address, int_req, episode_uuid)
        if enable_swarm_mode:
            assert shared_mem_dict is not None
            shared_mem_dict["latest_llm_call"] = {
                "input": body,
                "output": result,
            }
        return result


    @app.post("/v1/responses")
    async def responses(request: Request, authorization: str = Header(None)):
        """OpenAI Responses API endpoint (mirrors /v1/chat/completions behaviour).

        Translates ResponseCreateParams → ChatCompletionRequest, reuses the
        same ZMQ → worker → LLM pipeline, then converts the resulting
        ChatCompletion back into a Responses `Response` object. When the
        client asks for streaming (`stream: true`) the full Response is
        chunked into the SSE event sequence the OpenAI SDK expects.
        """
        from ajet.tuner_lib.experimental.oai_responses_adapter import (
            build_chat_completion_request,
            chat_completion_to_responses_dict,
            iter_responses_sse_events,
        )

        agent_name, target_tag, episode_uuid, episode_address = _parse_authorization_header(authorization)

        if VERBOSE: logger.info(f"Running [{episode_uuid}]: /v1/responses")

        body = await request.json()
        instructions = body.get("instructions") if isinstance(body, dict) else None

        new_req, original_stream = build_chat_completion_request(body)

        log_empty_content_messages(new_req.messages, episode_uuid=episode_uuid)

        timeline_uuid = uuid.uuid4().hex
        preserve_sampling_params = False

        if enable_swarm_mode:
            preserve_sampling_params = _check_swarm_episode_and_refresh(episode_uuid)

        # Always forward as non-streaming; the worker pipeline is non-incremental
        # and we synthesize Responses SSE events ourselves on the way back.
        new_req.stream = False
        new_req.stream_options = None

        int_req = InterchangeCompletionRequest(
            completion_request=new_req,
            agent_name=agent_name,
            target_tag=target_tag,
            episode_uuid=episode_uuid,
            timeline_uuid=timeline_uuid,
            preserve_sampling_params=preserve_sampling_params,
        )
        if DEBUG: logger.info(f"episode_uuid: {episode_uuid} | Received new responses request (outside thread)")
        loop = asyncio.get_running_loop()
        executor = request.app.state.executor
        model_name = body.get("model") if isinstance(body, dict) else None

        if original_stream:
            # Open the SSE stream immediately; comment keepalives hold it open
            # while the non-incremental worker computes the full response.
            async def _stream_responses():
                result = None
                async for item in _drain_result_with_keepalives(
                    executor, _begin_handle_chat_completion,
                    (episode_address, int_req, episode_uuid), ": keepalive\n\n", log_tag=f"ep={episode_uuid} tl={timeline_uuid} endpoint=/v1/responses",
                ):
                    if isinstance(item, ChatCompletion):
                        result = item
                        break
                    yield item
                if result is None:
                    return
                response_dict = chat_completion_to_responses_dict(
                    result,
                    model=model_name or result.model or "unknown",
                    instructions=instructions if isinstance(instructions, str) else None,
                )
                if enable_swarm_mode:
                    assert shared_mem_dict is not None
                    shared_mem_dict["latest_llm_call"] = {"input": body, "output": response_dict, "format": "responses"}
                for chunk in iter_responses_sse_events(response_dict):
                    yield chunk

            return StreamingResponse(
                _stream_responses(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result: ChatCompletion = await loop.run_in_executor(
            executor, _begin_handle_chat_completion, episode_address, int_req, episode_uuid
        )
        response_dict = chat_completion_to_responses_dict(
            result,
            model=model_name or result.model or "unknown",
            instructions=instructions if isinstance(instructions, str) else None,
        )

        if enable_swarm_mode:
            assert shared_mem_dict is not None
            shared_mem_dict["latest_llm_call"] = {
                "input": body,
                "output": response_dict,
                "format": "responses",
            }

        return response_dict


    @app.post("/v1/messages")
    async def messages(request: Request, authorization: str = Header(None), x_api_key: str = Header(None, alias="x-api-key")):
        """Anthropic Messages API endpoint (POST /v1/messages).

        Anthropic-compatible endpoint reusing the same ZMQ → worker → LLM
        pipeline as /v1/chat/completions and /v1/responses. The inbound
        Anthropic body is translated to a ChatCompletionRequest, the resulting
        ChatCompletion is translated back to an Anthropic Message dict (or
        Messages-style SSE events when streaming).

        The Anthropic SDK sends the API key as `x-api-key`, not `Authorization`;
        accept either header so both a stock SDK client and clients reusing the
        existing `authorization` convention route correctly.
        """
        from ajet.tuner_lib.experimental.anthropic_messages_adapter import (
            build_chat_completion_request,
            chat_completion_to_message_dict,
            iter_anthropic_sse_events,
        )

        token = authorization if authorization is not None else x_api_key
        agent_name, target_tag, episode_uuid, episode_address = _parse_authorization_header(token)

        if VERBOSE: logger.info(f"Running [{episode_uuid}]: /v1/messages")

        try:
            body = await request.json()
        except Exception as e:
            _reqmon(f"BODY_FAIL ep={episode_uuid} endpoint=/v1/messages err={type(e).__name__}:{str(e)[:120]}")
            raise

        new_req, original_stream = build_chat_completion_request(body)

        log_empty_content_messages(new_req.messages, episode_uuid=episode_uuid)

        timeline_uuid = uuid.uuid4().hex
        preserve_sampling_params = False

        if enable_swarm_mode:
            preserve_sampling_params = _check_swarm_episode_and_refresh(episode_uuid)

        # Always forward as non-streaming; the worker pipeline is non-incremental
        # and we synthesize Messages SSE events ourselves on the way back.
        new_req.stream = False
        new_req.stream_options = None
        _reqmon(f"REQ_START ep={episode_uuid} tl={timeline_uuid} endpoint=/v1/messages "
                f"stream={int(bool(original_stream))} msgs={len(new_req.messages)} "
                f"max_tokens={new_req.max_tokens}")

        int_req = InterchangeCompletionRequest(
            completion_request=new_req,
            agent_name=agent_name,
            target_tag=target_tag,
            episode_uuid=episode_uuid,
            timeline_uuid=timeline_uuid,
            preserve_sampling_params=preserve_sampling_params,
        )
        if DEBUG: logger.info(f"episode_uuid: {episode_uuid} | Received new messages request (outside thread)")
        loop = asyncio.get_running_loop()
        executor = request.app.state.executor
        model_name = body.get("model") if isinstance(body, dict) else None

        if original_stream:
            async def _stream_messages():
                result = None
                # claude-code (Bun binary 2.1.193) runs a byte-stream idle
                # watchdog (~300s) on /v1/messages streams that is NOT reset
                # by `event: ping` frames -- only by content_block_delta
                # traffic. A generation longer than 300s with ping-only
                # keepalives gets aborted and re-issued by the client
                # (ghost-sample bug, 2026-08-19). Feed the watchdog instead:
                # open the message and an empty text block immediately, then
                # heartbeat with EMPTY content_block_delta events (verified
                # end-to-end: a 320s stream with these heartbeats is
                # delivered and assembled correctly).
                from ajet.tuner_lib.experimental.anthropic_messages_adapter import _sse as _anthropic_sse
                yield _anthropic_sse(
                    "message_start",
                    {
                        "message": {
                            "id": timeline_uuid,
                            "type": "message",
                            "role": "assistant",
                            "model": model_name or "unknown",
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        }
                    },
                )
                yield _anthropic_sse(
                    "content_block_start",
                    {"index": 0, "content_block": {"type": "text", "text": ""}},
                )
                async for item in _drain_result_with_keepalives(
                    executor, _begin_handle_chat_completion,
                    (episode_address, int_req, episode_uuid),
                    # ping keeps generic SSE clients alive; the EMPTY text
                    # delta resets claude-code's byte-stream watchdog.
                    ('event: ping\ndata: {"type": "ping"}\n\n'
                     'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": ""}}\n\n'),
                    log_tag=f"ep={episode_uuid} tl={timeline_uuid} endpoint=/v1/messages",
                ):
                    if isinstance(item, ChatCompletion):
                        result = item
                        break
                    yield item
                if result is None:
                    return
                message_dict = chat_completion_to_message_dict(
                    result,
                    model=model_name or result.model or "unknown",
                )
                if enable_swarm_mode:
                    assert shared_mem_dict is not None
                    shared_mem_dict["latest_llm_call"] = {"input": body, "output": message_dict, "format": "messages"}
                for chunk in iter_anthropic_sse_events(message_dict, prologue_already_sent=True):
                    yield chunk
                _reqmon(f"RESP_SENT ep={episode_uuid} tl={timeline_uuid} endpoint=/v1/messages")

            return StreamingResponse(
                _stream_messages(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result: ChatCompletion = await loop.run_in_executor(
            executor, _begin_handle_chat_completion, episode_address, int_req, episode_uuid
        )
        message_dict = chat_completion_to_message_dict(
            result,
            model=model_name or result.model or "unknown",
        )

        if enable_swarm_mode:
            assert shared_mem_dict is not None
            shared_mem_dict["latest_llm_call"] = {
                "input": body,
                "output": message_dict,
                "format": "messages",
            }

        return message_dict


    if enable_swarm_mode:
        from ajet.tuner_lib.experimental.swarm_server import register_enable_swarm_mode_routes

        @app.post("/replay_latest_llm_call")
        async def replay_latest_llm_call():
            """Return the buffered latest LLM call result."""
            assert shared_mem_dict is not None
            if ("latest_llm_call" not in shared_mem_dict) or shared_mem_dict["latest_llm_call"] is None:
                raise HTTPException(status_code=404, detail="No LLM call has been made yet")
            return shared_mem_dict["latest_llm_call"]

        assert shared_mem_dict is not None, "shared_mem_dict must not be None when enable_swarm_mode is True."
        assert shared_mem_dict_lock is not None, "shared_mem_dict_lock must not be None when enable_swarm_mode is True."
        app, additional_coro = register_enable_swarm_mode_routes(app, zmq_context=context, shared_mem_dict=shared_mem_dict, shared_mem_dict_lock=shared_mem_dict_lock)

    else:

        additional_coro = None


    return app, additional_coro













def _bind_reuseport_socket(host: str, port: int):
    import socket as _socket
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        logger.warning("SO_REUSEPORT is not supported on this platform; multi-process workers may conflict on bind.")
    sock.bind((host, port))
    return sock


def _run_fastapi_worker(port, max_fastapi_threads, enable_swarm_mode, shared_mem_dict, shared_mem_dict_lock):
    """Entry point for a FastAPI worker subprocess.

    Each worker binds its own socket with SO_REUSEPORT so the kernel load-balances
    accepted connections across all workers sharing the same (host, port).
    """
    sock = _bind_reuseport_socket("0.0.0.0", port)
    app, _ = get_app(max_fastapi_threads, enable_swarm_mode, shared_mem_dict, shared_mem_dict_lock)
    # [2026-08-25 恢复 log_level=error] 引擎退出后被 get_engine_status 轮询 access log
    # 刷屏冲掉现场 (scrollback 8000 行全是它)。默认 info+access_log 是 08-19 debug 时改的,
    # 提交 59d122d 前这里一直是 log_level="error"。排查时可临时改回 info+access_log=True。
    config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="error")
    server = uvicorn.Server(config)
    try:
        asyncio.run(server.serve(sockets=[sock]))
    except KeyboardInterrupt:
        SERVER_SHUTDOWN_EVENT.set()


class InterchangeServer(Process):
    def __init__(self, experiment_dir: str, port: int, num_fastapi_process: int = 1, max_fastapi_threads: int = 512, enable_swarm_mode=False):
        super().__init__()
        self.experiment_dir = experiment_dir
        self.port = port
        self.num_fastapi_process = num_fastapi_process
        self.max_fastapi_threads = max_fastapi_threads
        self.enable_swarm_mode = enable_swarm_mode

    def run(self):
        logger.info(f"Starting Interchange Server on port {self.port} with {self.num_fastapi_process} processes and {self.max_fastapi_threads} threads per process.")

        multi_process = self.num_fastapi_process > 1

        if self.enable_swarm_mode:
            if multi_process:
                # Cross-process sharing requires Manager proxies (one dedicated server
                # process arbitrates all reads/writes and lock acquire/release).
                manager = Manager()
                shared_mem_dict = manager.dict()
                shared_mem_dict_lock = manager.Lock()
            else:
                # Single-process: plain dict + threading.Lock avoids the manager IPC
                # roundtrip on every access.
                shared_mem_dict = {}
                shared_mem_dict_lock = threading.Lock()
        else:
            shared_mem_dict = None
            shared_mem_dict_lock = None

        if multi_process:
            # Build the app once in the supervisor to obtain the janitor coroutine
            # (additional_coro). The supervisor does not serve HTTP — it only runs
            # the janitor and watches the workers.
            _, additional_coro = get_app(self.max_fastapi_threads, self.enable_swarm_mode, shared_mem_dict, shared_mem_dict_lock)

            workers = []
            for _ in range(self.num_fastapi_process):
                p = Process(
                    target=_run_fastapi_worker,
                    args=(self.port, self.max_fastapi_threads, self.enable_swarm_mode, shared_mem_dict, shared_mem_dict_lock),
                    daemon=True,
                )
                p.start()
                workers.append(p)
            logger.info(f"Spawned {len(workers)} FastAPI worker processes: pids={[p.pid for p in workers]}")

            async def supervise():
                async def _watch_workers():
                    while True:
                        await asyncio.sleep(1)
                        for p in workers:
                            if p.exitcode is not None:
                                logger.error(f"FastAPI worker (pid={p.pid}) exited unexpectedly with code {p.exitcode}.")
                                return
                tasks = [asyncio.create_task(_watch_workers())]
                if additional_coro:
                    tasks.append(asyncio.create_task(additional_coro))
                await asyncio.gather(*tasks)

            try:
                asyncio.run(supervise())
            except KeyboardInterrupt as e:
                SERVER_SHUTDOWN_EVENT.set()
                raise e
            finally:
                for p in workers:
                    try:
                        p.terminate()
                    except Exception:
                        pass

        else:
            app, additional_coro = get_app(self.max_fastapi_threads, self.enable_swarm_mode, shared_mem_dict, shared_mem_dict_lock)

            async def serve_with_monitor(additional_coro):
                # Start the server
                config = uvicorn.Config(
                    app=app,
                    host="0.0.0.0",
                    port=self.port,
                    # [2026-08-25 恢复 log_level=error] 同上: 防轮询 access log 刷屏。
                    log_level="error",
                )
                server = uvicorn.Server(config)
                if additional_coro:
                    coro_task_1 = asyncio.create_task(additional_coro)
                    coro_task_2 = asyncio.create_task(server.serve())
                    await asyncio.gather(coro_task_1, coro_task_2)
                else:
                    await server.serve()
            try:
                asyncio.run(serve_with_monitor(additional_coro))
            except KeyboardInterrupt as e:
                SERVER_SHUTDOWN_EVENT.set()
                raise e














# Convenience function for quick server startup
def start_interchange_server(config, blocking=False, env={}) -> int:
    # Read config
    already_started = config.ajet.interchange_server.already_started
    experiment_dir = config.ajet.experiment_dir
    num_fastapi_process = config.ajet.interchange_server.num_fastapi_process
    max_fastapi_threads = config.ajet.interchange_server.max_fastapi_threads
    enable_swarm_mode = config.ajet.enable_swarm_mode

    # Find a free port if not specified or invalid
    port = int(os.environ.get("AJET_DAT_INTERCHANGE_PORT", -1))
    if config.ajet.interchange_server.interchange_server_port != 'auto':
        port = int(config.ajet.interchange_server.interchange_server_port)
        os.environ["AJET_DAT_INTERCHANGE_PORT"] = str(port)
    if port <= 0:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        os.environ["AJET_DAT_INTERCHANGE_PORT"] = str(port)

    # init interchage server sub-process
    if not already_started:
        # apply env vars
        os.environ.update(env)
        # start interchange server
        interchange_server = InterchangeServer(
            experiment_dir,
            port,
            num_fastapi_process,
            max_fastapi_threads,
            enable_swarm_mode,
        )
        interchange_server.start()
    else:
        interchange_server = None

    # Wait for server to be ready
    health_url = f"http://127.0.0.1:{port}/health"
    localhost_url = f"http://127.0.0.1:{port}"
    master_node_ip = get_host_ip(os.environ.get("NETWORK_INTERFACE", None))
    host_url = f"http://{master_node_ip}:{port}"
    os.environ["MASTER_NODE_IP"] = str(master_node_ip)

    # polling for server ready
    start_time = time.time()
    _httpx_client = httpx.Client(timeout=0.5)
    while True:
        if interchange_server and interchange_server.exitcode is not None:
            logger.error(f"Interchange server subprocess failed to start. Return code: {interchange_server.exitcode}")
            raise RuntimeError("Interchange server subprocess failed to start.")
        if time.time() - start_time > 30:
            msg = f"Interchange server subprocess failed to start within {time.time() - start_time} seconds."
            logger.error(msg)
            raise RuntimeError(msg)
        try:
            if _httpx_client.get(health_url).status_code == 200:
                break
        except Exception:
            # keep waiting
            pass
        time.sleep(1)

    # register a termination handler
    if interchange_server:
        if DEBUG: logger.info(f"Interchange server subprocess started on port {port} (pid: {interchange_server.pid})")
        atexit.register(lambda: interchange_server.terminate())

    if not blocking:
        # return port
        return port
    else:
        logger.success(f"Interchange server is running in blocking mode on:\n------\n"
                       f"URL 1: {localhost_url}\n------\n"
                       f"URL 2: {host_url}\n------\n"
                       f"Press Ctrl+C to stop.")
        try:
            if interchange_server:
                interchange_server.join()
        except KeyboardInterrupt:
            logger.info("Shutting down interchange server...")
            try: _httpx_client.post(f"http://127.0.0.1:{port}/stop_engine", timeout=8).status_code
            except Exception: pass

            if interchange_server:
                interchange_server.terminate()
            if enable_swarm_mode:
                from ajet.tuner_lib.experimental.swarm_server import kill_process_tree
                kill_process_tree(None, None)
        return -1
