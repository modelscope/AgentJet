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

from loguru import logger
from pydantic import BaseModel
from functools import lru_cache
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from multiprocessing import Manager, Process
from concurrent.futures import ThreadPoolExecutor
from typing import Coroutine, Optional, Tuple

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from openai.types.chat.chat_completion import ChatCompletion, CompletionUsage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

from ajet.utils.networking import get_host_ip
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
        socket.setsockopt(zmq.RCVTIMEO, 6*1000)  # 6 second recv timeout
        socket.connect(f"{episode_address}")
        if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | connect done")

        # <send to>
        #   <to_sourcefile>: ajet/tuner_lib/experimental/oai_model_client.py
        #   <to_code>: message = self.socket.recv_string()
        socket.send_string(int_req.model_dump_json())

        if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | send_string")

        result_str = ""
        for _ in range(50):  # max 5 minutes wait

            if enable_swarm_mode:
                assert shared_mem_dict is not None
                ep_stat = shared_mem_dict[ep_key(episode_uuid)]
                episode_status = ep_stat.episode_status
                if episode_status != "claimed":
                    raise HTTPException(status_code=404, detail="The episode is not claimed, cannot accept new requests.")

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
            raise RuntimeError(f"Failed to get response from episode_address: {episode_address} after 5 attempts.")
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
            usage=CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=""),
                    finish_reason='stop' if tool_calls is None else 'tool_calls',
                )
            ]
        )
        yield dump_chunk(final_chunk)
        yield "data: [DONE]\n\n"


    @app.get("/health")
    async def health():
        return {"status": "ok"}


    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: str = Header(None)):
        """
        OpenAI-compatible chat completions endpoint.
        Receives ChatCompletionRequest and returns ChatCompletion.
        """

        # Parse authorization header (base64 encoded JSON)
        if not authorization:
            return HTTPException(status_code=401, detail="Missing authorization header")

        try:
            # Remove "Bearer " prefix if present
            auth_token = authorization.replace("Bearer ", "").replace("bearer ", "").replace(API_KEY_PREFIX, "")
            decoded = base64.b64decode(auth_token).decode('utf-8')
            auth_data = json.loads(decoded)

            agent_name = auth_data.get("agent_name")
            target_tag = auth_data.get("target_tag")
            episode_uuid = auth_data.get("episode_uuid")
            episode_address = auth_data.get("episode_address")

            if not all([agent_name, target_tag, episode_uuid]):
                return HTTPException(status_code=401, detail="Invalid authorization data")
        except Exception as e:
            return HTTPException(status_code=401, detail=f"Invalid authorization header: {str(e)}")

        if VERBOSE: logger.info(f"Running [{episode_uuid}]: /v1/chat/completions")

        # Parse request body
        body = await request.json()
        new_req = ChatCompletionRequest.model_validate(body)

        # Create timeline UUID
        timeline_uuid = uuid.uuid4().hex

        # if training, ignore all sampling parameters from request
        preserve_sampling_params = False

        # enable_swarm_mode
        if enable_swarm_mode:
            from ajet.tuner_lib.experimental.swarm_server import ep_key
            assert shared_mem_dict is not None
            assert shared_mem_dict_lock is not None

            if shared_mem_dict['engine_status'] not in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]:
                logger.error(f"The server is not in ENGINE.ROLLING status (current status: [{shared_mem_dict['engine_status']}]), cannot accept new requests.")
                raise HTTPException(status_code=404, detail="The server is not in ENGINE.ROLLING status, cannot accept new requests.")

            if ep_key(episode_uuid) not in shared_mem_dict:
                raise HTTPException(status_code=404, detail=f"Episode {episode_uuid} not found.")

            # update activate timestamp and increment llm call counter
            with shared_mem_dict_lock:
                es:EpisodeStatus = shared_mem_dict[ep_key(episode_uuid)]
                es.latest_activity_timestamp = time.time()
                es.llm_call_count += 1
                shared_mem_dict[ep_key(episode_uuid)] = es
            if es.episode_type == "eval":
                preserve_sampling_params = True

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
        result = await loop.run_in_executor(request.app.state.executor, _begin_handle_chat_completion, episode_address, int_req, episode_uuid)

        if enable_swarm_mode:
            assert shared_mem_dict is not None
            shared_mem_dict["latest_llm_call"] = {
                "input": body,
                "output": result,
            }

        if original_stream:
            result.model = "unknown_model" if not new_req.model else new_req.model
            return StreamingResponse(mock_as_stream_response(result), media_type="text/event-stream")

        return result


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













class InterchangeServer(Process):
    def __init__(self, experiment_dir: str, port: int, num_fastapi_process: int = 2, max_fastapi_threads: int = 512, enable_swarm_mode=False):
        super().__init__()
        self.experiment_dir = experiment_dir
        self.port = port
        self.num_fastapi_process = num_fastapi_process
        self.max_fastapi_threads = max_fastapi_threads
        self.enable_swarm_mode = enable_swarm_mode

    def run(self):
        logger.info(f"Starting Interchange Server on port {self.port} with {self.num_fastapi_process} processes and {self.max_fastapi_threads} threads per process.")

        if self.enable_swarm_mode:
            manager = Manager()
            shared_mem_dict = manager.dict()
            shared_mem_dict_lock = manager.Lock()
        else:
            shared_mem_dict = None
            shared_mem_dict_lock = None

        app, additional_coro = get_app(self.max_fastapi_threads, self.enable_swarm_mode, shared_mem_dict, shared_mem_dict_lock)

        async def serve_with_monitor(additional_coro):
            # Start the server
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=self.port,
                log_level="error",
                workers=self.num_fastapi_process
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
