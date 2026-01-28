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
from fastapi import FastAPI, Header, HTTPException, Request
from contextlib import asynccontextmanager
from multiprocessing import Manager, Process
from concurrent.futures import ThreadPoolExecutor
from typing import Coroutine, Optional, Tuple

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from openai.types.chat.chat_completion import ChatCompletion

from ajet.tuner_lib.weight_tuner.experimental.interchange_utils import EpisodeStatus
from ajet.utils.networking import find_free_port, get_host_ip
API_KEY_PREFIX = "sk-ajet-"

class InterchangeCompletionRequest(BaseModel):
    completion_request: ChatCompletionRequest
    agent_name: str
    target_tag: str
    episode_uuid: str
    timeline_uuid: str

class HealthCheckRequest(BaseModel):
    agent_name: str
    target_tag: str
    episode_uuid: str
    timeline_uuid: str
    health_check: bool = True

# Create FastAPI app
SERVER_SHUTDOWN_EVENT = threading.Event()
DEBUG = False
# DEBUG = True

context = zmq.Context()
atexit.register(context.term)









def get_app(max_fastapi_threads: int = 512, enable_tinkerscript_mode=False, shared_mem_dict=None, shared_mem_dict_lock=None) -> Tuple[FastAPI, Optional[Coroutine]]:


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


    def _begin_handle_chat_completion(episode_address, int_req: InterchangeCompletionRequest, episode_uuid, timeline_uuid, client_offline: threading.Event):
        """ run this in thread to avoid blocking main event loop
        """
        if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | Received new chat completion request (inside thread)")

        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 60*1000)  # 1 minute recv timeout
        socket.connect(f"{episode_address}")
        if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | connect done")
        socket.send_string(int_req.model_dump_json())
        if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | send_string")

        result_str = ""
        for _ in range(5):  # max 5 minutes wait
            try:
                if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | recv_string begin.")
                result_str = socket.recv_string()
                break
            except zmq.Again as e:
                if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | recv_string timeout, retrying.")
                continue

        if not result_str:
            raise RuntimeError(f"Failed to get response from episode_address: {episode_address} after 5 attempts.")
        else:
            if DEBUG: logger.success(f"[server] episode_uuid: {episode_uuid} | recv_string done.")
        result_object = ChatCompletion(**json.loads(result_str))
        return result_object


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

        # Parse request body
        body = await request.json()
        new_req = ChatCompletionRequest.model_validate(body)
        if new_req.stream:
            return HTTPException(status_code=400, detail="Streaming responses not supported in current AgentJet version, please set `stream=false` for now.")
        # Create timeline UUID
        timeline_uuid = uuid.uuid4().hex

        # enable_tinkerscript_mode
        if enable_tinkerscript_mode:
            assert shared_mem_dict is not None
            assert shared_mem_dict_lock is not None
            if shared_mem_dict['engine_status'] != "ROLLING":
                logger.error(f"The server is not in ROLLING status (current status: [{shared_mem_dict['engine_status']}]), cannot accept new requests.")
                raise HTTPException(status_code=503, detail="The server is not in ROLLING status, cannot accept new requests.")
            if (f"episodes-{episode_uuid}") not in shared_mem_dict:
                raise HTTPException(status_code=404, detail=f"Episode {episode_uuid} not found.")
            # update activate timestamp
            with shared_mem_dict_lock:
                es:EpisodeStatus = shared_mem_dict[f"episodes-{episode_uuid}"]
                es.latest_activity_timestamp = time.time()
                shared_mem_dict[f"episodes-{episode_uuid}"] = es

        # Add to received queue
        int_req = InterchangeCompletionRequest(
            completion_request = new_req,
            agent_name = agent_name,
            target_tag = target_tag,
            episode_uuid = episode_uuid,
            timeline_uuid = timeline_uuid,
        )
        if DEBUG: logger.info(f"episode_uuid: {episode_uuid} | Received new chat completion request (outside thread)")
        client_offline = threading.Event()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(request.app.state.executor, _begin_handle_chat_completion, episode_address, int_req, episode_uuid, timeline_uuid, client_offline)
        finally:
            client_offline.set()


    @app.post("/reset")
    async def reset():
        return {"status": "reset_complete"}


    if enable_tinkerscript_mode:
        from ajet.tuner_lib.weight_tuner.experimental.as_tinkerscript_server import register_enable_tinkerscript_mode_routes
        assert shared_mem_dict is not None, "shared_mem_dict must not be None when enable_tinkerscript_mode is True."
        assert shared_mem_dict_lock is not None, "shared_mem_dict_lock must not be None when enable_tinkerscript_mode is True."
        app, additional_coro = register_enable_tinkerscript_mode_routes(app, zmq_context=context, shared_mem_dict=shared_mem_dict, shared_mem_dict_lock=shared_mem_dict_lock)
    else:
        additional_coro = None


    return app, additional_coro













class InterchangeServer(Process):
    def __init__(self, experiment_dir: str, port: int, num_fastapi_process: int = 2, max_fastapi_threads: int = 512, enable_tinkerscript_mode=False):
        super().__init__()
        self.experiment_dir = experiment_dir
        self.port = port
        self.num_fastapi_process = num_fastapi_process
        self.max_fastapi_threads = max_fastapi_threads
        self.enable_tinkerscript_mode = enable_tinkerscript_mode

    def run(self):
        logger.info(f"Starting Interchange Server on port {self.port} with {self.num_fastapi_process} processes and {self.max_fastapi_threads} threads per process.")

        if self.enable_tinkerscript_mode:
            manager = Manager()
            shared_mem_dict = manager.dict()
            shared_mem_dict_lock = manager.Lock()
        else:
            shared_mem_dict = None
            shared_mem_dict_lock = None

        app, additional_coro = get_app(self.max_fastapi_threads, self.enable_tinkerscript_mode, shared_mem_dict, shared_mem_dict_lock)

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
def start_interchange_server(config, blocking=False) -> int:
    # Read config
    already_started = config.ajet.interchange_server.already_started
    experiment_dir = config.ajet.experiment_dir
    num_fastapi_process = config.ajet.interchange_server.num_fastapi_process
    max_fastapi_threads = config.ajet.interchange_server.max_fastapi_threads
    enable_tinkerscript_mode = config.ajet.enable_tinkerscript_mode

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
        interchange_server = InterchangeServer(
            experiment_dir,
            port,
            num_fastapi_process,
            max_fastapi_threads,
            enable_tinkerscript_mode,
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
    while True:
        if interchange_server and interchange_server.exitcode is not None:
            logger.error(f"Interchange server subprocess failed to start. Return code: {interchange_server.exitcode}")
            raise RuntimeError("Interchange server subprocess failed to start.")
        if time.time() - start_time > 30:
            msg = f"Interchange server subprocess failed to start within {time.time() - start_time} seconds."
            logger.error(msg)
            raise RuntimeError(msg)
        try:
            if httpx.get(health_url, timeout=0.5).status_code == 200:
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
        if interchange_server:
            interchange_server.join()
        return -1