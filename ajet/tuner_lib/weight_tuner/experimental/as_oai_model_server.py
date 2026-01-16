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
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from openai.types.chat.chat_completion import ChatCompletion

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


def get_app(max_fastapi_threads: int = 512) -> FastAPI:

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

    return app

class InterchangeServer(Process):
    def __init__(self, experiment_dir: str, port: int, num_fastapi_process: int = 2, max_fastapi_threads: int = 512):
        super().__init__()
        self.experiment_dir = experiment_dir
        self.port = port
        self.num_fastapi_process = num_fastapi_process
        self.max_fastapi_threads = max_fastapi_threads

    def run(self):
        logger.info(f"Starting Interchange Server on port {self.port} with {self.num_fastapi_process} processes and {self.max_fastapi_threads} threads per process.")
        app = get_app(self.max_fastapi_threads)
        async def serve_with_monitor():
            # Start the server
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=self.port,
                log_level="error",
                workers=self.num_fastapi_process
            )
            server = uvicorn.Server(config)
            await server.serve()
        try:
            asyncio.run(serve_with_monitor())
        except KeyboardInterrupt as e:
            SERVER_SHUTDOWN_EVENT.set()
            raise e


# Convenience function for quick server startup
def start_interchange_server(config) -> int:
    experiment_dir = config.ajet.experiment_dir
    num_fastapi_process = config.ajet.interchange_server.num_fastapi_process
    max_fastapi_threads = config.ajet.interchange_server.max_fastapi_threads
    # Find a free port if not specified or invalid
    port = int(os.environ.get("AJET_DAT_INTERCHANGE_PORT", -1))

    if config.ajet.interchange_server.interchange_server_port != 'auto':
        port = int(config.ajet.interchange_server.interchange_server_port)

    if port <= 0:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        os.environ["AJET_DAT_INTERCHANGE_PORT"] = str(port)

    interchange_server = InterchangeServer(experiment_dir, port, num_fastapi_process, max_fastapi_threads)
    interchange_server.start()

    # Wait for server to be ready
    health_url = f"http://localhost:{port}/health"
    start_time = time.time()
    while True:
        if interchange_server.exitcode is not None:
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
    if DEBUG: logger.info(f"Interchange server subprocess started on port {port} (pid: {interchange_server.pid})")
    atexit.register(lambda: interchange_server.terminate())

    # return port
    return port
