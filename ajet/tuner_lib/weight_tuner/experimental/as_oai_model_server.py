"""
A shadow FastAPI server for serving as interchange endpoint between Tuner and Workflow.

- This functionality is experimental.
- The code is very async, considering extreme efficiency for handling many concurrent requests,
  therefore, it may be hard to read.

---------------------------------------------------------------------------------------------

"""

import asyncio
from functools import cache
from multiprocessing import Process
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
import time
from collections import defaultdict
from typing import Dict, List
import base64
import json
import os
import pickle
import redis
from redis.exceptions import TimeoutError
from pprint import pformat
from loguru import logger

from pydantic import BaseModel, ConfigDict, model_validator
from fastapi import FastAPI, Header, HTTPException, Request, Body
from fastapi.responses import StreamingResponse
import uvicorn
import sys
import subprocess
import atexit
import argparse
import httpx

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from openai.types.chat.chat_completion import ChatCompletion


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

@cache
def get_redis_connection_pool():
    pool = redis.BlockingConnectionPool(
        host='localhost',
        port=6379,
        max_connections=256,
        socket_timeout=30,
        socket_connect_timeout=30,
        retry_on_timeout=True
    )
    return pool


def get_redis_client():
    pool = get_redis_connection_pool()
    return redis.Redis(connection_pool=pool, decode_responses=False, encoding='utf-8')


# Create FastAPI app
app = FastAPI(title="AJet Interchange Endpoint")

@app.on_event("startup")
async def startup_event():
    app.state.executor = ThreadPoolExecutor(max_workers=512)

@app.on_event("shutdown")
async def shutdown_event():
    app.state.executor.shutdown()


def _begin_handle_chat_completion(int_req, episode_uuid, timeline_uuid, client_offline: asyncio.Event):
    """ run this in thread to avoid blocking main event loop
    """
    logger.info(f"episode_uuid: {episode_uuid} | Received new chat completion request for episode_uuid: {episode_uuid}, timeline_uuid: {timeline_uuid} (inside thread)")

    redis_client = get_redis_client()
    episode_topic = f"episode_uuid:{episode_uuid}"
    timeline_topic = f"timeline_uuid:{timeline_uuid}/episode_uuid:{episode_uuid}"
    redis_sub = redis_client.pubsub()
    redis_sub.subscribe(timeline_topic)
    max_wait_time = 600  # 10 minutes timeout
    try:
        logger.info(f"episode_uuid: {episode_uuid} | redis_client.publish int_req ")
        redis_client.publish(episode_topic, pickle.dumps(int_req.model_dump_json()))
        logger.info(f"episode_uuid: {episode_uuid} | redis_client.publish int_req end")

        # record start
        begin_time = time.time()
        max_wait_time = 600  # 10 minutes timeout
        # wait for result
        while not client_offline.is_set():
            timepassed = time.time() - begin_time
            if timepassed > max_wait_time:
                return HTTPException(status_code=504, detail="Request timeout")
            try:
                logger.info(f"episode_uuid: {episode_uuid} | redis_sub.get_message(timeout=60)")
                result = redis_sub.get_message(timeout=60)
                logger.info(f"episode_uuid: {episode_uuid} | redis_sub.get_message(timeout=60) after")
                if result is None:
                    if timepassed > 60:
                        logger.warning(f"episode_uuid: {episode_uuid} |  LLM client infer still waiting... (time passed {timepassed}) for episode_uuid:{episode_uuid}, timeline_uuid:{timeline_uuid}...")
                    continue
                if result['type'] not in ['message', 'pmessage']:
                    continue
                logger.info(f"episode_uuid: {episode_uuid} | successfully get message from redis_sub")
                result_object_str = pickle.loads(result['data'])   # type: ignore
                if result_object_str.startswith('[ERR]'):
                    return HTTPException(status_code=500, detail="Error response, " + result_object_str)
                result_object = ChatCompletion(**json.loads(result_object_str))
                return result_object
            except TimeoutError:
                logger.info(f"episode_uuid: {episode_uuid} | still waiting, (time passed {timepassed}) for result for episode_uuid:{episode_uuid}, timeline_uuid:{timeline_uuid}...")
                continue

    except:
        return HTTPException(status_code=500, detail="ZMQ communication socket failed.")

    finally:
        redis_sub.close()
        redis_client.close()

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
        auth_token = authorization.replace("Bearer ", "").replace("bearer ", "")
        decoded = base64.b64decode(auth_token).decode('utf-8')
        auth_data = json.loads(decoded)

        agent_name = auth_data.get("agent_name")
        target_tag = auth_data.get("target_tag")
        episode_uuid = auth_data.get("episode_uuid")

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
    # logger.warning(f"Received new chat completion request for agent: {agent_name}, target_tag: {target_tag}, episode_uuid: {episode_uuid}, timeline_uuid: {timeline_uuid}")
    int_req = InterchangeCompletionRequest(
        completion_request = new_req,
        agent_name = agent_name,
        target_tag = target_tag,
        episode_uuid = episode_uuid,
        timeline_uuid = timeline_uuid,
    )
    logger.info(f"episode_uuid: {episode_uuid} | Received new chat completion request for episode_uuid: {episode_uuid}, timeline_uuid: {timeline_uuid} (outside thread)")
    client_offline = asyncio.Event()
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(request.app.state.executor, _begin_handle_chat_completion, int_req, episode_uuid, timeline_uuid, client_offline)
    finally:
        client_offline.set()




@app.post("/reset")
async def reset():

    return {"status": "reset_complete"}


async def monitor_debug_state(experiment_dir):
    """
    Background task to write debug state to ./interchange_debug.txt every 1 second.
    """
    while True:
        await asyncio.sleep(4)


def ensure_dat_interchange_server_cache_clear():
    return



class InterchangeServer(Process):
    def __init__(self, experiment_dir: str, port: int):
        super().__init__()
        self.experiment_dir = experiment_dir
        self.port = port

    def run(self):
        async def serve_with_monitor():
            # Start the monitor task
            asyncio.create_task(monitor_debug_state(self.experiment_dir))
            # Start the server
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=self.port,
                log_level="error",
                # workers=16
            )
            server = uvicorn.Server(config)
            await server.serve()

        asyncio.run(serve_with_monitor())


# Convenience function for quick server startup
def start_interchange_server(experiment_dir) -> int:
    # Find a free port if not specified or invalid
    port = int(os.environ.get("AJET_DAT_INTERCHANGE_PORT", -1))
    if port <= 0:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        os.environ["AJET_DAT_INTERCHANGE_PORT"] = str(port)

    interchange_server = InterchangeServer(experiment_dir, port)
    interchange_server.daemon = True
    interchange_server.start()

    # Wait for server to be ready
    health_url = f"http://localhost:{port}/health"
    start_time = time.time()
    while time.time() - start_time < 20:
        if interchange_server.exitcode is not None:
            logger.error(f"Interchange server subprocess failed to start. Return code: {interchange_server.exitcode}")
            break
        try:
            if httpx.get(health_url, timeout=0.5).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)

    logger.info(f"Interchange server subprocess started on port {port} (pid: {interchange_server.pid})")
    return port


