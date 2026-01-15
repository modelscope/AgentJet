
import asyncio
import atexit
import json
import threading
import os
import redis
import time
from loguru import logger
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse
from openai.types.chat.chat_completion import ChatCompletion
from ajet.tuner_lib.weight_tuner.experimental.as_oai_model_server import InterchangeCompletionRequest
from redis.exceptions import TimeoutError
from ajet.utils.free_port import find_free_port
from ajet.utils.sington import ThreadExecutorLlmInferSingleton, ThreadExecutorSingleton
from functools import cache

import pickle
import httpx
import zmq
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

import base64
import json

if TYPE_CHECKING:
    from ajet.context_tracker.multiagent_tracking import MultiAgentContextTracker

DEBUG = False
# DEBUG = True

def generate_auth_token(agent_name, target_tag, episode_uuid, episode_address):
    """
    Generate a Base64-encoded auth_token from the given agent_name, target_tag, and episode_uuid.

    Args:
        agent_name (str): The name of the agent.
        target_tag (str): The target tag.
        episode_uuid (str): The UUID of the episode.

    Returns:
        str: The generated auth_token in the format "Bearer <base64_encoded_string>".
    """
    # Step 1: Construct the auth_data dictionary
    auth_data = {
        "agent_name": agent_name,
        "target_tag": target_tag,
        "episode_uuid": episode_uuid,
        "episode_address": episode_address,
    }

    # Step 2: Convert the dictionary to a JSON string
    json_string = json.dumps(auth_data)

    # Step 3: Encode the JSON string into Base64
    base64_encoded = base64.b64encode(json_string.encode('utf-8')).decode('utf-8')

    # Step 4: Prepend "Bearer " to the Base64-encoded string
    auth_token = f"Bearer {base64_encoded}"

    return auth_token


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

@cache
def get_redis_client():
    pool = get_redis_connection_pool()
    return redis.Redis(connection_pool=pool, decode_responses=False, encoding='utf-8')


context = zmq.Context()
atexit.register(context.term)

class InterchangeClient:

    def __init__(self, episode_uuid: str, context_tracker: "MultiAgentContextTracker", llm_inference_fn, config):
        self.episode_uuid = episode_uuid
        self.context_tracker = context_tracker
        self.llm_inference_fn = llm_inference_fn
        self.config = config
        self._should_terminate = False

        # self.episode_contect_address = f"tcp://localhost:{find_free_port()}"
        self.ipc_path = f"/tmp/ajet/{self.episode_uuid}.sock"
        self.episode_contect_address = f"ipc://{self.ipc_path}"


    async def llm_infer(
            self,
            req: ChatCompletionRequest,
            timeline_uuid: str,
            agent_name: str,
            target_tag: str,
            episode_uuid: str,
        ) -> ChatCompletion:
        from ajet.task_rollout.async_llm_bridge import OpenaiLlmProxyWithTracker

        req_as_dict = req.model_dump()

        self.llm_proxy_with_tracker = OpenaiLlmProxyWithTracker(
            context_tracker=self.context_tracker,
            config=self.config,
            llm_inference_fn=self.llm_inference_fn,
        )

        # infer + process with context tracker
        response = await self.llm_proxy_with_tracker(
            messages=req_as_dict["messages"],
            tools=req_as_dict["tools"],
            tool_choice="auto",
        )

        # this is an important id assignment
        response.id = timeline_uuid
        assert isinstance(response, ChatCompletion)
        return response


    @property
    def should_terminate(self) -> bool:
        return self._should_terminate

    def begin_service(self):
        """
        Starts the SSE service loop.
        """
        if DEBUG: logger.info(f"[client] {self.episode_uuid} | Starting InterchangeClient service loop...")
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"{self.episode_contect_address}")
        self.socket.setsockopt(zmq.RCVTIMEO, 2*1000)  # 60 秒超时

        self.executor = ThreadExecutorSingleton().get_executor()
        if DEBUG: logger.info(f"[client] {self.episode_uuid} | Submitting _begin_service_threading to executor...")
        future = self.executor.submit(self._begin_service_threading)
        time.sleep(1)
        while future._state == 'PENDING':
            time.sleep(1)
        if DEBUG: logger.info(f"[client] {self.episode_uuid} | Future ready...")

        # t = threading.Thread(target=self._begin_service_threading, daemon=True)
        # t.start()
        return self.episode_contect_address


    def _begin_service_threading(self):
        """begin listening for service requests in a threading model
        """

        begin_time = time.time()
        if DEBUG: logger.info(f"[client] {self.episode_uuid} | Starting ZMQ socket bind complete")

        try:
            while not self.should_terminate:

                try:
                    if DEBUG: logger.info(f"[client] {self.episode_uuid} | socket.recv_string() has begun")
                    message = self.socket.recv_string()
                    if DEBUG: logger.info(f"[client] {self.episode_uuid} | socket.recv_string() is done")
                except zmq.Again as e:
                    if self.should_terminate:
                        if DEBUG: logger.info(f"[client] {self.episode_uuid} | episode over")
                        break
                    timepassed = time.time() - begin_time
                    if timepassed > 60:
                        logger.warning(f"[client] {self.episode_uuid} | Still waiting for first message... (time passed {timepassed}) for episode_uuid:{self.episode_uuid}...")
                    continue

                if DEBUG: logger.info(f"[client] {self.episode_uuid} | before json.loads(message)")
                data_as_json = json.loads(message)
                parsed_msg = InterchangeCompletionRequest(**data_as_json)

                if DEBUG: logger.info(f"[client] {self.episode_uuid} | before asyncio run self.llm_infer")

                try:
                    loop = asyncio.get_running_loop()
                except:
                    loop = asyncio.new_event_loop()
                executor = ThreadExecutorLlmInferSingleton().get_executor()
                future = loop.run_in_executor(
                    executor,  # executor
                    asyncio.run,
                    self.llm_infer(
                        req=parsed_msg.completion_request,
                        timeline_uuid=parsed_msg.timeline_uuid,
                        agent_name=parsed_msg.agent_name,
                        target_tag=parsed_msg.target_tag,
                        episode_uuid=parsed_msg.episode_uuid,
                    )
                )
                result = loop.run_until_complete(future).model_dump_json()  # type: ignore

                if DEBUG: logger.info(f"[client] {self.episode_uuid} | before send_string")
                self.socket.send_string(result)
        except:
            logger.exception(f"[client] {self.episode_uuid} | Exception occurred in service loop.")
        finally:
            self.socket.close()
            if DEBUG: logger.info(f"[client] {self.episode_uuid} | ZMQ socket closed, service loop terminated.")
            if os.path.exists(self.ipc_path):
                os.remove(self.ipc_path)
                if DEBUG: logger.info(f"[client] {self.episode_uuid} | IPC socket file {self.ipc_path} removed.")
