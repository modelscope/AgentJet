
import asyncio
import json
import threading
import os
import redis
import time
from loguru import logger
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse
from openai.types.chat.chat_completion import ChatCompletion
from redis.exceptions import TimeoutError

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

def generate_auth_token(agent_name, target_tag, episode_uuid):
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
        "episode_uuid": episode_uuid
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


def get_redis_client():
    pool = get_redis_connection_pool()
    return redis.Redis(connection_pool=pool, decode_responses=False, encoding='utf-8')


class InterchangeClient:

    def __init__(self, episode_uuid: str, context_tracker: "MultiAgentContextTracker", llm_inference_fn, config):
        self.episode_uuid = episode_uuid
        self.context_tracker = context_tracker
        self.llm_inference_fn = llm_inference_fn
        self.config = config
        self._should_terminate = False
        self.begin_service()


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
        t = threading.Thread(target=self._begin_service_threading, daemon=True)
        t.start()


    def _handle_service_request(self, msg: bytes, sem: threading.Semaphore):
        """handle a single service request in its own thread
        """
        from ajet.tuner_lib.weight_tuner.experimental.as_oai_model_server import InterchangeCompletionRequest
        logger.info(f"[client] {self.episode_uuid} | inside _handle_service_request")
        redis_client = get_redis_client()
        logger.info(f"[client] {self.episode_uuid} | get_redis_client")
        data_as_json = ""
        topic = ""
        try:
            data_as_json = json.loads(pickle.loads(msg))
            timeline_uuid = data_as_json["timeline_uuid"]
            topic = f"stream:timeline:{timeline_uuid}"
            logger.info(f"[client] {self.episode_uuid} | json.loads(pickle.loads(msg))")


            if "health_check" in data_as_json and data_as_json["health_check"]:
                # logger.info(f"Received health check for timeline_uuid: {timeline_uuid}")
                result = '{"health_check_ok": "True"}'
                # logger.success(f"Health check OK for timeline_uuid: {timeline_uuid}")
            else:
                parsed_msg = InterchangeCompletionRequest(**data_as_json)
                # start llm request
                result = asyncio.run(self.llm_infer(
                    req=parsed_msg.completion_request,
                    timeline_uuid=parsed_msg.timeline_uuid,
                    agent_name=parsed_msg.agent_name,
                    target_tag=parsed_msg.target_tag,
                    episode_uuid=parsed_msg.episode_uuid,
                )).model_dump_json()
                # logger.success(f"LLM inference completed for timeline_uuid: {timeline_uuid}")
            logger.info(f"[client] {self.episode_uuid} | result = asyncio.run(self.llm_infer")
            # send result back
            bytes_arr = pickle.dumps(result)
            logger.info(f"[client] {self.episode_uuid} | bytes_arr = pickle.dumps(result)")
            redis_client.xadd(topic, {'data': bytes_arr})
            redis_client.expire(topic, 600)  # expire after 10 mins
            logger.info(f"[client] {self.episode_uuid} | redis_client.xadd(topic, ...)")

        except Exception as e:
            err = f"[ERR]: Error when processing data: {data_as_json} Error: {e}"
            result = err
            logger.error(err)
            if topic:
                redis_client.xadd(topic, {'data': pickle.dumps(result)})
                redis_client.expire(topic, 600)

        finally:
            # release semaphore when done
            sem.release()
            redis_client.close()




    def _begin_service_threading(self):
        """begin listening for service requests in a threading model
        """
        # logger.success(f"InterchangeClient starting for episode_uuid:{self.episode_uuid}")
        # debug_logs = []
        begin_time = time.time()
        logger.info(f"[client] {self.episode_uuid} | Starting InterchangeClient service loop...")
        redis_client = get_redis_client()
        episode_stream = f"stream:episode:{self.episode_uuid}"

        sem = threading.Semaphore(8)    # 4 concurrent requests max
        logger.info(f"[client] {self.episode_uuid} | Listening to stream {episode_stream}, waiting for messages...")

        last_id = '0-0'
        is_init = True

        try:
            while not self.should_terminate:
                # wait for a new message
                logger.info(f"[client] {self.episode_uuid} | Waiting for new message on stream {episode_stream}...")

                # Check messages
                try:
                    response = redis_client.xread({episode_stream: last_id}, count=1, block=30*1000)   # block for 30 seconds (30000 ms)
                except TimeoutError:
                    time.sleep(5)
                    continue

                timepassed = time.time() - begin_time

                if not response:
                    if is_init and timepassed > 30:
                        logger.warning(f"[client] Still waiting for first message... (time passed {timepassed}) for episode_uuid:{self.episode_uuid}...")
                    continue

                # Got message
                is_init = False
                logger.info(f"[client] {self.episode_uuid} | get message...")

                stream_result = response[0]
                messages = stream_result[1]
                msg_id, data_dict = messages[0]

                last_id = msg_id

                if b'data' in data_dict:
                    msg: bytes = data_dict[b'data']
                else:
                    logger.error(f"Missing 'data' in stream message {msg_id}")
                    continue

                # are we free to spawn a new thread?
                sem.acquire()
                logger.info(f"[client] {self.episode_uuid} | sem acquire...")
                # begin a new thread to handle this request
                threading.Thread(target=self._handle_service_request, args=(msg, sem), daemon=True).start()


        except KeyboardInterrupt:
            return

        finally:
            redis_client.delete(episode_stream)
            redis_client.close()

