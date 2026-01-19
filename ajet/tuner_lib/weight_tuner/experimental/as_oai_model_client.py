
import asyncio
import atexit
import json
import os
import time
import zmq
import base64
import json

from loguru import logger
from typing import TYPE_CHECKING
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from openai.types.chat.chat_completion import ChatCompletion
from ajet.tuner_lib.weight_tuner.experimental.as_oai_model_server import InterchangeCompletionRequest, API_KEY_PREFIX
from ajet.utils.thread_executors import SharedInferenceTrackerThreadExecutor, SharedInterchangeThreadExecutor
from ajet.tuner_lib.weight_tuner.experimental.interchange_utils import get_zmq_socket

context = zmq.Context()
atexit.register(context.term)

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
    auth_token = f"{API_KEY_PREFIX}{base64_encoded}"    # API_KEY_PREFIX: Literal['sk-ajet-']

    return auth_token


class InterchangeClient:
    """ InterchangeClient is re-created in each episode
    """

    def __init__(self, episode_uuid: str, context_tracker: "MultiAgentContextTracker", llm_inference_fn, config):
        self.episode_uuid = episode_uuid
        self.context_tracker = context_tracker
        self.llm_inference_fn = llm_inference_fn
        self.config = config
        self._should_terminate = False
        self.episode_contect_address, ipc_path = get_zmq_socket(config, episode_uuid, tag="llm")
        self.ipc_path = ipc_path
        self.interchange_method = config.ajet.interchange_server.interchange_method
        self.max_inference_tracker_threads = config.ajet.interchange_server.max_inference_tracker_threads

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
        Starts the zmq communication loop.
        """
        if DEBUG: logger.info(f"[client] {self.episode_uuid} | Starting InterchangeClient service loop...")
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"{self.episode_contect_address}")
        self.socket.setsockopt(zmq.RCVTIMEO, 3*1000)  # 3 second timeout for REP

        self.executor = SharedInterchangeThreadExecutor(self.max_inference_tracker_threads).get_shared_executor()
        if DEBUG: logger.info(f"[client] {self.episode_uuid} | Submitting _begin_service_threading to executor...")
        future = self.executor.submit(self._begin_service_threading)

        # wait till service begin running
        time.sleep(0.5)
        w_time = 1
        while future._state == 'PENDING':
            time.sleep(min(w_time * 2, 10))
            w_time += 1

        if DEBUG: logger.info(f"[client] {self.episode_uuid} | Future ready...")
        return self.episode_contect_address


    def _begin_service_threading(self):
        """begin listening for service requests in a threading model
        """

        begin_time = time.time()
        if DEBUG: logger.info(f"[client] {self.episode_uuid} | Starting ZMQ socket bind complete")

        try:
            while not self.should_terminate:
                # listen for next request from remote
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
                        if DEBUG: logger.warning(f"[client] {self.episode_uuid} | Still waiting for first message... (time passed {timepassed}) for episode_uuid:{self.episode_uuid}...")
                    continue

                # parse the incoming request
                if DEBUG: logger.info(f"[client] {self.episode_uuid} | before json.loads(message)")
                data_as_json = json.loads(message)
                parsed_msg = InterchangeCompletionRequest(**data_as_json)

                # begin to run the llm request, monitored by context tracker
                # we re-use previously created thread for best performance
                if DEBUG: logger.info(f"[client] {self.episode_uuid} | before asyncio run self.llm_infer")
                try:
                    loop = asyncio.get_running_loop()
                except:
                    loop = asyncio.new_event_loop()
                context_tracker_executor = SharedInferenceTrackerThreadExecutor(self.max_inference_tracker_threads).get_shared_executor()
                future = loop.run_in_executor(
                    context_tracker_executor,
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

                # great, let's send back the result
                if DEBUG: logger.info(f"[client] {self.episode_uuid} | before send_string")
                self.socket.send_string(result)
        except:
            logger.exception(f"[client] {self.episode_uuid} | Exception occurred in service loop.")
        finally:
            self.socket.close()
            if DEBUG: logger.info(f"[client] {self.episode_uuid} | ZMQ socket closed, service loop terminated.")
            if self.interchange_method == 'ipc':
                if os.path.exists(self.ipc_path):
                    os.remove(self.ipc_path)
                    if DEBUG: logger.info(f"[client] {self.episode_uuid} | IPC socket file {self.ipc_path} removed.")
