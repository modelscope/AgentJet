
import asyncio
import atexit
import json
import os
import time
import zmq
import zmq.asyncio

from loguru import logger
from typing import TYPE_CHECKING
from ajet.tuner_lib.experimental.oai_model_server import InterchangeCompletionRequest
from ajet.utils.thread_executors import SharedInterchangeThreadExecutor
from ajet.tuner_lib.experimental.interchange_utils import get_zmq_socket
from ajet.tuner_lib.experimental.interchange_utils import DEBUG

if TYPE_CHECKING:
    pass

context = zmq.asyncio.Context()
atexit.register(context.term)

if TYPE_CHECKING:
    from ajet.context_tracker.multiagent_tracking import MultiAgentContextTracker


class InterchangeClient:
    """ InterchangeClient is re-created in each episode
    """

    def __init__(self, episode_uuid: str, context_tracker: "MultiAgentContextTracker", llm_inference_fn, config):
        from ajet.task_rollout.async_llm_bridge import OpenaiLlmProxyWithTracker
        self.episode_uuid = episode_uuid
        self.context_tracker = context_tracker
        self.llm_inference_fn = llm_inference_fn
        self.config = config
        self._should_terminate = False
        self.episode_contect_address, ipc_path = get_zmq_socket(config, episode_uuid, tag="llm")
        self.ipc_path = ipc_path
        self.interchange_method = config.ajet.interchange_server.interchange_method
        self.max_inference_tracker_threads = config.ajet.interchange_server.max_inference_tracker_threads
        self.llm_proxy_with_tracker = OpenaiLlmProxyWithTracker(
            context_tracker=self.context_tracker,
            config=self.config,
            llm_inference_fn=self.llm_inference_fn,
        )

    @property
    def should_soft_terminate(self) -> bool:
        if self._should_terminate:
            return True
        return self.context_tracker.should_interrupt_soft_fn()

    @property
    def should_hard_terminate(self) -> bool:
        if self._should_terminate:
            return True
        if not self.config.ajet.enable_swarm_mode:
            return self.should_soft_terminate
        else:
            return self.context_tracker.should_interrupt_hard_fn()



    def begin_service(self):
        """
        Starts the zmq communication loop.
        """
        if self.should_soft_terminate or self.should_hard_terminate:
            return self.episode_contect_address

        if DEBUG: logger.info(f"[client] {self.episode_uuid} | Starting InterchangeClient service loop...")
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"{self.episode_contect_address}")

        self.executor = SharedInterchangeThreadExecutor(self.max_inference_tracker_threads).get_shared_executor()
        if DEBUG: logger.info(f"[client] {self.episode_uuid} | Submitting _run_service_loop to executor...")
        future = self.executor.submit(self._run_service_loop)

        # wait till service begin running
        wait_time = 1
        time.sleep(wait_time)
        while future._state == 'PENDING':
            if self.should_soft_terminate or self.should_hard_terminate:
                future.cancel()
                self.socket.close()
                if os.path.exists(self.ipc_path): os.remove(self.ipc_path)
                return self.episode_contect_address
            time.sleep(min(wait_time * 2, 10))
            wait_time += 1

        if DEBUG: logger.info(f"[client] {self.episode_uuid} | Future ready...")
        return self.episode_contect_address


    def _run_service_loop(self):
        """Runs a dedicated asyncio event loop for this episode's zmq service.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._begin_service_async())
        finally:
            loop.close()
            asyncio.set_event_loop(None)


    async def _begin_service_async(self):
        """begin listening for service requests using zmq.asyncio
        """

        begin_time = time.time()
        ever_receive_anything = False
        if DEBUG: logger.info(f"[client] {self.episode_uuid} | Starting ZMQ socket bind complete")

        poller = zmq.asyncio.Poller()
        poller.register(self.socket, zmq.POLLIN)

        try:
            while not self.should_hard_terminate:
                events = dict(await poller.poll(timeout=1000))  # 1 second
                if self.socket not in events:
                    if self.should_hard_terminate:
                        # abort_episode()
                        if DEBUG: logger.info(f"[client] {self.episode_uuid} | episode over")
                        break
                    timepassed = time.time() - begin_time
                    if (not ever_receive_anything) and (timepassed > 100):
                        if DEBUG: logger.warning(f"[client] {self.episode_uuid} | Still waiting for first message... (time passed {timepassed}) for episode_uuid:{self.episode_uuid}...")
                    continue

                # <wait for>:
                #   <from_sourcefile>: ajet/tuner_lib/experimental/oai_model_server.py
                #   <from_code>: socket.send_string(int_req.model_dump_json())
                #   <expect>: InterchangeCompletionRequest object in JSON string format
                message = await self.socket.recv_string()
                ever_receive_anything = True

                # parse the incoming request
                if DEBUG: logger.info(f"[client] {self.episode_uuid} | before json.loads(message)")
                data_as_json = json.loads(message)
                parsed_msg = InterchangeCompletionRequest(**data_as_json)

                # run the llm request, monitored by context tracker
                if DEBUG: logger.info(f"[client] {self.episode_uuid} | before awaiting self.llm_infer")
                response = await self.llm_proxy_with_tracker.chat_completion_request(
                    req=parsed_msg.completion_request,
                    timeline_uuid=parsed_msg.timeline_uuid,
                    agent_name=parsed_msg.agent_name,
                    target_tag=parsed_msg.target_tag,
                    episode_uuid=parsed_msg.episode_uuid,
                )
                result = response.model_dump_json()

                if DEBUG: logger.info(f"[client] {self.episode_uuid} | before send_string (send llm call result)")

                # <send to>
                #   <to_sourcefile>: ajet/tuner_lib/experimental/oai_model_server.py
                #   <to_code>: result_str = socket.recv_string()
                await self.socket.send_string(result)

                if DEBUG: logger.info(f"[client] {self.episode_uuid} | after send_string (send llm call result)")
        except:
            logger.exception(f"[client] {self.episode_uuid} | Exception occurred in service loop.")
        finally:
            self.socket.close()
            if DEBUG: logger.info(f"[client] {self.episode_uuid} | ZMQ socket closed, service loop terminated.")
            if self.interchange_method == 'ipc':
                if os.path.exists(self.ipc_path):
                    os.remove(self.ipc_path)
                    if DEBUG: logger.info(f"[client] {self.episode_uuid} | IPC socket file {self.ipc_path} removed.")
