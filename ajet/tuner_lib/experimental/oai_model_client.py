
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

        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"{self.episode_contect_address}")

        self.executor = SharedInterchangeThreadExecutor(self.max_inference_tracker_threads).get_shared_executor()
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

        poller = zmq.asyncio.Poller()
        poller.register(self.socket, zmq.POLLIN)

        _idle_polls = 0
        _in_flight_tl = None
        try:
            while not self.should_hard_terminate:
                events = dict(await poller.poll(timeout=1000))  # 1 second
                if self.socket not in events:
                    if self.should_hard_terminate:
                        # abort_episode()
                        break
                    _idle_polls += 1
                    # heartbeat every ~30s so we can tell "idle waiting for request"
                    # apart from "stuck mid-request" in the logs.
                    if _idle_polls % 30 == 0:
                        logger.info(f"[client][IDLE-POLL] episode_uuid={self.episode_uuid} addr={self.episode_contect_address} idle_polls={_idle_polls} last_tl={_in_flight_tl} -> REP socket alive, no request pending.")
                    timepassed = time.time() - begin_time
                    continue

                # <wait for>:
                #   <from_sourcefile>: ajet/tuner_lib/experimental/oai_model_server.py
                #   <from_code>: socket.send_string(int_req.model_dump_json())
                #   <expect>: InterchangeCompletionRequest object in JSON string format
                message = await self.socket.recv_string()
                ever_receive_anything = True

                _t_recv = time.time()
                # parse the incoming request
                data_as_json = json.loads(message)
                parsed_msg = InterchangeCompletionRequest(**data_as_json)
                _tl = parsed_msg.timeline_uuid
                _in_flight_tl = _tl
                _idle_polls = 0
                logger.info(f"[client][RECV] episode_uuid={self.episode_uuid} tl={_tl} -> got request, dispatching to LLM.")

                # run the llm request, monitored by context tracker
                response = await self.llm_proxy_with_tracker.chat_completion_request(
                    req=parsed_msg.completion_request,
                    timeline_uuid=parsed_msg.timeline_uuid,
                    agent_name=parsed_msg.agent_name,
                    target_tag=parsed_msg.target_tag,
                    episode_uuid=parsed_msg.episode_uuid,
                )
                _t_infer = time.time()
                logger.info(f"[client][INFER-DONE] episode_uuid={self.episode_uuid} tl={_tl} infer_took={_t_infer-_t_recv:.1f}s -> sending reply.")
                result = response.model_dump_json()

                # <send to>
                #   <to_sourcefile>: ajet/tuner_lib/experimental/oai_model_server.py
                #   <to_code>: result_str = socket.recv_string()
                await self.socket.send_string(result)
                logger.info(f"[client][SENT-REPLY] episode_uuid={self.episode_uuid} tl={_tl} send_took={time.time()-_t_infer:.2f}s total={time.time()-_t_recv:.1f}s.")
        except BaseException as e:
            import traceback
            print(f"[client] {self.episode_uuid} | Exception occurred in service loop: {e!r}", flush=True)
            traceback.print_exc()
            logger.exception(f"[client] {self.episode_uuid} | Exception occurred in service loop.")
        finally:
            logger.warning(
                f"[client][SVC-LOOP-CLOSE] episode_uuid={self.episode_uuid} "
                f"addr={self.episode_contect_address} ever_received={ever_receive_anything} "
                f"hard_terminate={self.should_hard_terminate} soft_terminate={self.should_soft_terminate} "
                f"uptime={time.time()-begin_time:.1f}s -> closing REP socket; further requests to this addr will time out."
            )
            self.socket.close()
            if DEBUG: logger.info(f"[client] {self.episode_uuid} | ZMQ socket closed, service loop terminated.")
            if self.interchange_method == 'ipc':
                if os.path.exists(self.ipc_path):
                    os.remove(self.ipc_path)
                    if DEBUG: logger.info(f"[client] {self.episode_uuid} | IPC socket file {self.ipc_path} removed.")
