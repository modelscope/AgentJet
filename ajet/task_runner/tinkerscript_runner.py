
import atexit
import json
import zmq
import os
from ajet import AjetTuner
from ajet import WorkflowOutput
from ajet.context_tracker.multiagent_tracking import (
    MultiAgentContextTracker,
)
from ajet.context_tracker.basic_tracker import BaseContextTracker
from ajet.schema.task import WorkflowTask
from ajet.schema.trajectory import Reward
from ajet.task_runner.base_runner import BaseAgentRunner
from ajet.tuner_lib.weight_tuner.experimental.interchange_utils import http_register_episode, get_zmq_socket
from loguru import logger
from ajet import Workflow

DEBUG = False

context = zmq.Context()
atexit.register(context.term)

class TinkerScriptRunner(BaseAgentRunner):

    def register_episode_and_wait_output(self, episode_uuid: str, openai_base_url: str, openai_api_key: str) -> WorkflowOutput:
        """Register the episode as ready in the TinkerScript data interchange center."""
        # parse episode_uuid, openai_base_url, openai_api_key
        zmq_listen_result_addr, ipc_path = get_zmq_socket(self.config, episode_uuid, tag="workflow")
        http_register_episode(
            self.config,
            episode_uuid=episode_uuid,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            zmq_listen_result_addr=zmq_listen_result_addr,
        )
        if DEBUG: logger.info(f"zmq_listen_result_addr: {zmq_listen_result_addr}")

        # begin wait for result
        zmq_socket = zmq.Context().socket(zmq.REP)
        zmq_socket.bind(zmq_listen_result_addr)

        # <wait for>:
        #   <from_sourcefile>: ajet/tuner_lib/weight_tuner/experimental/as_tinkerscript_server.py
        #   <from_code>: socket.send_string(workflow_output.model_dump_json())
        #   <expect>: workflow_output: WorkflowOutput
        message = zmq_socket.recv_string()

        logger.success(f"Received workflow output for episode {episode_uuid}")
        zmq_socket.send_string("ack")
        zmq_socket.close()
        if ipc_path and os.path.exists(ipc_path): os.remove(ipc_path)

        return WorkflowOutput(**json.loads(message))


    def execute(self, workflow_task: WorkflowTask) -> BaseContextTracker:
        observation_window = workflow_task.observation_window
        task_thread_index = workflow_task.task_thread_index

        hooks = self.runner_hooks(
            observation_window=observation_window,
            task_thread_index=task_thread_index,
            workflow_task=workflow_task,
        )
        context_tracker = MultiAgentContextTracker(
            llm_inference_fn=self.llm_inference_fn,
            tokenizer=self.tokenizer,
            config=self.config,
            workflow_task = workflow_task,
            **hooks,
        )
        tuner = AjetTuner(
            context_tracker=context_tracker,
            llm_inference_fn=self.llm_inference_fn,
            workflow_cls=Workflow,
            config=self.config,
        )

        baseurl_apikey = tuner.as_oai_baseurl_apikey()
        base_url = baseurl_apikey.base_url
        api_key = baseurl_apikey.api_key

        workflow_output: WorkflowOutput = self.register_episode_and_wait_output(
            episode_uuid=context_tracker.episode_uuid,
            openai_base_url=base_url,
            openai_api_key=api_key,
        )

        if workflow_output.reward is not None:
            raw_reward, is_success = (
                workflow_output.reward,
                workflow_output.is_success,
            )
        else:
            raise ValueError("workflow_output.reward is None in TinkerScriptRunner, this is currently not allowed.")

        workflow_task.gym_env = None  # clear gym env client reference to avoid serialization issue

        assert not isinstance(
            raw_reward, list
        ), "AgentJet will support step reward in future versions."

        # register reward
        # TODO: support multi-step reward
        reward = Reward(
            raw_reward=raw_reward,
            raw_step_reward=None,  # "AgentJet will support step reward in future versions."
            success_rate=1.0 if is_success else 0.0,
            madness=0,
            description="",
        )
        context_tracker.process_reward(reward)
        # generate token before merging
        context_tracker.group_merge()
        # after merging, process and align reward again
        context_tracker.process_reward(reward)
        # mark the thread as ended
        observation_window["step"][task_thread_index] = -1
        tuner.terminate_episode()
        context_tracker.log_metrics = workflow_output.log_metrics
        return context_tracker
