
import atexit
import json
import zmq
import os
from ajet.tuner import AjetTuner
from ajet.schema.task import WorkflowOutput
from ajet.context_tracker.multiagent_tracking import MultiAgentContextTracker
from ajet.context_tracker.basic_tracker import BaseContextTracker
from ajet.schema.task import WorkflowTask
from ajet.schema.trajectory import Reward
from ajet.task_runner.base_runner import BaseAgentRunner
from ajet.utils.retry import SwarmReceiveAbortException
from ajet.tuner_lib.weight_tuner.experimental.interchange_utils import http_register_episode, get_zmq_socket, is_episode_claimed
from loguru import logger
from ajet import Workflow
from typing import Callable

# DEBUG = True
DEBUG = False

context = zmq.Context()
atexit.register(context.term)

class SwarmRunner(BaseAgentRunner):

    def register_episode_and_wait_output(
        self,
        episode_uuid: str,
        openai_base_url: str,
        openai_api_key: str,
        context_tracker: BaseContextTracker,
        tuner:AjetTuner,
        should_exit_soft:Callable,
        should_exit_hard:Callable
    ) -> WorkflowOutput | None:
        """Register the episode as ready in the Swarm data interchange center."""
        # parse episode_uuid, openai_base_url, openai_api_key
        zmq_listen_result_addr, ipc_path = get_zmq_socket(self.config, episode_uuid, tag="workflow")
        success = http_register_episode(
            self.config,
            episode_uuid=episode_uuid,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            zmq_listen_result_addr=zmq_listen_result_addr,
            should_exit_soft=should_exit_soft,
        )
        if not success:
            return None # type: ignore

        if DEBUG: logger.info(f"zmq_listen_result_addr: {zmq_listen_result_addr}")

        # begin wait for result
        zmq_socket = zmq.Context().socket(zmq.REP)
        zmq_socket.bind(zmq_listen_result_addr)
        zmq_socket.setsockopt(zmq.RCVTIMEO, 1*1000)  # 1 second timeout for REP

        speicial_messages = [
            "RUNNER.SPECIAL.RESET_CONTEXT_TRACKER",
            "RUNNER.SPECIAL.ABORT"
        ]

        try:

            while True:
                # <wait for 1/2>:
                #   <from_sourcefile>: ajet/tuner_lib/weight_tuner/experimental/as_swarm_server.py
                #   <from_code>: socket.send_string(workflow_output.model_dump_json())
                #   <expect>: workflow_output: WorkflowOutput
                # <wait for 2/2>:
                #   <from_sourcefile>: ajet/tuner_lib/weight_tuner/experimental/as_swarm_server.py
                #   <from_code>: socket.send_string("RUNNER.SPECIAL.RESET_CONTEXT_TRACKER")
                #   <expect>: "RUNNER.SPECIAL.RESET_CONTEXT_TRACKER"
                try:
                    message = zmq_socket.recv_string()
                except zmq.Again as e:
                    if should_exit_hard():
                        # logger.warning(f'{episode_uuid} Exiting workflow due to should_exit_hard signal.')
                        context_tracker.reset()
                        raise SwarmReceiveAbortException(f"Episode {episode_uuid} aborted due to system exit.")
                    else:
                        continue
                # process messages
                if message not in speicial_messages:
                    zmq_socket.send_string("ack")
                    break
                elif message == "RUNNER.SPECIAL.RESET_CONTEXT_TRACKER":
                    logger.warning(f"Received reset command for episode {episode_uuid}.")
                    context_tracker.reset()
                    zmq_socket.send_string("ack")
                    continue
                elif message == "RUNNER.SPECIAL.ABORT":
                    logger.warning(f"Received abort command for episode {episode_uuid}.")
                    context_tracker.reset()
                    zmq_socket.send_string("ack")
                    return None
                else:
                    raise RuntimeError(f"Unknown special message received: {message}")

            final_output = WorkflowOutput(**json.loads(message))
            reward = final_output.reward
            logger.success(f"Received workflow output for episode {episode_uuid} (Reward: {reward})")

        except Exception as exc:
            raise exc

        finally:
            tuner.terminate_episode()   # this is very important to avoid resource leak
            zmq_socket.close()
            if ipc_path and os.path.exists(ipc_path): os.remove(ipc_path)

        return final_output


    def execute(self, workflow_task: WorkflowTask) -> BaseContextTracker:

        observation_window = workflow_task.observation_window
        task_thread_index = workflow_task.task_thread_index

        hooks = self.runner_hooks(
            observation_window=observation_window,
            task_thread_index=task_thread_index,
            workflow_task=workflow_task,
        )

        should_exit_soft = hooks['should_interrupt_soft_fn']
        should_exit_hard = hooks['should_interrupt_hard_fn']

        if should_exit_soft() or should_exit_hard():
            # print(f'Exiting workflow worker due to interrupt signal for episode {workflow_task.episode_uuid}.')
            raise SwarmReceiveAbortException(f"Episode {workflow_task.episode_uuid} aborted due to interrupt signal.")

        # context tracker will trace and gather everything we need for training
        context_tracker = MultiAgentContextTracker(
            llm_inference_fn=self.llm_inference_fn,
            tokenizer=self.tokenizer,
            config=self.config,
            workflow_task = workflow_task,
            **hooks,
        )
        # tuner will handle the communication and provide `baseurl_apikey`
        tuner = AjetTuner(
            context_tracker=context_tracker,
            llm_inference_fn=self.llm_inference_fn,
            workflow_cls=Workflow,
            config=self.config,
        )

        # from tuner, we get base_url and api_key
        baseurl_apikey = tuner.as_oai_baseurl_apikey()

        base_url = baseurl_apikey.base_url
        api_key = baseurl_apikey.api_key

        # wait for remote client to return workflow output
        workflow_output: WorkflowOutput | None = self.register_episode_and_wait_output(
            episode_uuid=context_tracker.episode_uuid,
            openai_base_url=base_url,
            openai_api_key=api_key,
            context_tracker=context_tracker,
            tuner=tuner,
            should_exit_soft=should_exit_soft,
            should_exit_hard=should_exit_hard,
        )
        if not workflow_output:
            return None  # type: ignore

        # the most important thing is to fix task_id to client task_id, set task_id to workflow_task and context_tracker task_id
        assert "task_id" in workflow_output.metadata, "workflow_output.metadata must contain task_id"
        task_id = workflow_output.metadata.get("task_id", "")
        workflow_task.task_id = task_id
        context_tracker.task_id = task_id

        # process reward
        if workflow_output.reward is not None:
            raw_reward, is_success = (
                workflow_output.reward,
                workflow_output.is_success,
            )
        else:
            raise ValueError("workflow_output.reward is None in SwarmRunner, this is currently not allowed.")

        # release gym_env
        workflow_task.gym_env = None  # clear gym env client reference to avoid serialization issue

        # check reward
        assert not isinstance(raw_reward, list), "AgentJet will support step reward in future versions."

        # register reward
        # TODO: support multi-step reward
        reward = Reward(
            raw_reward=raw_reward,
            raw_step_reward=None,  # "AgentJet will support step reward in future versions."
            success_rate=1.0 if is_success else 0.0,
            madness=0,
            description="",
        )
        # process reward
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
