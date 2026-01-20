
from ajet import AjetTuner
from ajet import WorkflowOutput
from ajet.context_tracker.multiagent_tracking import (
    MultiAgentContextTracker,
)
from ajet.context_tracker.basic_tracker import BaseContextTracker
from ajet.schema.task import WorkflowTask
from ajet.schema.trajectory import Reward
from ajet.task_runner.base_runner import BaseAgentRunner
from ajet.utils.dynamic_import import dynamic_import


class GeneralRunner(BaseAgentRunner):

    def execute(self, workflow_task: WorkflowTask) -> BaseContextTracker:
        observation_window = workflow_task.observation_window
        task_thread_index = workflow_task.task_thread_index

        workflow_import = self.config.ajet.rollout.user_workflow
        workflow_cls = dynamic_import(workflow_import)

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
            workflow_cls=workflow_cls,
            config=self.config,
        )

        # run workflow
        # user_workflow: Workflow = workflow_cls(name="ajet-workflow")
        workflow_output: WorkflowOutput = self.run_user_workflow(
            workflow_cls,
            workflow_task,
            tuner,
        )

        if workflow_output.reward is not None:
            raw_reward, is_success = (
                workflow_output.reward,
                workflow_output.is_success,
            )
        else:
            raw_reward, is_success = self.get_judge().compute_reward(workflow_task, workflow_output)
            # Sync reward_stats from metadata to log_metrics after judge computation

            if "reward_stats" in workflow_output.metadata:

                workflow_output.log_metrics["reward_stats"] = workflow_output.metadata["reward_stats"]
                
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
