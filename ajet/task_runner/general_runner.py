import asyncio
from venv import logger

from ajet import AjetTuner
from ajet import Workflow, WorkflowOutput
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
        task_batch_index = workflow_task.task_batch_index
        task_tag = workflow_task.task_tag
        task_id = workflow_task.task_id

        workflow_import = self.config.ajet.rollout.user_workflow
        workflow_cls = dynamic_import(workflow_import)
        user_workflow: Workflow = workflow_cls(name="ajet-trinity")

        hooks = self.runner_hooks(
            observation_window=observation_window,
            task_thread_index=task_thread_index,
            workflow_task=workflow_task,
        )
        context_tracker = MultiAgentContextTracker(
            llm_inference_fn=self.llm_inference_fn,
            tokenizer=self.tokenizer,
            config=self.config,
            task_batch_index=task_batch_index,
            task_tag=task_tag,
            task_id=task_id,
            episode_uuid=workflow_task.episode_uuid,
            **hooks,
        )
        tuner = AjetTuner(
            context_tracker=context_tracker,
            llm_inference_fn=self.llm_inference_fn,
            user_workflow=user_workflow,
            config=self.config,
        )

        workflow_output: WorkflowOutput = asyncio.run(
            user_workflow.execute(workflow_task, tuner)
        )
        # set workflow metadata to context tracker metadata
        context_tracker.workflow_metadata = workflow_output.metadata
        if workflow_output.reward is not None:
            raw_reward, is_success = (
                workflow_output.reward,
                workflow_output.is_success,
            )
        else:
            raw_reward, is_success = self.get_judge().compute_reward(workflow_task, workflow_output)
            
            # ✅ Critical Fix: After calling `judge`, write the updated `reward_stats` back to `workflow_metadata`
            # # Ensure that `native_compat_trainer` reads the actual value calculated by `judge`, not the 0 value returned by `env`.
            if workflow_output.metadata and 'reward_stats' in workflow_output.metadata:
                context_tracker.workflow_metadata['reward_stats'] = workflow_output.metadata['reward_stats']
            else:
                # fallback: If the judge does not update reward_stats, use the default value.
                logger.warning(f"[WARN] reward_stats not found in metadata after judge call, creating default values")
                default_reward_stats = {
                    'original_reward': raw_reward,  
                    'penalty': 0.0,
                    'step_reward': raw_reward,
                }
                if workflow_output.metadata:
                    workflow_output.metadata['reward_stats'] = default_reward_stats
                    context_tracker.workflow_metadata['reward_stats'] = default_reward_stats
                else:
                    context_tracker.workflow_metadata = {'reward_stats': default_reward_stats}
        
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
        context_tracker.task_id = task_id
        context_tracker.task_tag = task_tag
        context_tracker.group_merge()
        # after merging, process and align reward again
        context_tracker.process_reward(reward)
        # mark the thread as ended
        observation_window["step"][task_thread_index] = -1
        tuner.terminate_episode()
        return context_tracker
