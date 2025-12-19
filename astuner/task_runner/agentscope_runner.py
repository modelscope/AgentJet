import asyncio

from astuner import ModelTuner, Workflow, WorkflowOutput
from astuner.context_tracker.agentscope_tracker.multiagent_tracking import (
    MultiAgentContextTracker,
)
from astuner.context_tracker.basic_tracker import BaseContextTracker
from astuner.schema.task import WorkflowTask
from astuner.schema.trajectory import Reward
from astuner.task_runner import BaseAgentRunner
from astuner.utils.dynamic_import import dynamic_import


class AgentScopeRunner(BaseAgentRunner):
    def execute(self, workflow_task: WorkflowTask) -> BaseContextTracker:
        observation_window = workflow_task.observation_window
        task_thread_index = workflow_task.task_thread_index
        task_batch_index = workflow_task.task_batch_index
        task_tag = workflow_task.task_tag
        task_id = workflow_task.task_id

        workflow_import = self.config.astuner.rollout.agentscope_workflow
        workflow_cls = dynamic_import(workflow_import)
        agentscope_workflow: Workflow = workflow_cls(name="astuner-trinity")

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
            **hooks,
        )
        m_tuner = ModelTuner(
            context_tracker=context_tracker,
            llm_inference_fn=self.llm_inference_fn,
            tokenizer=self.tokenizer,
            agentscope_workflow=agentscope_workflow,
            config=self.config,
        )

        workflow_output: WorkflowOutput = asyncio.run(
            agentscope_workflow.execute(workflow_task, m_tuner)
        )
        if workflow_output.reward is not None:
            raw_reward, is_success = (
                workflow_output.reward,
                workflow_output.is_success,
            )
        else:
            raw_reward, is_success = self.get_judge().compute_reward(workflow_task, workflow_output)
        workflow_task.gym_env = None  # clear gym env client reference to avoid serialization issue

        assert not isinstance(
            raw_reward, list
        ), "ASTune will support step reward in future versions."

        # register reward
        # TODO: support multi-step reward
        reward = Reward(
            raw_reward=raw_reward,
            raw_step_reward=None,  # "ASTune will support step reward in future versions."
            success_rate=1.0 if is_success else 0.0,
            madness=0,
            description="",
        )
        context_tracker.process_reward(reward)
        # generate token before merging
        context_tracker.remove_last_context()
        context_tracker.task_id = task_id
        context_tracker.task_tag = task_tag
        context_tracker.group_merge()
        # after merging, process and align reward again
        context_tracker.process_reward(reward)
        return context_tracker
