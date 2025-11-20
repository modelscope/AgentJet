import threading
import importlib
import torch
import copy
import asyncio
from astune import ModelTuner, Workflow, WorkflowOutput
from astune.utils.env_service_client.env_client import EnvClient
from astune.task_runner import BaseAgentRunner
from astune.context_tracker.basic_tracker import (
    BasicContextTracker,
    ExtendedMessage,
    replace_token_ids,
    BasicContextTracker,
)
from astune.context_tracker.agentscope_tracker.multiagent_tracking import (
    MultiAgentContextTracking,
)
from astune.schema.trajectory import Reward, Trajectory
from astune.schema.trajectory import Sample, Reward
from astune.schema.task import Task, WorkflowTask
from astune.utils.dynamic_import import dynamic_import
from typing import Any, Dict, List, Union, Tuple


class AgentScopeRunner(BaseAgentRunner):

    def execute(self, env: EnvClient, workflow_task: WorkflowTask) -> BasicContextTracker:
        obs_window = workflow_task.obs_window
        task_thread_index = workflow_task.task_thread_index
        task_batch_index = workflow_task.task_batch_index
        task_tag = workflow_task.task_tag
        task_id = workflow_task.task_id

        workflow_import = self.config.astune.rollout.agentscope_learn_protocol
        workflow_cls = dynamic_import(workflow_import)
        agentscope_workflow: Workflow = workflow_cls(name="astune-trinity")

        hooks = self.agentscope_runner_hooks(
            obs_window=obs_window,
            task_thread_index=task_thread_index,
            workflow_task=workflow_task,
            env=env,
        )
        context_tracker = MultiAgentContextTracking(
            llm_chat_fn=self.llm_chat_fn,
            tokenizer=self.tokenizer,
            config=self.config,
            task_batch_index=task_batch_index,
            task_tag=task_tag,
            task_id=task_id,
            **hooks
        )
        m_tuner = ModelTuner(
            context_tracker=context_tracker,
            llm_chat_fn=self.llm_chat_fn,
            tokenizer=self.tokenizer,
            agentscope_workflow=agentscope_workflow,
            config=self.config,
        )
        workflow_task.gym_env = self.generate_gym_env(
            env, workflow_task.task_env_uuid, task_thread_index, obs_window
        )

        workflow_output: WorkflowOutput = asyncio.run(
            agentscope_workflow.agentscope_execute(workflow_task, m_tuner)
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
        reward = Reward(
            raw_reward=raw_reward,
            raw_step_reward=None,
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
