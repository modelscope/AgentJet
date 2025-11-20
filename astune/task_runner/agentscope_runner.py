import threading
import importlib
import torch
import copy
import asyncio
from astune import ModelTuner, Workflow, WorkflowOutput
from astune.utils.env_service_client.env_client import EnvClient
from astune.task_runner import BaseAgentRunner
from astune.context_tracker.basic_tracker import BasicContextTracker, ExtendedMessage, replace_token_ids, BasicContextTracker
from astune.context_tracker.agentscope_tracker.multiagent_tracking import MultiAgentContextTracking
from astune.schema.trajectory import Reward, Trajectory
from astune.schema.trajectory import Sample, Reward
from astune.schema.task import Task, WorkflowTask
from astnue.task_judge.judge_base import JudgeBase
from astune.utils.dynamic_import import dynamic_import
from typing import Any, Dict, List, Union, Tuple


class RunnerWithCallback(BaseAgentRunner):

    def agentscope_runner_hooks(self, obs_window, task_thread_index, task_core_arg, env):

        def env_step_fn(action: dict) -> Tuple[str, float, bool, dict]:
            obs_window['step'][task_thread_index] += 1
            env_output = env.step(
                instance_id=task_core_arg.task_env_uuid,
                action=action,
            )
            obs = ""
            assert isinstance(env_output, dict)
            if ('content' not in env_output["state"]) and ('error' in env_output["state"]):
                obs = f"[Error from environment: {env_output['error']}]"
            elif (env_output["state"]['content']==""):
                obs = 'Warning: the environment does not provide any feedback, please provide valid inpu and try again.'
            else:
                obs = env_output["state"]['content']
            reward = 0
            info = {}
            terminate = env_output["is_terminated"]
            return obs, reward, terminate, info

        def should_interrupt_fn() -> bool:
            if (obs_window['stop'] is not None) and obs_window['stop'][task_thread_index]: # Check if the thread should stop (because other threads have completed, making this thread useless)
                return True
            return False

        def generated_token_callback_fn(token_array):
            obs_window['token'][task_thread_index] += len(token_array)

        return {
            "env_step_fn":env_step_fn,
            "should_interrupt_fn":should_interrupt_fn,
            "generated_token_callback_fn":generated_token_callback_fn
        }


    def get_judge(self) -> JudgeBase:
        judge_protocol = self.config.astune.task_judge.judge_protocol
        return dynamic_import(judge_protocol)(self.config)  # type: ignore


class AgentScopeRunner(RunnerWithCallback):

    def execute(self, env: EnvClient, task_core_arg: WorkflowTask) -> BasicContextTracker:
        obs_window = task_core_arg.obs_window
        task_thread_index = task_core_arg.task_thread_index
        task_batch_index = task_core_arg.task_batch_index
        task_tag = task_core_arg.task_tag
        task_id = task_core_arg.task_id

        workflow_import = self.config.astune.rollout.agentscope_learn_protocol
        workflow_cls = dynamic_import(workflow_import)()
        agentscope_workflow: Workflow = workflow_cls(trainer='astune-trinity', AgentRunner_name='appworld')

        context_tracker = MultiAgentContextTracking(
            llm_chat_fn=self.llm_chat_fn,
            tokenizer=self.tokenizer,
            config=self.config,
            task_batch_index=task_batch_index,
            task_tag=task_tag,
            task_id=task_id,
            **self.agentscope_runner_hooks(
                obs_window=obs_window,
                task_thread_index=task_thread_index,
                task_core_arg=task_core_arg,
                env=env
            )
        )
        astune_proxy = ModelTuner(
            context_tracker=context_tracker,
            llm_chat_fn=self.llm_chat_fn,
            tokenizer=self.tokenizer,
            agentscope_workflow=agentscope_workflow,
            config=self.config,
        )

        workflow_output: WorkflowOutput = asyncio.run(agentscope_workflow.agentscope_execute(task_core_arg, astune_proxy))
        if workflow_output.reward is not None:
            raw_reward, is_success = workflow_output.reward, workflow_output.is_success
        else:
            raw_reward, is_success = self.get_judge().compute_reward(workflow_output.metadata)

        assert not isinstance(raw_reward, list), "ASTune will support step reward in future versions."

        # register reward
        reward = Reward(
            raw_reward=raw_reward,
            raw_step_reward=None,
            success_rate=1.0 if is_success else 0.0,
            madness=0,
            description=""
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


