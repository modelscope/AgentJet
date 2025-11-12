import threading
import importlib
import torch
import copy
import asyncio
from astune.env_service_client.env_client import EnvClient
from astune.workflow_controller.basic_agentflow import BaseAgentFlow
from astune.schema.trajectory import Reward, Trajectory
from astune.protocol.agentscope_protocol import AgentScopeLearnProtocol
from astune.context_manager.cmt_linear import CMTLinear, ExtendedMessage, replace_token_ids, CMTLinear
from astune.schema.trajectory import Sample, Reward
from typing import Any, Dict, List, Union, Tuple
from astune.context_manager.agentscope_cm.cmt_agentscope import ASTuneProxy
from astune.schema.task import Task, TaskLaunchCoreArgument

log_generate_lock = threading.Lock()

class AgentScopeWorkflow(BaseAgentFlow):

    def execute(self, init_messages: List[dict], env: EnvClient, task_core_arg: TaskLaunchCoreArgument) -> CMTLinear:
        obs_window = task_core_arg.obs_window
        task_thread_index = task_core_arg.task_thread_index
        task_batch_index = task_core_arg.task_batch_index
        task_tag = task_core_arg.task_tag
        task_id = task_core_arg.task_id

        # fetch learn protocol
        protocol = self.config.astune.rollout.agentscope_learn_protocol
        module_, class_ = protocol.split('->')
        protocol_cls: AgentScopeLearnProtocol = getattr(importlib.import_module(module_), class_)
        agentscope_protocol = protocol_cls(trainer='astune-trinity', agentflow_name='appworld')  # type: ignore

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
            with log_generate_lock:
                obs_window['token'][task_thread_index] += len(token_array)

        astune_proxy = ASTuneProxy(
            llm_chat_fn=self.llm_chat_fn,
            tokenizer=self.tokenizer,
            config=self.config,
            model_name='astune-proxy',
            api_key='dummy-api-key',
            task_batch_index=task_batch_index,
            task_tag=task_tag,
            task_id=task_id,
            env_step_fn=env_step_fn,
            should_interrupt_fn=should_interrupt_fn,
            generated_token_callback_fn=generated_token_callback_fn,
        )

        astune_proxy.update_agentscope_input_dictionary(task_core_arg=task_core_arg)
        astune_proxy = asyncio.run(agentscope_protocol.agentscope_execute(init_messages, astune_proxy, self.config))
        astune_proxy.update_judge_input_dictionary(task_core_arg=task_core_arg)
        astune_proxy.update_judge_input_dictionary(env=env)
        astune_proxy.update_judge_input_dictionary(grouped_steps=astune_proxy.grouped_steps)

        raw_reward, is_success = astune_proxy.get_judge().compute_reward(
            astune_proxy.get_judge_input_dictionary()
        )

        # evaluate
        reward = Reward(
            raw_reward=raw_reward,
            raw_step_reward=None,
            success_rate=1.0 if is_success else 0.0,
            madness=0,
            description=""
        )
        astune_proxy.process_reward(reward)

        # generate token before merging
        astune_proxy.remove_last_context()
        astune_proxy.task_id = task_id
        astune_proxy.task_tag = task_tag
        astune_proxy.group_merge()
        astune_proxy.process_reward(reward)
        return astune_proxy


