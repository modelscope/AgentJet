import time
import os

from loguru import logger
from astune.env_service_client.env_client import EnvClient
from astune.utils.utils import convert_tool_to_user_message
from astune.schema.trajectory import Reward
from astune.context_manager.agentflow_cm.cmt_linear import CMTLinear, ExtendedMessage
from astune.context_manager.agentflow_cm.cmt_linear_think import LinearThinkCMT
from astune.context_manager.agentflow_cm.cmt_context_clip import SelfContextClipCMT
from astune.context_manager.agentflow_cm.cmt_sliding_window import SlidingWindowCMT
from astune.workflow_controller.basic_agentflow import BaseAgentFlow
from typing import Any, Dict, List, Union, Callable
from beast_logger import print_listofdict
import threading

log_generate_lock = threading.Lock()



class AgentFlow(BaseAgentFlow):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_step_reward_from_env: bool = self.config.astune.rollout.get("use_step_reward_from_env", False)
        self.step_reward = []


    def execute(self, init_messages: List[dict], env: EnvClient, task_core_arg) -> CMTLinear:
        obs_window = task_core_arg.obs_window
        task_thread_index = task_core_arg.task_thread_index

        # 1. 🚀 Initialize messages
        if self.config.astune.context_manager.context_manager_type == "linear":
            self.cmt = CMTLinear(self.config, self.tokenizer)
        elif self.config.astune.context_manager.context_manager_type == "linear_think":
            self.cmt = LinearThinkCMT(self.config, self.tokenizer)
        elif self.config.astune.context_manager.context_manager_type == "context_selfclip":
            self.cmt = SelfContextClipCMT(self.config, self.tokenizer, self.llm_chat_fn)
        elif self.config.astune.context_manager.context_manager_type == "sliding_window":
            self.cmt = SlidingWindowCMT(self.config, self.tokenizer, self.llm_chat_fn)
        else:
            raise ValueError(f"Unsupported context template: {self.config.astune.context_manager.context_manager_type}")

        assert not (self.config.astune.rollout.force_think and self.config.astune.rollout.force_no_think), "Cannot force both think and no_think"
        add_nothink = self.config.astune.rollout.force_no_think

        self.cmt.save_init_input(init_messages, add_nothink)

        request_id: str = ""
        for act_step in range(self.max_steps):
            # 2. 🔄 Update thread progress
            obs_window['step'][task_thread_index] = act_step
            if (obs_window['stop'] is not None) and obs_window['stop'][task_thread_index]: # Check if the thread should obs_window['stop'] (because other threads have completed, making this thread useless)
                self.cmt.discarded = True
                break

            # 3. ⏮️ get previous steps
            try:
                step_input_message_arr = self.cmt.prepare_next_llm_context()
            except Exception as e:
                print_listofdict(self.cmt.to_role_content(self.cmt.full_context), mod='exception', header="Before Crash")
                raise e

            # 4. ⚠️ check token overflow
            is_safe, info = self.cmt.check_context_token_num_safe(step_input_message_arr)
            if not is_safe:
                logger.warning(f"[{info}] detected at step {act_step}. Current token count exceeds the limit.")
                self.cmt.is_terminated = True
                break

            # 5. 🤖 call llm
            llm_output = self.llm_chat_fn(step_input_message_arr, request_id=request_id)
            if (obs_window['stop'] is not None) and obs_window['stop'][task_thread_index]:  # Check if the thread should obs_window['stop'] (because other threads have completed, making this thread useless)
                self.cmt.discarded = True
                break

            # 6. 💾 save llm output
            self.cmt.save_llm_output(llm_output, input_msg_ref=step_input_message_arr)
            obs_window['token'][task_thread_index] += self.cmt.generated_token_cnt

            # 7. 🌍 world interaction
            try:
                env_output = env.step(
                    instance_id=task_core_arg.task_env_uuid,
                    action={"content": self.cmt.prepare_world_interaction(), "role": "assistant"},
                    params={"step_skip_action": self.config.astune.rollout.step_skip_action}
                )
                if env_output["state"]["role"] == "tool":
                    env_output["state"] = convert_tool_to_user_message(env_output["state"], self.tokenizer, format="qwen")
                # if self.console_debug_mode:
                #     if isinstance(env_output["state"], dict):
                #         print_listofdict(
                #             step_input_message_arr +
                #             [{'role': 'llm_latest', 'content': llm_output['content']}] +
                #             [{'role': 'env',        'content': env_output["state"]['content']}]
                #         , mod='c')
            except Exception as e:
                logger.bind(exception=True).exception(f"call env.step error with {e}")
                self.cmt.is_terminated = True
                state = {"content": str(e), "role": "user"}
                env_output = {
                    "reward": 0,
                    "is_terminated": True,
                    "state": state,
                }

            # 8. 📥 save environment output
            state = env_output["state"]
            state.pop('tool_calls', None)
            self.cmt.save_env_output(state, input_msg_ref=step_input_message_arr, add_nothink=add_nothink)
            self.cmt.round_cnt += 1
            if self.use_step_reward_from_env:
                self.step_reward += [env_output["reward"]]

            # 9. 🔚 determine if the episode is terminated
            self.cmt.is_terminated = env_output["is_terminated"]
            if self.cmt.is_terminated:
                break

        self.cmt.ensure_terminate_rollout_stage()
        obs_window['step'][task_thread_index] = -1
        raw_reward = 0
        raw_reward = env.evaluate(task_core_arg.task_env_uuid, params={"sparse": False})
        if raw_reward >= 1:
            success_rate = 1.0
        else:
            success_rate = 0.0
        if not self.use_step_reward_from_env:
            if self.config.astune.rollout.add_special_success_reward:
                if success_rate == 1:
                    raw_reward = 1.0 + raw_reward * 0.5
                else:
                    raw_reward = 0.0 + raw_reward * 0.5
            if self.config.astune.rollout.binary_reward:
                raw_reward = success_rate
            self.cmt.process_reward(
                reward_structure = Reward(
                    raw_reward=raw_reward,
                    raw_step_reward=None,
                    success_rate=success_rate,
                    madness=0,
                    description="Success=1, Failure=0"
                )
            )
        else:
            self.cmt.process_reward(
                reward_structure = Reward(
                    raw_reward=raw_reward,
                    raw_step_reward=self.step_reward,
                    success_rate=success_rate,
                    madness=0,
                    description="Step Reward from Environment"
                )
            )

        self.cmt.remove_last_context()

        return self.cmt
