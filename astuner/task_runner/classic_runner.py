from beast_logger import print_listofdict
from loguru import logger

from astuner.context_tracker.basic_tracker import BaseContextTracker
from astuner.schema.trajectory import Reward
from astuner.utils.utils import convert_tool_to_user_message
from .base_runner import BaseAgentRunner


class AgentRunner(BaseAgentRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self, workflow_task) -> BaseContextTracker:
        observation_window = workflow_task.observation_window
        task_thread_index = workflow_task.task_thread_index
        init_messages = workflow_task.init_messages
        env = workflow_task.gym_env

        # 1. 🚀 Initialize messages
        if self.config.astuner.context_tracker.context_tracker_type == "linear":
            self.tracker = BaseContextTracker(self.config, self.tokenizer)
        else:
            raise ValueError(
                f"Unsupported context template: {self.config.astuner.context_tracker.context_tracker_type}"
            )

        add_nothink = False
        self.tracker.save_init_input(init_messages, add_nothink)

        request_id: str = ""
        for act_step in range(self.max_steps):
            # 2. 🔄 Update thread progress
            observation_window["step"][task_thread_index] = act_step
            if (observation_window["stop"] is not None) and observation_window["stop"][
                task_thread_index
            ]:  # Check if the thread should observation_window['stop'] (because other threads have completed, making this thread useless)
                self.tracker.discarded = True
                break

            # 3. ⏮️ get previous steps
            try:
                step_input_message_arr = self.tracker.prepare_next_llm_context()
            except Exception as e:
                print_listofdict(
                    self.tracker.to_role_content(self.tracker.full_context),
                    mod="exception",
                    header="Before Crash",
                )
                raise e

            # 4. ⚠️ check token overflow
            is_safe, token_overflow, info = self.tracker.check_context_token_num_safe(
                step_input_message_arr
            )
            if not is_safe:
                logger.warning(
                    f"[{info}] detected at step {act_step}. Current token count exceeds the limit."
                )
                self.tracker.is_terminated = True
                break

            # 5. 🤖 call llm
            llm_output = self.llm_inference_fn(step_input_message_arr, request_id=request_id)
            if (observation_window["stop"] is not None) and observation_window["stop"][
                task_thread_index
            ]:  # Check if the thread should observation_window['stop'] (because other threads have completed, making this thread useless)
                self.tracker.discarded = True
                break

            # 6. 💾 save llm output
            self.tracker.save_llm_output(llm_output, input_msg_ref=step_input_message_arr)
            observation_window["token"][task_thread_index] += self.tracker.generated_token_cnt

            # 7. 🌍 world interaction
            try:
                env_output = env.step(
                    instance_id=workflow_task.task_env_uuid,
                    action={
                        "content": self.tracker.prepare_world_interaction(),
                        "role": "assistant",
                    },
                    params={"step_skip_action": self.config.astuner.rollout.step_skip_action},
                )
                if env_output["state"]["role"] == "tool":
                    env_output["state"] = convert_tool_to_user_message(
                        env_output["state"], self.tokenizer, format="qwen"
                    )
            except Exception as e:
                logger.bind(exception=True).exception(f"call env.step error with {e}")
                self.tracker.is_terminated = True
                state = {"content": str(e), "role": "user"}
                env_output = {
                    "reward": 0,
                    "is_terminated": True,
                    "state": state,
                }

            # 8. 📥 save environment output
            state = env_output["state"]
            state.pop("tool_calls", None)  # type: ignore
            self.tracker.save_env_output(state, input_msg_ref=step_input_message_arr, add_nothink=add_nothink)  # type: ignore
            self.tracker.round_cnt += 1

            # 9. 🔚 determine if the episode is terminated
            self.tracker.is_terminated = env_output["is_terminated"]
            if self.tracker.is_terminated:
                break

        observation_window["step"][task_thread_index] = -1
        raw_reward = 0
        raw_reward = env.evaluate(workflow_task.task_env_uuid, params={"sparse": False})
        if raw_reward >= 1:
            success_rate = 1.0
        else:
            success_rate = 0.0

        # TODO: support multi-step reward
        self.tracker.process_reward(
            reward_structure=Reward(
                raw_reward=raw_reward,
                raw_step_reward=None,  # we do not support step reward yet
                success_rate=success_rate,
                madness=0,
                description="Success=1, Failure=0",
            )
        )
        self.tracker.remove_last_context()
        return self.tracker
