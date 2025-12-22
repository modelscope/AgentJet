from agentscope.message import Msg
from pydantic import Field

from astuner import ModelTuner, Workflow, WorkflowOutput, WorkflowTask


class ExampleAgentScopeWorkflow(Workflow):
    trainer: str = Field(default="astuner-trinity")

    async def execute(
        self, workflow_task: WorkflowTask, model_tuner: ModelTuner
    ) -> WorkflowOutput:
        from agentscope.agent import ReActAgent
        from agentscope.formatter import DashScopeChatFormatter
        from agentscope.memory import InMemoryMemory

        init_messages = workflow_task.task.init_messages
        if len(init_messages) >= 2:
            first_msg, init_messages = init_messages[0], init_messages[1:]
        else:
            first_msg = {"content": "You're a helpful assistant."}
        interaction_message = []
        for msg in init_messages:
            interaction_message.append(
                Msg(
                    name=msg.get("name", "user"),
                    content=msg.get("content", ""),
                    role=msg.get("role", "user"),
                )
            )

        agent = ReActAgent(
            name="Qwen",
            sys_prompt=first_msg["content"],
            model=model_tuner,
            formatter=DashScopeChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=None,
            print_hint_msg=False,
        )
        agent.set_console_output_enabled(False)
        env = workflow_task.gym_env
        step = 0
        for step in range(model_tuner.config.astuner.rollout.multi_turn.max_steps):
            # agentscope deal with interaction message
            reply_message = await agent(interaction_message)
            # env service protocol
            obs, _, terminate, _ = env.step(
                action={"content": reply_message.content, "role": "assistant"}
            )
            # generate new message from env output
            interaction_message = Msg(name="env", content=obs, role="user")
            # is terminated?
            if terminate:
                break
            if model_tuner.get_context_tracker().context_overflow:
                break

        return WorkflowOutput(reward=None, metadata={"total_step": step})
