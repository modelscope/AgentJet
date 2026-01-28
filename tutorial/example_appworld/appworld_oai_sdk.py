
from ajet import Workflow, WorkflowOutput, WorkflowTask
from ajet import AjetTuner
from openai.types.chat.chat_completion import ChatCompletion


class ExampleAgentScopeWorkflow(Workflow):

    async def execute(self, workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:

        init_messages = workflow_task.task.init_messages
        if len(init_messages) >= 2:
            first_msg, init_messages = init_messages[0], init_messages[1:]
        else:
            first_msg = {"content": "You're a helpful assistant."}
        interaction_message = [
            {
                "content": first_msg.get("content", "You're a helpful assistant."),
                "role": "system",
            }
        ]
        for msg in init_messages:
            interaction_message.append(
                {
                    "content": msg.get("content", ""),
                    "role": msg.get("role", "user"),
                }
            )

        client = tuner.as_raw_openai_sdk_client()
        env = workflow_task.gym_env
        step = 0
        for step in range(tuner.config.ajet.rollout.multi_turn.max_steps):
            # agentscope deal with interaction message
            reply_message: ChatCompletion = await client.chat.completions.create(interaction_message)
            # env service protocol
            obs, _, terminate, _ = env.step(
                action={"content": reply_message.choices[0].message.content, "role": "assistant"}
            )
            # generate new message from env output
            interaction_message.extend(
                [
                    {
                        "content": reply_message.choices[0].message.content,
                        "role": "assistant",
                    },
                    {
                        "content": obs,
                        "role": "user",
                    }
                ]
            )
            # is terminated?
            if terminate:
                break
            if tuner.get_context_tracker().context_overflow:
                break

        return WorkflowOutput(reward=None, metadata={"total_step": step})
