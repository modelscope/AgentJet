from astune.agentscope_flow import BeyondAgentProxy
from agentscope.message import Msg
from pydantic import BaseModel, Field
from astune.protocol.agentscope_protocol import AgentScopeLearnProtocol

class ExampleAgentScopeLearnProtocol(AgentScopeLearnProtocol):

    trainer: str = Field(default="agentscorpion-trinity")

    async def agentscope_execute(self, init_messages, beyondagent_proxy: BeyondAgentProxy, config):
        from agentscope.agent import ReActAgent
        from agentscope.formatter import DashScopeChatFormatter
        from agentscope.memory import InMemoryMemory

        if len(init_messages) >= 2: first_msg, init_messages = init_messages[0], init_messages[1:]
        else: first_msg = {"content": "You're a helpful assistant."}
        interaction_message = []
        for msg in init_messages:
            interaction_message.append(Msg(name=msg.get("name", "user"), content=msg.get("content", ""), role=msg.get("role", "user")))

        agent = ReActAgent(
            name="Qwen",
            sys_prompt=first_msg['content'],
            model=beyondagent_proxy,  # type: ignore
            # model=beyondagent_proxy: use beyondagent_proxy as model
            formatter=DashScopeChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=None,
            print_hint_msg=False,
        )
        agent.set_console_output_enabled(False)

        for _ in range(config.astune.rollout.multi_turn.max_steps):
            # agentscope deal with interaction message
            reply_message = await agent(interaction_message)
            print(reply_message.content)
            # env service protocol
            obs, _, terminate, _ = beyondagent_proxy.env_step_fn(action={"content": reply_message.content, "role": "assistant"})
            # generate new message from env output
            interaction_message = Msg(name="env", content=obs, role="user")
            # is terminated?
            if terminate: break
            if beyondagent_proxy.context_overflow: break

        return beyondagent_proxy

