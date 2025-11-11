from agentscope.message import Msg
from pydantic import BaseModel, Field
from typing import Callable, List
try: from astune.agentscope_flow import ASTuneProxy
except ImportError: pass

class AgentScopeLearnProtocol(BaseModel):
    model_config = {"extra": "allow"}
    # Trainer to use; default "trinity". Optional: "agentscorpion-trinity".
    trainer: str = Field(default="trinity")
    # Experiment name
    agentflow_name: str = Field(default="agent-flow")
    # In multi-agent settings, specify the list of trainable agent target names
    trainable_agent_targets: List[str] = Field(default=[])
    # Use dataset provided by the trainer (True: read each query from workflow input; False: AgentScope handles each query)
    external_dataset: bool = Field(default=True)
    # Use external environment provided by the trainer (True: read environment handle from input; False: AgentScope runs environment and tools)
    external_environment: bool = Field(default=True)
    # Use external reward provided by the trainer (True: compute reward outside AgentScope after workflow; False: AgentScope computes reward)
    external_reward: bool = Field(default=True)
    # Other settings
    multiturn_token_consolidation: bool = Field(default=True)

    async def agentscope_execute(self, init_messages, astune_proxy: "ASTuneProxy", config)->"ASTuneProxy":
        raise NotImplementedError

