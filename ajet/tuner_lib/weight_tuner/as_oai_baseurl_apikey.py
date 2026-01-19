import os
from typing import Any
from pydantic import BaseModel, Field
from ajet.context_tracker.multiagent_tracking import (
    MultiAgentContextTracker,
)
from openai.resources.chat.chat import AsyncChat
from openai.resources.completions import AsyncCompletions
from .experimental.as_oai_model_client import generate_auth_token


class MockAsyncCompletions(AsyncCompletions):
    async def create(self, *args, **kwargs) -> Any: # type: ignore
        return await self._client.create(*args, **kwargs) # type: ignore

class MockAsyncChat(AsyncChat):
    @property
    def completions(self) -> MockAsyncCompletions:  # type: ignore
        return MockAsyncCompletions(self._client)

class OpenaiBaseUrlAndApiKey(BaseModel):
    """ At this layer, we will determine which model to use:
        - training model
        - debug model assigned by user, used when this target is not being trained
    """

    base_url: str = Field(default="http://localhost:27788/v1", description="The base URL for the Ajet's fake OpenAI API")
    api_key: str = Field(default="invalid_apikey", description="The Ajet's fake key, which is not a real key, it is a encoded string contain episode_uuid and other stuff.")
    model: str = Field(default="reserved_field", description="reserved field.")


class OpenaiClientBaseUrlTuner(BaseModel):
    """ At this layer, we will determine which model to use:
        - training model
        - debug model assigned by user, used when this target is not being trained
    """

    base_url: str = Field(default="http://localhost:27788/v1", description="The base URL for the Ajet's fake OpenAI API")
    api_key: str = Field(default="invalid_apikey", description="The Ajet's fake key, which is not a real key, it is a encoded string contain episode_uuid and other stuff.")
    model: str = Field(default="reserved_field", description="reserved field.")

    def __init__(
        self,
        config,
        context_tracker: MultiAgentContextTracker,
        target_tag: str,
        agent_name: str,
        episode_uuid: str,
        episode_contect_address: str,
        **kwargs,
    ):

        port = os.getenv("AJET_DAT_INTERCHANGE_PORT")
        if config.ajet.interchange_server.interchange_server_port != 'auto':
            port = str(int(config.ajet.interchange_server.interchange_server_port))

        assert port is not None, "AJET_DAT_INTERCHANGE_PORT env var must be set"
        master_node_ip = os.getenv("MASTER_NODE_IP", "localhost")

        base_url = f"http://{master_node_ip}:{port}/v1"
        api_key = generate_auth_token(
            agent_name=agent_name,
            target_tag=target_tag,
            episode_uuid=episode_uuid,
            episode_address=episode_contect_address,
        )
        model = "reserved_field"

        # Properly initialize the Pydantic BaseModel
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
