from typing import List
from loguru import logger
from pydantic import BaseModel, Field
from ajet import AjetTuner, Workflow, WorkflowOutput, WorkflowTask

import os
import requests


# ------------------------------------------------------
# Simple version - no tool call
# ------------------------------------------------------


class DeepResearchInputSchema(BaseModel):
    base_url: str = Field(default="", description="The base URL of the OpenAI-compatible API.")
    api_key: str = Field(default="", description="The API key for authentication.")
    init_messages: List[dict] = Field(default=[], description="The initial messages for the deep research task.")
    task_id: str = Field(default="", description="The unique identifier for the research task.")
    main_query: str = Field(default="", description="The main query for the research task.")
    max_steps: int = Field(default=20, description="The maximum number of steps for the research task.")
    env_service_url: str = Field(default="", description="The URL of the environment service.")


class ExampleMaDeepResearch(Workflow):
    name: str = "multiagent_deep_research_workflow"

    async def execute(self, workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:  # type: ignore
        # Extract base URL and API key from the tuner
        url_and_apikey = tuner.as_oai_baseurl_apikey()
        base_url = url_and_apikey.base_url
        api_key = url_and_apikey.api_key
        init_messages = workflow_task.task.init_messages

        # Get the AGENT_SERVER_URL from environment variables or use a default value
        agent_server_url = os.getenv("AGENT_SERVER_URL", "http://localhost:8000")

        # Prepare the payload using DeepResearchInputSchema
        payload = DeepResearchInputSchema(
            base_url=base_url,
            api_key=api_key,
            init_messages=init_messages,
            task_id=workflow_task.task.task_id,
            main_query=workflow_task.task.main_query,
            max_steps=tuner.config.astune.rollout.multi_turn.max_steps,
            env_service_url=workflow_task.gym_env.service_url,
        )

        try:
            # Send the HTTP POST request to the AGENT_SERVER_URL
            headers = {
                "Content-Type": "application/json",
            }

            response = requests.post(
                agent_server_url,
                headers=headers,
                data=payload.model_dump(),
            )

            # Check if the request was successful
            if response.status_code == 200:
                result_data = response.json()
                logger.info(f"Successfully received response: {result_data}")
                result = WorkflowOutput(**result_data)
                return result

        except Exception as e:
            logger.error(f"An error occurred while sending the request: {e}")
