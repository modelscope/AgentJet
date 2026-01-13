
import asyncio
import json
import threading
import os
import time
from loguru import logger
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse
from openai.types.chat.chat_completion import ChatCompletion

import pickle
import httpx
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

import base64
import json

if TYPE_CHECKING:
    from ajet.context_tracker.multiagent_tracking import MultiAgentContextTracker

def generate_auth_token(agent_name, target_tag, episode_uuid):
    """
    Generate a Base64-encoded auth_token from the given agent_name, target_tag, and episode_uuid.

    Args:
        agent_name (str): The name of the agent.
        target_tag (str): The target tag.
        episode_uuid (str): The UUID of the episode.

    Returns:
        str: The generated auth_token in the format "Bearer <base64_encoded_string>".
    """
    # Step 1: Construct the auth_data dictionary
    auth_data = {
        "agent_name": agent_name,
        "target_tag": target_tag,
        "episode_uuid": episode_uuid
    }

    # Step 2: Convert the dictionary to a JSON string
    json_string = json.dumps(auth_data)

    # Step 3: Encode the JSON string into Base64
    base64_encoded = base64.b64encode(json_string.encode('utf-8')).decode('utf-8')

    # Step 4: Prepend "Bearer " to the Base64-encoded string
    auth_token = f"Bearer {base64_encoded}"

    return auth_token


class InterchangeClient:

    def __init__(self, episode_uuid: str, context_tracker: "MultiAgentContextTracker", llm_inference_fn, config):
        self.episode_uuid = episode_uuid
        self.context_tracker = context_tracker
        self.llm_inference_fn = llm_inference_fn
        self.config = config
        self._should_terminate = False
        self.begin_service()


    async def llm_infer(
            self,
            req: ChatCompletionRequest,
            timeline_uuid: str,
            agent_name: str,
            target_tag: str,
            episode_uuid: str,
        ) -> ChatCompletion:
        from ajet.task_rollout.async_llm_bridge import OpenaiLlmProxyWithTracker

        req_as_dict = req.model_dump()

        self.llm_proxy_with_tracker = OpenaiLlmProxyWithTracker(
            context_tracker=self.context_tracker,
            config=self.config,
            llm_inference_fn=self.llm_inference_fn,
        )

        # infer + process with context tracker
        response = await self.llm_proxy_with_tracker(
            messages=req_as_dict["messages"],
            tools=req_as_dict["tools"],
            tool_choice="auto",
        )

        # this is an important id assignment
        response.id = timeline_uuid
        assert isinstance(response, ChatCompletion)
        return response


    @property
    def should_terminate(self) -> bool:
        return self._should_terminate

    def begin_service(self):
        """
        Starts the SSE service loop.
        """
        t = threading.Thread(target=lambda: asyncio.run(self._ensure_service_loop()), daemon=True)
        t.start()

    async def _ensure_service_loop(self):
        while not self.should_terminate:
            try:
                await self._service_loop()
            except Exception as e:
                logger.warning(f"InterchangeClient service loop error: {e}. Restarting...")
                await asyncio.sleep(4)  # brief pause before reconnecting

    async def _service_loop(self):
        """
        In fact this is not a service,
        it is a client that pretends to be a service, by interacting with a local interchange server via SSE.

        This design is for efficiency
        """

        from ajet.tuner_lib.weight_tuner.experimental.as_oai_model_server import InterchangeCompletionRequest

        port = os.getenv("AJET_DAT_INTERCHANGE_PORT")
        assert port is not None, "AJET_DAT_INTERCHANGE_PORT env var must be set"

        base_url = f"http://127.0.0.1:{port}"
        listen_url = f"{base_url}/hook/context_tracker_client_listen"
        response_url = f"{base_url}/hook/context_tracker_client_response"
        key = f"episode_uuid:{self.episode_uuid}"

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("GET", listen_url, params={"episode_uuid": self.episode_uuid}, timeout=None) as response:
                    async for line in response.aiter_lines():
                        if self.should_terminate:
                            break

                        if not line.strip():
                            continue

                        if line.startswith(":"): # keepalive
                            continue

                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if not data:
                                continue

                            try:
                                try:
                                    parsed_msg = InterchangeCompletionRequest(**json.loads(data))
                                except Exception as e:
                                    logger.error(f"Failed to parse SSE event data: {e}" + data)
                                    continue

                                result = await self.llm_infer(
                                    req=parsed_msg.completion_request,
                                    timeline_uuid=parsed_msg.timeline_uuid,
                                    agent_name=parsed_msg.agent_name,
                                    target_tag=parsed_msg.target_tag,
                                    episode_uuid=parsed_msg.episode_uuid,
                                )

                                # Send response back
                                await client.post(
                                    response_url,
                                    params={"key": key},
                                    content=pickle.dumps(result),
                                    headers={"Content-Type": "application/octet-stream"}
                                )

                            except Exception as e:
                                logger.error(f"Error processing SSE event: {e}")
                                continue

            except httpx.RequestError as e:
                 logger.warning(f"SSE connection error: {e}")
                 raise # Let ensure_service_loop handle restart

            # Send terminate signal if we are exiting cleanly
            try:
                await client.post(
                    response_url,
                    params={"key": key},
                    content=pickle.dumps("terminate"),
                    headers={"Content-Type": "application/octet-stream"}
                )
            except:
                pass



