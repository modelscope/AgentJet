
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
import websockets
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
        Starts the websocket service loop.
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
        it is a client that pretends to be a service, by interacting with a local websocket interchange server.

        This design is for efficiency
        """

        from ajet.tuner_lib.weight_tuner.experimental.as_oai_model_server import InterchangeCompletionRequest

        port = os.getenv("AJET_DAT_INTERCHANGE_PORT")
        assert port is not None, "AJET_DAT_INTERCHANGE_PORT env var must be set"
        uri = f"ws://127.0.0.1:{port}/hook/context_tracker_client_listen"

        async with websockets.connect(uri, ping_timeout=3600, open_timeout=16) as websocket:
            try:
                # Send initialization parameters
                # Sending as a list [agent_name, target_tag, episode_uuid] to match "input (a,b,c)" structure
                await websocket.send(pickle.dumps(f"episode_uuid:{self.episode_uuid}"))

                while not self.should_terminate:

                    try:
                        # wait message from ajet/tuner_lib/weight_tuner/experimental/as_oai_model_server.py
                        parsed_msg_str: str = pickle.loads(
                            await asyncio.wait_for(websocket.recv(decode=False), timeout=0.25)
                        )
                        parsed_msg:InterchangeCompletionRequest = InterchangeCompletionRequest(**json.loads(parsed_msg_str))

                        response = await self.llm_infer(
                            req=parsed_msg.completion_request,
                            timeline_uuid=parsed_msg.timeline_uuid,
                            agent_name=parsed_msg.agent_name,
                            target_tag=parsed_msg.target_tag,
                            episode_uuid=parsed_msg.episode_uuid,
                        )
                        await websocket.send(pickle.dumps(response))

                    except asyncio.TimeoutError:
                        # 0.25s timeout, loop back to check should_terminate
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Websocket connection closed by server")
                        return # Exit inner loop to reconnect or finish

                await websocket.send(pickle.dumps("terminate"))

            except (OSError, IOError) as e:
                logger.warning(f"Websocket connection error: {e}")
                pass



