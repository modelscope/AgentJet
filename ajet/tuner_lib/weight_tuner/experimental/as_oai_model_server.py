"""
A shadow FastAPI server for serving as interchange endpoint between agents and model handlers.

- This functionality is experimental.
"""

import asyncio
import threading
import uuid
from collections import defaultdict
from typing import Dict, List
import base64
import json
import os
import pickle

from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException, Request
import uvicorn

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from openai.types.chat.chat_completion import ChatCompletion


# Global variables for tracking requests and responses
class TypeCompletionRequest(BaseModel):
    completion_request: ChatCompletionRequest
    agent_name: str
    target_tag: str
    episode_uuid: str
    timeline_uuid: str = ""  # to be filled when sending to client


ajet_remote_handler_received: Dict[str, Dict[str, TypeCompletionRequest]] = defaultdict(dict)
ajet_remote_handler_in_progress: Dict[str, Dict[str, TypeCompletionRequest]] = defaultdict(dict)
ajet_remote_handler_completed: Dict[str, Dict[str, ChatCompletion]] = defaultdict(dict)
ajet_remote_handler_discarded: Dict[str, Dict[str, bool]] = defaultdict(dict)
active_websockets = {}

# Create FastAPI app
app = FastAPI(title="AJet Interchange Endpoint")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}



async def coro_task_1_lookup_dict_received__send_loop(key, websocket: WebSocket, stop_event: asyncio.Event):
    # Monitor for new requests
    try:
        while not stop_event.is_set():
            # Check for new requests in ajet_remote_handler_received
            if (key in ajet_remote_handler_received) and len(ajet_remote_handler_received[key]) > 0:

                timeline_uuid = list(ajet_remote_handler_received[key].keys())[0]

                # Get the next request
                new_req: TypeCompletionRequest = ajet_remote_handler_received[key].pop(timeline_uuid)

                assert timeline_uuid == new_req.timeline_uuid

                # Move to in_progress
                ajet_remote_handler_in_progress[key][timeline_uuid] = new_req

                # will be received by:
                #   ajet/tuner_lib/weight_tuner/experimental/as_oai_model_client.py
                #       await asyncio.wait_for(websocket.recv(decode=False), timeout=0.25)
                await websocket.send_bytes(pickle.dumps(new_req))

    except WebSocketDisconnect:
        stop_event.set()
        return

    except Exception as e:
        stop_event.set()
        print(f"Error in websocket handler: {e}")
        return


async def coro_task_2_lookup_dict_received__receive_loop(key, websocket: WebSocket, stop_event: asyncio.Event):
    try:
        while not stop_event.is_set():
            # Wait for client response:
            #   ajet/tuner_lib/weight_tuner/experimental/as_oai_model_client.py
            #       await websocket.send(pickle.dumps(response))
            response_data = pickle.loads(await websocket.receive_bytes())

            if not isinstance(response_data, ChatCompletion):
                stop_event.set()
                assert response_data == "terminate", "Invalid terminate signal from client"
                await websocket.close()
                return

            # Process the response
            openai_response: ChatCompletion = response_data

            # see `ajet/tuner_lib/weight_tuner/experimental/as_oai_model_client.py::response.id = timeline_uuid`
            timeline_uuid = openai_response.id

            # Remove from in_progress
            if timeline_uuid in ajet_remote_handler_in_progress[key]:
                ajet_remote_handler_in_progress[key].pop(timeline_uuid)

            # Add to completed if not discarded
            if (key not in ajet_remote_handler_discarded) or (timeline_uuid not in ajet_remote_handler_discarded[key]):
                # openai_response should already be a ChatCompletion object if client sent pickle
                ajet_remote_handler_completed[key][timeline_uuid] = openai_response

    except WebSocketDisconnect:
        stop_event.set()
        return

    except Exception as e:
        stop_event.set()
        print(f"Error in websocket handler: {e}")
        return



@app.websocket("/hook/context_tracker_client_listen")
async def context_tracker_client_listen(websocket: WebSocket):
    """
    WebSocket endpoint for clients to listen for completion requests.
    Clients send (agent_name, target_tag, episode_uuid) and receive requests to process.
    """
    await websocket.accept()

    key = ""
    try:
        # Receive initial connection data (
        #   ajet/tuner_lib/weight_tuner/experimental/as_oai_model_client.py
        #       await websocket.send(f"episode_uuid:{self.episode_uuid}"))
        episode_uuid_str = pickle.loads(await websocket.receive_bytes())
        assert episode_uuid_str.startswith("episode_uuid:")
        episode_uuid = episode_uuid_str.split("episode_uuid:")[-1]

        if not all([episode_uuid]):
            await websocket.send_json({"error": "Missing required fields"})
            await websocket.close()
            return

        key = f"episode_uuid:{episode_uuid}"
        active_websockets[key] = websocket

        stop_event = asyncio.Event()
        asyncio.create_task(coro_task_1_lookup_dict_received__send_loop(key, websocket, stop_event))
        asyncio.create_task(coro_task_2_lookup_dict_received__receive_loop(key, websocket, stop_event))

    finally:

        if key:
            # Clean up any in-progress requests for this key
            for container in [
                ajet_remote_handler_received,
                ajet_remote_handler_in_progress,
                ajet_remote_handler_completed,
            ]:
                if key in container:
                    container.pop(key)

            if key in ajet_remote_handler_discarded:
                ajet_remote_handler_discarded.pop(key)

            if key in active_websockets:
                websocket = active_websockets.pop(key)
                try:
                    websocket.close()
                except:
                    pass


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str = Header(None)):
    """
    OpenAI-compatible chat completions endpoint.
    Receives ChatCompletionRequest and returns ChatCompletion.
    """
    # Parse authorization header (base64 encoded JSON)
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    try:
        # Remove "Bearer " prefix if present
        auth_token = authorization.replace("Bearer ", "").replace("bearer ", "")
        decoded = base64.b64decode(auth_token).decode('utf-8')
        auth_data = json.loads(decoded)

        agent_name = auth_data.get("agent_name")
        target_tag = auth_data.get("target_tag")
        episode_uuid = auth_data.get("episode_uuid")

        if not all([agent_name, target_tag, episode_uuid]):
            raise HTTPException(status_code=401, detail="Invalid authorization data")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid authorization header: {str(e)}")

    # Build key
    key = f"episode_uuid:{episode_uuid}"

    # Parse request body
    body = await request.json()
    new_req = ChatCompletionRequest(**body)
    # Create timeline UUID
    timeline_uuid = uuid.uuid4().hex
    # Add to received queue
    ajet_remote_handler_received[key][timeline_uuid] = TypeCompletionRequest(
        completion_request = new_req,
        agent_name = agent_name,
        target_tag = target_tag,
        episode_uuid = episode_uuid,
        timeline_uuid = timeline_uuid,
    )

    # Wait for response (with periodic checks for client disconnect)
    max_wait_time = 1800  # 30 minutes timeout
    check_interval = 0.25  # Check every 250ms
    elapsed_time = 0

    try:
        while elapsed_time < max_wait_time:
            # Check if response is available
            if (key in ajet_remote_handler_completed) \
                and timeline_uuid in ajet_remote_handler_completed[key]:
                openai_response = ajet_remote_handler_completed[key][timeline_uuid]
                return openai_response
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval

        # Timeout reached
        raise HTTPException(status_code=504, detail="Request timeout")
    except asyncio.CancelledError:
        # Client disconnected
        if key not in ajet_remote_handler_discarded:
            ajet_remote_handler_discarded[key] = {}
        ajet_remote_handler_discarded[key][timeline_uuid] = True
        raise HTTPException(status_code=499, detail="Client disconnected")


@app.post("/reset")
async def reset():
    """
    Reset endpoint to clear all state and disconnect all websockets.
    """
    # Disconnect all websockets
    for key, ws in list(active_websockets.items()):
        try:
            await ws.close()
        except:
            pass

    active_websockets.clear()

    # Clear all global state
    ajet_remote_handler_received.clear()
    ajet_remote_handler_in_progress.clear()
    ajet_remote_handler_completed.clear()
    ajet_remote_handler_discarded.clear()

    return {"status": "reset_complete"}


class InterchangeEndpointServer:
    """
    Class to manage the FastAPI interchange endpoint server.
    """

    def __init__(self):
        self.server_thread = None
        self.server = None

    def start(self) -> int:
        """
        Start the FastAPI server on a free port.

        Returns:
            int: The port number the server is running on.
        """
        # Find a free port
        self.port = int(os.environ.get("AJET_DAT_INTERCHANGE_PORT", -1))

        # Create server thread
        def run_server():
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=self.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            asyncio.run(server.serve())

        # Start server in a new thread
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        return self.port

    def stop(self):
        """
        Stop the server (note: due to uvicorn limitations, this may not fully stop the thread).
        """
        # This is a simple implementation - in production you'd want more robust shutdown
        pass


# Convenience function for quick server startup
def start_interchange_server() -> int:
    """
    Start the interchange endpoint server and return the port number.

    Returns:
        int: The port number the server is running on.
    """
    server = InterchangeEndpointServer()
    port = server.start()
    return port

