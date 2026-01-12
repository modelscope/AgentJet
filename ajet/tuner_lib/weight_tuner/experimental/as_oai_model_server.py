"""
A shadow FastAPI server for serving as interchange endpoint between Tuner and Workflow.

- This functionality is experimental.
- The code is very async, considering extreme efficiency for handling many concurrent requests,
  therefore, it may be hard to read.

---------------------------------------------------------------------------------------------

"""

import asyncio
import threading
import uuid
import time
from collections import defaultdict
from typing import Dict, List
import base64
import json
import os
import pickle
from pprint import pformat
from loguru import logger

from pydantic import BaseModel, ConfigDict, model_validator
from fastapi import FastAPI, Header, HTTPException, Request, Body
from fastapi.responses import StreamingResponse
import uvicorn
import sys
import subprocess
import atexit
import argparse

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from openai.types.chat.chat_completion import ChatCompletion


# Global variables for tracking requests and responses
# class ChatCompletionRequestDeferBuildOff(ChatCompletionRequest):
#     model_config = ConfigDict(
#         defer_build=False,
#         validate_default=True,
#         validate_assignment=True,
#     )
# ChatCompletionRequest.model_validate_json(x)

class InterchangeCompletionRequest(BaseModel):
    completion_request: ChatCompletionRequest
    agent_name: str
    target_tag: str
    episode_uuid: str
    timeline_uuid: str

ajet_remote_handler_received: Dict[str, Dict[str, InterchangeCompletionRequest]] = defaultdict(dict)
ajet_remote_handler_in_progress: Dict[str, Dict[str, InterchangeCompletionRequest]] = defaultdict(dict)
ajet_remote_handler_completed: Dict[str, Dict[str, ChatCompletion]] = defaultdict(dict)
ajet_remote_handler_discarded: Dict[str, Dict[str, bool]] = defaultdict(dict)
active_websockets: Dict[str, asyncio.Event] = {}

# Create FastAPI app
app = FastAPI(title="AJet Interchange Endpoint")

POLL_INTERVAL_SECONDS = 0.5

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


async def sse_event_generator(key: str, stop_event: asyncio.Event, request: Request):
    """
    Generator for Server-Sent Events.
    Yields requests to the client.
    """
    try:
        wait_cnt = 0
        while not stop_event.is_set():

            if await request.is_disconnected():
                logger.info(f"Client disconnected for key: {key}")
                stop_event.set()
                break

            # Check for new requests in ajet_remote_handler_received
            if (key in ajet_remote_handler_received) and len(ajet_remote_handler_received[key]) > 0:
                timeline_uuid = list(ajet_remote_handler_received[key].keys())[0]

                # Get the next request
                new_req: InterchangeCompletionRequest = ajet_remote_handler_received[key].pop(timeline_uuid)

                assert timeline_uuid == new_req.timeline_uuid

                # Move to in_progress
                ajet_remote_handler_in_progress[key][timeline_uuid] = new_req

                # Send via SSE
                # We send the JSON representation of the request
                # Client expects: parsed_msg:InterchangeCompletionRequest

                # We simply yield the json string.
                # The client side will need to read this json.
                json_data = new_req.model_dump_json()
                yield f"data: {json_data}\n\n"
            else:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                wait_cnt += 1
                if wait_cnt * POLL_INTERVAL_SECONDS >= 5:
                    wait_cnt = 0
                    # Send keepalive comment to prevent timeouts
                    yield ": keepalive\n\n"

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    except Exception as e:
        logger.exception(f"Error in SSE generator: {e}")

    finally:
        stop_event.set()
        # Cleanup
        if key in active_websockets:
            active_websockets.pop(key)

        # Clean up any in-progress requests for this key could be here, or explicitly on reset/timeout
        # Mirroring original finally block logic:
        # Note: In SSE, we might not want to aggressively clear everything on disconnect if retries are expected,
        # but the original code cleaned up on websocket disconnect.
        if key:
             for container in [
                ajet_remote_handler_received,
                ajet_remote_handler_in_progress,
                ajet_remote_handler_completed,
            ]:
                if key in container:
                    container.pop(key)

             if key in ajet_remote_handler_discarded:
                ajet_remote_handler_discarded.pop(key)


@app.post("/hook/context_tracker_client_response")
async def context_tracker_client_response(key: str, response_data: bytes = Body(...)):
    """
    Endpoint to receive processing results from the client.
    """
    try:
        # Decode response
        # The client sends pickled ChatCompletion object or "terminate" string
        try:
            openai_response = pickle.loads(response_data)
        except Exception as e:
            logger.error(f"Pickle load failed: {e}")
            # Try assuming it might not be pickled if we change client, but let's stick to pickle for complex objects
            raise HTTPException(status_code=400, detail="Invalid response format")

        if openai_response == "terminate":
             # Handle termination signal if needed, though usually handled by disconnect
             if key in active_websockets:
                 active_websockets[key].set()
             return {"status": "terminated"}

        if not isinstance(openai_response, ChatCompletion):
             logger.error(f"Invalid response object type: {type(openai_response)}")
             raise HTTPException(status_code=400, detail="Invalid response object")

        timeline_uuid = openai_response.id

        if key in ajet_remote_handler_in_progress and timeline_uuid in ajet_remote_handler_in_progress[key]:
            ajet_remote_handler_in_progress[key].pop(timeline_uuid)

        # Add to completed if not discarded
        if (key not in ajet_remote_handler_discarded) or (timeline_uuid not in ajet_remote_handler_discarded[key]):
            ajet_remote_handler_completed[key][timeline_uuid] = openai_response

        return {"status": "accepted"}

    except Exception as e:
        logger.exception(f"Error in response handler: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hook/context_tracker_client_listen")
async def context_tracker_client_listen(request: Request, episode_uuid: str):
    """
    SSE endpoint for clients to listen for completion requests.
    """
    key = f"episode_uuid:{episode_uuid}"

    stop_event = asyncio.Event()
    active_websockets[key] = stop_event # Storing stop_event instead of websocket

    return StreamingResponse(
        sse_event_generator(key, stop_event, request),
        media_type="text/event-stream"
    )

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
    new_req = ChatCompletionRequest.model_validate(body)
    if new_req.stream:
        raise HTTPException(status_code=400, detail="Streaming responses not supported in current AgentJet version, please set `stream=false` for now.")
    # Create timeline UUID
    timeline_uuid = uuid.uuid4().hex

    # Add to received queue
    # logger.warning(f"Received new chat completion request for agent: {agent_name}, target_tag: {target_tag}, episode_uuid: {episode_uuid}, timeline_uuid: {timeline_uuid}")
    int_req = InterchangeCompletionRequest(
        completion_request = new_req,
        agent_name = agent_name,
        target_tag = target_tag,
        episode_uuid = episode_uuid,
        timeline_uuid = timeline_uuid,
    )

    # # fix Pydantic validation issue for tool_calls field
    # for msg in int_req.completion_request.messages:
    #     if isinstance(msg, dict) and 'tool_calls' in msg:
    #         tc = msg['tool_calls']
    #         if not isinstance(tc, list):
    #             msg['tool_calls'] = list(tc) if tc else []

    ajet_remote_handler_received[key][timeline_uuid] = int_req

    # Wait for response (with periodic checks for client disconnect)
    max_wait_time = 600  # 10 minutes timeout
    elapsed_time = 0

    try:
        while elapsed_time < max_wait_time:
            # Check if response is available
            if (key in ajet_remote_handler_completed) \
                and timeline_uuid in ajet_remote_handler_completed[key]:
                openai_response = ajet_remote_handler_completed[key][timeline_uuid]
                return openai_response
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            elapsed_time += POLL_INTERVAL_SECONDS

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
    # logger.warning("Resetting interchange endpoint server state.")
    # Disconnect all websockets
    for key, stop_event in list(active_websockets.items()):
        try:
            stop_event.set()
        except:
            pass

    active_websockets.clear()

    # Clear all global state
    ajet_remote_handler_received.clear()
    ajet_remote_handler_in_progress.clear()
    ajet_remote_handler_completed.clear()
    ajet_remote_handler_discarded.clear()

    return {"status": "reset_complete"}


async def monitor_debug_state(experiment_dir):
    """
    Background task to write debug state to ./interchange_debug.txt every 1 second.
    """
    while True:
        try:
            debug_info = {
                'ajet_remote_handler_received': dict(ajet_remote_handler_received),
                'ajet_remote_handler_in_progress': dict(ajet_remote_handler_in_progress),
                'ajet_remote_handler_completed': dict(ajet_remote_handler_completed),
                'ajet_remote_handler_discarded': dict(ajet_remote_handler_discarded),
                'active_websockets': list(active_websockets.keys())
            }

            with open(f'{experiment_dir}/interchange_debug.txt', 'w') as f:
                f.write(pformat(debug_info, width=120, indent=2))
                f.write('\n')

            await asyncio.sleep(4)
        except Exception as e:
            logger.error(f"Error in monitor_debug_state: {e}")
            await asyncio.sleep(4)


def ensure_dat_interchange_server_cache_clear():
    """
    send http request to clear the interchange server state.
    """
    import httpx

    port = os.getenv("AJET_DAT_INTERCHANGE_PORT")
    assert port is not None, "AJET_DAT_INTERCHANGE_PORT env var must be set"
    url = f"http://localhost:{port}/reset"
    try:
        response = httpx.post(url)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        pass
    except httpx.HTTPStatusError as e:
        pass

    return


class InterchangeEndpointServer:
    """
    Class to manage the FastAPI interchange endpoint server.
    """

    def __init__(self):
        self.server_thread = None
        self.server = None

    def start(self, experiment_dir) -> int:
        """
        Start the FastAPI server on a free port.

        Returns:
            int: The port number the server is running on.
        """
        # Find a free port
        self.port = int(os.environ.get("AJET_DAT_INTERCHANGE_PORT", -1))

        # Create server thread
        def run_server():
            async def serve_with_monitor():
                # Start the monitor task
                asyncio.create_task(monitor_debug_state(experiment_dir))

                # Start the server
                config = uvicorn.Config(
                    app=app,
                    host="0.0.0.0",
                    port=self.port,
                    log_level="error",
                )
                server = uvicorn.Server(config)
                await server.serve()

            asyncio.run(serve_with_monitor())

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
def start_interchange_server(experiment_dir) -> int:
    """
    Start the interchange endpoint server and return the port number.
    This launches a subprocess to run the server.

    Returns:
        int: The port number the server is running on.
    """
    # Find a free port if not specified or invalid
    port = int(os.environ.get("AJET_DAT_INTERCHANGE_PORT", -1))
    if port <= 0:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        os.environ["AJET_DAT_INTERCHANGE_PORT"] = str(port)

    # Launch as subprocess
    env = os.environ.copy()

    # We run this file as a script
    cmd = [sys.executable, os.path.abspath(__file__), "--experiment_dir", experiment_dir, "--port", str(port)]

    process = subprocess.Popen(
        cmd,
        env=env,
        # redirect stdout/stderr if needed, but keeping them might be useful for debug
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL
    )

    def cleanup():
        if process.poll() is None:
            logger.info("Terminating interchange server subprocess")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    # Wait for server to be ready
    import httpx
    health_url = f"http://localhost:{port}/health"
    start_time = time.time()
    while time.time() - start_time < 20:
        if process.poll() is not None:
            logger.error(f"Interchange server subprocess failed to start. Return code: {process.returncode}")
            break

        try:
            if httpx.get(health_url, timeout=0.5).status_code == 200:
                break
        except Exception:
            pass

        time.sleep(0.5)

    atexit.register(cleanup)

    logger.info(f"Interchange server subprocess started on port {port} (pid: {process.pid})")
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AJet Interchange Endpoint Server")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Directory to store debug info")
    parser.add_argument("--port", type=int, required=True, help="Port to run the server on")

    args = parser.parse_args()

    async def serve_with_monitor():
        # Start the monitor task
        asyncio.create_task(monitor_debug_state(args.experiment_dir))

        # Start the server
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=args.port,
            log_level="error",
        )
        server = uvicorn.Server(config)
        await server.serve()

    try:
        asyncio.run(serve_with_monitor())
    except KeyboardInterrupt:
        pass

