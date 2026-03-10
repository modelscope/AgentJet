# -*- coding: utf-8 -*-

"""
A one-to-many proxy server for LLM requests with reinforcement learning.

This server implements a one-to-many request pattern where each user request
is processed by multiple parallel episodes, and the best response is selected
based on computed rewards.

Architecture Overview:
---------------------
1. Server Initialization:
   - Connects to swarm server and syncs training config
   - Starts the engine with specified AgentJetJob configuration

2. Request Processing Flow:
   - Receives LLM request and creates a Task
   - Runs NUM_REPEAT parallel episodes
   - Computes rewards for each episode response
   - Returns the best response to user

Usage:
    python -m ajet.tuner_lib.experimental.oai_model_one2many


"""

import os
import uuid
import random
import asyncio
import httpx
import json
import threading
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from ajet.schema.task import Task, WorkflowOutput
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from beast_logger import print_listofdict


# =============================================================================
# Configuration Constants
# =============================================================================

SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
NUM_REPEAT = int(os.getenv("NUM_REPEAT", "8"))
USER_PREFERENCE = "我希望我的助手足够幽默"

# =============================================================================
# Global State
# =============================================================================
USER_REQUEST_RECORD: List[Dict] = []
REQUEST_COUNTER = 0
swarm_client: Optional[SwarmClient] = None
ajet_job = AgentJetJob(
    algorithm="grpo",
    project_name="ajet-swarm",
    experiment_name="test",
    n_gpu=8,
    model='/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct',
    batch_size=16,
    num_repeat=NUM_REPEAT,
)

# =============================================================================
# Pydantic Models
# =============================================================================

class EpisodeResult(BaseModel):
    """Result from a single episode execution."""
    episode_uuid: str
    response: Dict | List[bytes]


# =============================================================================
# User Request Record Management
# =============================================================================

async def on_user_submit_new_requests(request_id: str, task: Task) -> None:
    """
    Store user request record when a new request is submitted.

    This function maintains a chronological record of all user requests,
    which can be used for tracking and debugging purposes.

    Args:
        request_id: Unique identifier for this request
        task: The Task object containing query information
    """
    USER_REQUEST_RECORD.append({
        "request_id": request_id,
        "task_id": task.task_id,
        "query": task.main_query,
    })
    # here, add some code to update OpenJudge grader according to new user preference (if user indicate any)


# =============================================================================
# Reward Computation
# =============================================================================

async def on_compute_relative_reward(valid_results: List[EpisodeResult], all_answers: List[Dict]) -> List[float]:
    """
    Compute relative rewards for all episode results.

    This function calculates a reward score for each episode response.
    Currently implements a random reward generator as a placeholder.

    Future implementations should compare responses and generate
    meaningful scores based on quality metrics.

    Args:
        valid_results: List of successful episode results

    Returns:
        List of reward scores in range [-1.0, 1.0]
    """

    # here, use OpenJudge to compute relative scores
    rewards = [random.uniform(-1.0, 1.0) for _ in valid_results]

    # Add reward to each answer for logging
    for answer, reward in zip(all_answers, rewards):
        answer["reward"] = reward
    print_listofdict(all_answers, header="on_compute_relative_reward")

    return rewards


# =============================================================================
# Response Processing Utilities
# =============================================================================

def extract_assistant_message(resp: Dict | List[bytes]) -> Dict:
    """
    Extract assistant message from response (handles both stream and non-stream).

    For streaming responses, accumulates delta content from all chunks.
    For non-streaming responses, extracts the message directly.

    Args:
        resp: Response data (list of chunks for stream, dict for non-stream)

    Returns:
        Dictionary containing the assistant's message with role, content,
        and optionally tool_calls
    """
    if isinstance(resp, list):
        # Stream response: accumulate delta content from all chunks
        content_parts: List[str] = []
        tool_calls_map: Dict[int, Dict] = {}

        for raw in resp:
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue

            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break

            try:
                chunk = json.loads(payload)
                delta = chunk.get("choices", [{}])[0].get("delta", {})

                # Accumulate content
                if delta.get("content"):
                    content_parts.append(delta["content"])

                # Accumulate tool calls
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = tc
                    else:
                        existing_args = tool_calls_map[idx].get("function", {}).get("arguments", "")
                        new_args = tc.get("function", {}).get("arguments", "")
                        tool_calls_map[idx].setdefault("function", {})["arguments"] = existing_args + new_args
            except Exception:
                pass

        msg: Dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls_map:
            msg["tool_calls"] = list(tool_calls_map.values())
        return msg
    else:
        # Non-stream: standard OpenAI response dict
        return resp.get("choices", [{}])[0].get("message", {})


# =============================================================================
# HTTP Proxy Functions
# =============================================================================

async def proxy_chat_completion(
    base_url: str,
    api_key: str,
    request: Request,
    is_stream: bool = False
) -> Dict | List[bytes]:
    """
    Proxy a chat completion request to the specified base URL.

    Args:
        base_url: Target server base URL
        api_key: API key for authentication
        request: Original FastAPI request object
        is_stream: Whether to use streaming response

    Returns:
        Response data (dict for non-stream, list of chunks for stream)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Connection": "close",
    }

    json_data = await request.json()
    json_data["stream"] = is_stream

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json=json_data,
            headers=headers,
        )
        resp.raise_for_status()

        if is_stream:
            chunks = []
            async for line in resp.aiter_lines():
                if line.strip():
                    chunks.append(line.encode() if isinstance(line, str) else line)
            return chunks
        else:
            return resp.json()


# =============================================================================
# Episode Execution
# =============================================================================

async def run_single_episode(
    episode_index: int,
    request: Request,
    is_stream: bool,
) -> EpisodeResult:
    """
    Run a single episode with the swarm client.

    Args:
        episode_index: Index of this episode (for logging)
        request: Original FastAPI request object
        is_stream: Whether to use streaming response

    Returns:
        EpisodeResult containing the episode UUID and response data

    Raises:
        Exception: If the episode fails (after aborting the episode)
    """
    assert swarm_client is not None, "Swarm client not initialized"

    loop = asyncio.get_event_loop()
    episode_uuid, api_baseurl_key = await loop.run_in_executor(
        None, lambda: swarm_client.begin_episode(discard_episode_timeout=120)  # type: ignore[union-attr]
    )

    try:
        response_data = await proxy_chat_completion(
            base_url=api_baseurl_key.base_url,
            api_key=api_baseurl_key.api_key,
            request=request,
            is_stream=is_stream,
        )
        return EpisodeResult(episode_uuid=episode_uuid, response=response_data)
    except Exception as e:
        logger.error(f"Error in episode {episode_index}: {e}")
        swarm_client.abort_episode(episode_uuid)
        raise


async def run_all_episodes(request: Request, is_stream: bool) -> List[EpisodeResult]:
    """
    Run all episodes in parallel and collect valid results.

    Args:
        request: Original FastAPI request object
        is_stream: Whether to use streaming response

    Returns:
        List of successful episode results

    Raises:
        HTTPException: If all episodes fail
    """
    episode_tasks = [
        run_single_episode(i, request, is_stream)
        for i in range(NUM_REPEAT)
    ]

    results = await asyncio.gather(*episode_tasks, return_exceptions=True)

    valid_results: List[EpisodeResult] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Episode failed: {result}")
        elif isinstance(result, EpisodeResult):
            valid_results.append(result)

    if not valid_results:
        raise HTTPException(status_code=500, detail="All episodes failed")

    return valid_results


async def finalize_episodes(
    task: Task,
    valid_results: List[EpisodeResult],
    rewards: List[float]
) -> None:
    """
    Finalize all episodes by sending rewards to the swarm client.

    Args:
        task: The Task object for this request
        valid_results: List of successful episode results
        rewards: List of computed rewards for each result
    """
    assert swarm_client is not None, "Swarm client not initialized"

    loop = asyncio.get_event_loop()

    for episode_result, reward in zip(valid_results, rewards):
        workflow_output = WorkflowOutput(reward=reward, metadata={})
        await loop.run_in_executor(
            None,
            lambda ep=episode_result, wo=workflow_output: swarm_client.end_episode(  # type: ignore[union-attr]
                task, ep.episode_uuid, wo
            ),
        )


# =============================================================================
# Main Request Handler
# =============================================================================

async def handle_one2many_request(request: Request, request_id: str) -> Dict | List[bytes]:
    """
    Handle a one-to-many request by running multiple episodes in parallel.

    This is the main entry point for processing chat completion requests.
    It orchestrates the entire flow:
    1. Parse request and create task
    2. Store request record
    3. Run parallel episodes
    4. Compute rewards
    5. Select and return best response

    Args:
        request: FastAPI request object
        request_id: Unique identifier for this request

    Returns:
        Best response data (dict or list of stream chunks)
    """
    # Parse request
    json_data = await request.json()
    is_stream = json_data.get('stream', False)
    messages = json_data.get('messages', [])
    message_latest = messages[-1]
    user_query = str(message_latest.get("content", "") if isinstance(message_latest, dict) else "")

    # Create task and store request record
    task = Task(
        task_id=str(uuid.uuid4()),
        main_query=user_query,
        metadata={"user_preference": USER_PREFERENCE}
    )
    await on_user_submit_new_requests(request_id, task)

    # Run all episodes in parallel
    valid_results = await run_all_episodes(request, is_stream)

    # Extract answers and compute rewards
    all_answers = [extract_assistant_message(r.response) for r in valid_results]
    rewards = await on_compute_relative_reward(valid_results, all_answers)


    # Finalize episodes with rewards
    await finalize_episodes(task, valid_results, rewards)

    # Select and return best response
    best_idx = rewards.index(max(rewards))
    return valid_results[best_idx].response


# =============================================================================
# FastAPI Application Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global swarm_client
    global ajet_job

    logger.info(f"Initializing swarm client with URL: {SWARM_URL}")
    swarm_client = SwarmClient(SWARM_URL)


    logger.info(f"Syncing train config and starting engine with num_repeat={NUM_REPEAT}")

    def start_engine_background():
        try:
            swarm_client.auto_sync_train_config_and_start_engine(  # type: ignore[union-attr]
                ajet_job,
                force_restart=False,
            )
            logger.info("Swarm engine is ready!")
        except Exception as e:
            logger.warning(f"Engine auto-sync skipped or failed: {e}")

    engine_thread = threading.Thread(target=start_engine_background, daemon=True)
    engine_thread.start()

    yield


app = FastAPI(title="One-to-Many Proxy Server", lifespan=lifespan)


# =============================================================================
# API Endpoints
# =============================================================================

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def one2many_proxy(request: Request, path: str):
    """Main proxy endpoint for OpenAI-compatible API requests."""
    global REQUEST_COUNTER

    try:
        if request.method == "POST" and path == "chat/completions":
            REQUEST_COUNTER += 1
            request_id = f"req_{REQUEST_COUNTER}_{uuid.uuid4().hex[:8]}"
            logger.info(f"Received chat completion request {request_id}")

            response_data = await handle_one2many_request(request, request_id)

            if isinstance(response_data, list):
                # Stream response: replay recorded chunks
                async def stream_chunks(chunks: List[bytes]):
                    for chunk in chunks:
                        yield chunk + b"\n\n"

                return StreamingResponse(
                    stream_chunks(response_data),
                    media_type="text/event-stream",
                )

            return response_data
        else:
            raise HTTPException(status_code=404, detail="Not Found")

    except httpx.TimeoutException:
        logger.error(f"Timeout proxying {request.method} {path}")
        raise HTTPException(status_code=504, detail="Gateway Timeout")

    except httpx.ConnectError:
        logger.error(f"Connection error proxying {request.method} {path}")
        raise HTTPException(status_code=502, detail="Bad Gateway")

    except Exception as e:
        logger.exception(f"Unexpected error proxying {request.method} {path}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "user_preference": USER_PREFERENCE}


@app.get("/requests")
async def get_requests():
    """Get all recorded user requests."""
    return {"requests": USER_REQUEST_RECORD}


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)


# =============================================================================
# Test Script (for reference)
# =============================================================================

''' Test Script:

# -*- coding: utf-8 -*-

import os
import time
import requests
from typing import List, Dict

PROXY_URL = os.getenv("PROXY_URL", "http://localhost:10010")

MESSAGES = [
    [{"role": "user", "content": "Hello, how are you?"}],
    [{"role": "user", "content": "Tell me a joke."}],
    [{"role": "user", "content": "What's the weather like today?"}],
    [{"role": "user", "content": "Write a short poem about coding."}],
    [{"role": "user", "content": "What is Python?"}],
    [{"role": "user", "content": "How do I learn machine learning?"}],
    [{"role": "user", "content": "Tell me about your hobbies."}],
    [{"role": "user", "content": "What's your favorite programming language?"}],
    [{"role": "user", "content": "Explain what is an API."}],
    [{"role": "user", "content": "Give me a recipe for pasta."}],
]


def send_chat_request(messages: List[Dict[str, str]], stream: bool = False) -> Dict:
    """Send a chat completion request to the proxy server."""
    payload = {
        "model": "test-model",
        "messages": messages,
        "stream": stream,
    }

    try:
        response = requests.post(
            f"{PROXY_URL}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def main():
    print(f"Starting client, sending requests to {PROXY_URL}")
    print("Press Ctrl+C to stop\n")

    request_count = 0

    while True:
        request_count += 1
        messages = MESSAGES[request_count % len(MESSAGES)]

        print(f"[Request {request_count}] Sending: {messages[0]['content'][:50]}...")

        result = send_chat_request(messages)

        if "error" in result:
            print(f"[Request {request_count}] Error: {result['error']}")
        else:
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"[Request {request_count}] Response: {content[:100]}...")

        print()

        time.sleep(5)


if __name__ == "__main__":
    try:
        health = requests.get(f"{PROXY_URL}/health", timeout=5)
        print(f"Server health: {health.json()}\n")
    except Exception as e:
        print(f"Warning: Could not connect to server: {e}\n")

    main()

'''
