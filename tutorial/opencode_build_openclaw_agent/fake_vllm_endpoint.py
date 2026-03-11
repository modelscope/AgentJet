# -*- coding: utf-8 -*-
"""
Fake vLLM endpoint for OpenClaw agent training.
Based on ajet/tuner_lib/experimental/oai_model_one2many.py
"""

import os
import uuid
import asyncio
import httpx
import json
import threading
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from ajet.schema.task import Task, WorkflowOutput
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient

import sys
sys.path.insert(0, os.path.dirname(__file__))

from on_user_submit_new_requests import on_user_submit_new_requests
from on_compute_relative_reward import on_compute_relative_reward

# Configuration
SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
NUM_REPEAT = int(os.getenv("NUM_REPEAT", "4"))
TRAINING_OBJECTIVE = "Train model to be more extraverted"

# Global State
USER_REQUEST_RECORD: List[Dict] = []
REQUEST_COUNTER = 0
swarm_client: Optional[SwarmClient] = None
ajet_job = AgentJetJob(
    algorithm="grpo",
    project_name="openclaw-extraversion",
    experiment_name="extraversion_training",
    n_gpu=8,
    model='/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-3B-Instruct',
    batch_size=32,
    num_repeat=NUM_REPEAT,
    max_prompt_length=16000,    # at least 16000
    max_response_length=8000,
    max_model_len=24000,        # bigger than / equal to `max_prompt_length + max_response_length`
)

class EpisodeResult(BaseModel):
    """Result from a single episode execution."""
    episode_uuid: str
    response: Dict | List[bytes]


def extract_assistant_message(resp: Dict | List[bytes]) -> Dict:
    """Extract assistant message from response."""
    if isinstance(resp, list):
        content_parts: List[str] = []
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
                if delta.get("content"):
                    content_parts.append(delta["content"])
            except Exception:
                pass
        return {"role": "assistant", "content": "".join(content_parts)}
    else:
        return resp.get("choices", [{}])[0].get("message", {})


async def proxy_chat_completion(base_url: str, api_key: str, request: Request, is_stream: bool = False) -> Dict | List[bytes]:
    """Proxy a chat completion request."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Connection": "close",
    }
    json_data = await request.json()
    json_data["stream"] = is_stream

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{base_url}/chat/completions", json=json_data, headers=headers)
        resp.raise_for_status()
        if is_stream:
            chunks = []
            async for line in resp.aiter_lines():
                if line.strip():
                    chunks.append(line.encode() if isinstance(line, str) else line)
            return chunks
        else:
            return resp.json()


async def run_single_episode(episode_index: int, request: Request, is_stream: bool) -> EpisodeResult:
    """Run a single episode."""
    assert swarm_client is not None
    episode_uuid, api_baseurl_key = await asyncio.to_thread(swarm_client.begin_episode)
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
    """Run all episodes in parallel."""
    episode_tasks = [run_single_episode(i, request, is_stream) for i in range(NUM_REPEAT)]
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


async def finalize_episodes(task: Task, valid_results: List[EpisodeResult], rewards: List[float]) -> None:
    """Finalize all episodes by sending rewards."""
    assert swarm_client is not None
    loop = asyncio.get_event_loop()
    for episode_result, reward in zip(valid_results, rewards):
        workflow_output = WorkflowOutput(reward=reward, metadata={})
        await loop.run_in_executor(
            None,
            lambda ep=episode_result, wo=workflow_output: swarm_client.end_episode(task, ep.episode_uuid, wo),
        )


async def handle_one2many_request(request: Request, request_id: str) -> Dict | List[bytes]:
    """Handle a one-to-many request."""
    json_data = await request.json()
    is_stream = json_data.get('stream', False)
    messages = json_data.get('messages', [])
    message_latest = messages[-1]
    user_query = str(message_latest.get("content", "") if isinstance(message_latest, dict) else "")

    task = Task(task_id=str(uuid.uuid4()), main_query=user_query, metadata={"TRAINING_OBJECTIVE": TRAINING_OBJECTIVE})
    await on_user_submit_new_requests(request_id, task)

    valid_results = await run_all_episodes(request, is_stream)
    all_answers = [extract_assistant_message(r.response) for r in valid_results]
    rewards = await on_compute_relative_reward(valid_results, all_answers)

    await finalize_episodes(task, valid_results, rewards)

    best_idx = rewards.index(max(rewards))
    return valid_results[best_idx].response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global swarm_client
    logger.info(f"Initializing swarm client with URL: {SWARM_URL}")
    swarm_client = SwarmClient(SWARM_URL)
    logger.info(f"Syncing train config and starting engine with num_repeat={NUM_REPEAT}")

    def start_engine_background():
        try:
            swarm_client.auto_sync_train_config_and_start_engine(ajet_job, force_restart=False)
            logger.info("Swarm engine is ready!")
        except Exception as e:
            logger.warning(f"Engine auto-sync skipped or failed: {e}")

    engine_thread = threading.Thread(target=start_engine_background, daemon=True)
    engine_thread.start()
    yield


app = FastAPI(title="OpenClaw Extraversion Training", lifespan=lifespan)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def one2many_proxy(request: Request, path: str):
    """Main proxy endpoint."""
    global REQUEST_COUNTER
    try:
        if request.method == "POST" and path == "chat/completions":
            REQUEST_COUNTER += 1
            request_id = f"req_{REQUEST_COUNTER}_{uuid.uuid4().hex[:8]}"
            logger.info(f"Received chat completion request {request_id}")
            response_data = await handle_one2many_request(request, request_id)
            if isinstance(response_data, list):
                async def stream_chunks(chunks: List[bytes]):
                    for chunk in chunks:
                        yield chunk + b"\n\n"
                return StreamingResponse(stream_chunks(response_data), media_type="text/event-stream")
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
    return {"status": "healthy"}


@app.get("/requests")
async def get_requests():
    """Get all recorded user requests."""
    return {"requests": USER_REQUEST_RECORD}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
