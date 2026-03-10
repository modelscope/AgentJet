# -*- coding: utf-8 -*-

"""
A one-to-many proxy server.

------------------------------------------------------
------------------------------------------------------
When this server initialize, do the following things:
1. connect to swarm server, sync training config and start engine with specified AgentJetJob, wait until the engine is ready.
```
SWARM_URL = "http://localhost:10086"
num_repeat = 8
swarm_client = SwarmClient(SWARM_URL)
swarm_client.auto_sync_train_config_and_start_engine(
    AgentJetJob(
        algorithm="grpo",
        project_name="ajet-swarm",
        experiment_name="test",
        n_gpu=8,
        model='/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct',
        batch_size=16,
        num_repeat=num_repeat,
    ),
)
```

2. read a user preference: 我希望我的助手足够幽默


------------------------------------------------------
------------------------------------------------------
This server do the following things when receiving a LLM request:
0. init task:
    Task(task_id="{a_random_uuid}, main_query="{user_request_message}")
1. keep a record of user requests, ordered by arrival time, and assign a unique request_id to each request.
2. repeat `num_repeat` times (in parallel):
    2-1. get episode base-url and api-key of this `repeat`:
        ```
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=120)
        ```
    2-2. proxy chat completion request to the base-url with api-key, store (episode_response, episode_uuid).

3. when all `num_repeat` episodes finish:
    3-1. compare all episode_responses, generate score (-1 ~ +1) for each episode_response (write a random score generator for now)
    3-2. now we should have (episode_response_array, episode_uuid_array, episode_relative_reward_array)
    3-3. run for loop:
        for ... in (episode_response_array, episode_uuid_array, episode_relative_reward_array)
            workflow_output = WorkflowOutput(
                reward=relative_reward,
                metadata={},
            )
            swarm_worker.end_episode(task, episode_uuid, workflow_output)

4. select the episode_response with the highest score, return it to user (pretend it is a stream, although it is actually not).

5. end this request.


------------------------

for swarm api, refer to tutorial/example_math_swarm/math.py


# python -m ajet.tuner_lib.experimental.oai_model_one2many
# python -m ajet.tuner_lib.experimental.oai_model_one2many_client


"""

import os
import uuid
import random
import asyncio
import httpx
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from ajet.schema.task import Task, WorkflowOutput
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.tuner_lib.experimental.interchange_utils import (
    ClaimEpisodeRequest,
    ClaimEpisodeResponse,
)


SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
NUM_REPEAT = int(os.getenv("NUM_REPEAT", "8"))
USER_REQUEST_RECORD: List[Dict] = []
REQUEST_COUNTER = 0
USER_PREFERENCE = "我希望我的助手足够幽默"

swarm_client: Optional[SwarmClient] = SwarmClient(SWARM_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global swarm_client
    logger.info(f"Initializing swarm client with URL: {SWARM_URL}")
    swarm_client = SwarmClient(SWARM_URL)

    ajet_job = AgentJetJob(
        algorithm="grpo",
        project_name="ajet-swarm",
        experiment_name="test",
        n_gpu=8,
        model='/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct',
        batch_size=16,
        num_repeat=NUM_REPEAT,
    )

    logger.info(f"Syncing train config and starting engine with num_repeat={NUM_REPEAT}")

    import threading
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



class ChatCompletionRequest(BaseModel):
    model: str = "fill_whatever_model"
    messages: List[Dict[str, str]]
    stream: bool = False


class EpisodeResult(BaseModel):
    episode_uuid: str
    response: Dict
    reward: float


async def proxy_chat_completion(
    base_url: str,
    api_key: str,
    request_data: ChatCompletionRequest
) -> Dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Connection": "close",
    }

    # Force stream=False for internal requests
    request_dict = request_data.model_dump()
    request_dict["stream"] = False

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json=request_dict,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()


def generate_random_score() -> float:
    return random.uniform(-1.0, 1.0)


def begin_episode_direct(swarm_client: SwarmClient, discard_episode_timeout: int = 120, max_retries: int = 10) -> Tuple[str, OpenaiBaseUrlAndApiKey]:
    """Custom begin_episode that doesn't replace the base_url host."""

    for attempt in range(max_retries):
        try:
            req_obj = ClaimEpisodeRequest(
                client_uuid=swarm_client.client_uuid,
                episode_type="train",
                discard_episode_timeout=discard_episode_timeout,
                throttle_policy=None
            )
            resp = swarm_client._http_client.post(
                f"{swarm_client.server_url}/claim_episode",
                json=req_obj.model_dump()
            )
            resp.raise_for_status()
            data = ClaimEpisodeResponse.model_validate(resp.json())

            if data.success:
                episode_uuid = data.episode_uuid
                openai_base_url = data.openai_base_url
                openai_api_key = data.openai_api_key

                logger.info(f"Claimed episode {episode_uuid}")
                return episode_uuid, OpenaiBaseUrlAndApiKey(
                    base_url=openai_base_url,
                    api_key=openai_api_key,
                    episode_uuid=episode_uuid
                )
            else:
                logger.warning(f"Failed to claim episode: {data.fail_cause}")
                if "No available episodes" in data.fail_cause:
                    time.sleep(2)
                else:
                    time.sleep(5)
        except Exception as e:
            logger.error(f"Error claiming episode: {e}")
            time.sleep(2)

    raise RuntimeError(f"Failed to claim episode after {max_retries} attempts")


async def handle_one2many_request(request_data: ChatCompletionRequest, request_id: str) -> Dict:
    global USER_REQUEST_RECORD

    task = Task(
        task_id=str(uuid.uuid4()),
        main_query=request_data.messages[-1]["content"] if request_data.messages else "",
        metadata={"user_preference": USER_PREFERENCE}
    )

    assert swarm_client is not None, "Swarm client not initialized"

    USER_REQUEST_RECORD.append({
        "request_id": request_id,
        "task_id": task.task_id,
        "query": task.main_query,
    })

    async def run_episode(episode_index: int) -> EpisodeResult:
        loop = asyncio.get_event_loop()
        episode_uuid, api_baseurl_key = await loop.run_in_executor(
            None, lambda: begin_episode_direct(swarm_client, 300)  # type: ignore[arg-type]
        )

        try:
            response_data = await proxy_chat_completion(
                base_url=api_baseurl_key.base_url,
                api_key=api_baseurl_key.api_key,
                request_data=request_data,
            )

            reward = generate_random_score()

            return EpisodeResult(
                episode_uuid=episode_uuid,
                response=response_data,
                reward=reward,
            )
        except Exception as e:
            logger.error(f"Error in episode {episode_index}: {e}")
            swarm_client.abort_episode(episode_uuid)  # type: ignore[union-attr]
            raise

    tasks = [run_episode(i) for i in range(NUM_REPEAT)]
    episode_results: List[EpisodeResult | BaseException] = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results: List[EpisodeResult] = []
    for result in episode_results:
        if isinstance(result, Exception):
            logger.error(f"Episode failed with exception: {result}")
            continue
        if isinstance(result, EpisodeResult):
            valid_results.append(result)

    for result in valid_results:
        workflow_output = WorkflowOutput(
            reward=result.reward,
            metadata={},
        )
        swarm_client.end_episode(task, result.episode_uuid, workflow_output)  # type: ignore[union-attr]

    if not valid_results:
        raise HTTPException(status_code=500, detail="All episodes failed")

    best_result = max(valid_results, key=lambda x: x.reward)

    return best_result.response


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def one2many_proxy(request: Request, path: str):
    global REQUEST_COUNTER

    try:
        if request.method == "POST" and path == "chat/completions":
            body = await request.json()
            request_data = ChatCompletionRequest(**body)

            REQUEST_COUNTER += 1
            request_id = f"req_{REQUEST_COUNTER}_{uuid.uuid4().hex[:8]}"

            logger.info(f"Received chat completion request {request_id}")

            if request_data.stream:
                response_data = await handle_one2many_request(request_data, request_id)

                async def stream_response():
                    import json
                    yield f"data: {json.dumps(response_data)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(stream_response(), media_type="text/event-stream")
            else:
                response_data = await handle_one2many_request(request_data, request_id)
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
    return {"status": "healthy", "user_preference": USER_PREFERENCE}


@app.get("/requests")
async def get_requests():
    return {"requests": USER_REQUEST_RECORD}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
