import os
import time
import httpx
from typing import List, Dict
from pydantic import BaseModel
from loguru import logger
from ajet.schema.task import WorkflowOutput
from ajet.utils.networking import find_free_port
from ajet.utils.retry import retry_with_backoff
from ajet.utils.cache import cache_with_ttl

VALID_STATUSES = [
    "ENGINE.OFFLINE",
    "ENGINE.BOOTING",
    "ENGINE.ROLLING",
    "ENGINE.ROLLING_POST",
    "ENGINE.WEIGHT_SYNCING",
    "ENGINE.WEIGHT_EXPORTING"
]

class SyncTrainConfigRequest(BaseModel):
    yaml_as_string: str

class ClaimEpisodeRequest(BaseModel):
    client_uuid: str
    episode_type: str
    discard_episode_timeout: float

class ClaimEpisodeResponse(BaseModel):
    success: bool
    client_uuid: str
    episode_uuid: str
    openai_base_url: str = ""
    openai_api_key: str = ""
    fail_cause: str = ""

class CanContinueEpisodeRequest(BaseModel):
    client_uuid: str
    episode_uuid: str

class CanContinueEpisodeResponse(BaseModel):
    can_continue: bool

class EndEpisodeRequest(BaseModel):
    client_uuid: str
    episode_uuid: str
    workflow_output: WorkflowOutput
    task_id: str

class EndEpisodeResponse(BaseModel):
    success: bool


class EpisodeStatus(BaseModel):
    episode_uuid: str
    episode_status: str = ""
    episode_type: str = "train"
    openai_base_url: str = ""
    openai_api_key: str = ""
    client_uuid: str = ""
    zmq_listen_result_addr: str = ""
    latest_activity_timestamp: float = time.time()
    discard_episode_timeout: float
    debug_log: List[str] = []

class EpisodeBufferResponse(BaseModel):
    buffer: List[EpisodeStatus]

class BoolResponse(BaseModel):
    success: bool
    failure_reason: str = ""

class RegisterEpisodeRequest(BaseModel):
    episode_uuid: str
    openai_base_url: str = ""
    openai_api_key: str = ""
    zmq_listen_result_addr: str = ""


class UpdateEngineStatusRequest(BaseModel):
    engine_status: str = ""
    engine_status_detail: str|None = None
    global_step: int|None = None


class CurrentBatchRolloutPoolInformation(BaseModel):
    sample_collection_method: str = ""
    completed_episodes: int = 0
    completed_episode_target: int = 0
    completed_tasks: int = 0
    completed_task_target: int = 0
    completed_non_dummy_tasks: int = 0
    completed_non_dummy_task_target: int = 0
    task_expected_num_repeat: int = 0
    completed_tasks_details: Dict[str, List[str]] = {}  # task_id -> list of episode_uuids
    running_episode_details: Dict[str, Dict[str, str]] | None = None # episode_uuid -> { "episode_status": ..., "time_since_last_activity": ...}
    engine_status: str | None = None
    global_step: int | None = None


DEBUG = False
# DEBUG = True

VERBOSE = True

def get_interchange_server_url(config):
    port = os.getenv("AJET_DAT_INTERCHANGE_PORT")
    if config.ajet.interchange_server.interchange_server_port != 'auto':
        port = str(int(config.ajet.interchange_server.interchange_server_port))
    assert port is not None, "AJET_DAT_INTERCHANGE_PORT env var must be set"
    master_node_ip = os.getenv("MASTER_NODE_IP", "localhost")
    base_url = f"http://{master_node_ip}:{port}"
    return base_url


def http_change_engine_status(config, new_status: str, new_status_detail: str|None = None, global_step: int|None = None):
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Invalid engine status: {new_status}")

    resp = httpx.post(
        f"{get_interchange_server_url(config)}/update_engine_status",
        json={"engine_status": new_status, "engine_status_detail": new_status_detail, "global_step": global_step},
        timeout=10
    )
    resp.raise_for_status()
    logger.success(f"Changed engine status to {new_status}")


@cache_with_ttl(ttl=1.0)
def is_episode_claimed(config, episode_uuid: str) -> bool:
    resp = httpx.post(
        f"{get_interchange_server_url(config)}/is_episode_claimed",
        json={"client_uuid": "", "episode_uuid": episode_uuid},
        timeout=5
    )
    resp.raise_for_status()
    result = BoolResponse.model_validate(resp.json())
    return result.success


@retry_with_backoff(max_retry=15, backoff_fn=lambda attempt: 2)
def http_register_episode(config,
                          episode_uuid: str,
                          openai_base_url: str,
                          openai_api_key: str,
                          zmq_listen_result_addr: str,
                          should_exit_soft):

    if should_exit_soft():
        logger.warning(f"Exiting before registering episode {episode_uuid}")
        return None

    # parse episode_uuid, openai_base_url, openai_api_key
    interchange_http_addr = get_interchange_server_url(config)
    rer = RegisterEpisodeRequest(
        episode_uuid=episode_uuid,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        zmq_listen_result_addr=zmq_listen_result_addr,
    )
    # send http request to swarm server to register episode
    response = httpx.post(
        f"{interchange_http_addr}/register_episode",
        json=rer.model_dump(),  # 或者 rer.model_dump() 如果使用 Pydantic v2
        timeout=2
    )
    response.raise_for_status()
    result = response.json()
    if not result.get('success'):
        logger.warning(f"Failed to register episode {episode_uuid}")
        return None
    if DEBUG: logger.info(f"Successfully registered episode {episode_uuid}")

    return True


def http_update_rollout_pool_information(config, pool_info: CurrentBatchRolloutPoolInformation):
    """
    Update the rollout pool information on the interchange server.

    Args:
        config: The configuration object
        pool_info: CurrentBatchRolloutPoolInformation object with rollout statistics
    """
    try:
        resp = httpx.post(
            f"{get_interchange_server_url(config)}/update_current_batch_rollout_pool_information",
            json=pool_info.model_dump(),
            timeout=5
        )
        resp.raise_for_status()
    except Exception as e:
        if DEBUG:
            logger.warning(f"Failed to update rollout pool information: {e}")


def get_zmq_socket(config, episode_uuid: str, tag: str = ""):
    interchange_method = config.ajet.interchange_server.interchange_method
    if interchange_method == 'tcp':
        ipc_path = ""
        master_node_ip = os.getenv("MASTER_NODE_IP", "localhost")
        zmq_contect_address = f"tcp://{master_node_ip}:{find_free_port()}"
    elif interchange_method == 'ipc':
        ipc_path = f"/tmp/ajet/{episode_uuid}-{tag}.sock"
        zmq_contect_address = f"ipc://{ipc_path}"
    else:
        raise RuntimeError(f"Unknown interchange_method: {interchange_method}")
    return zmq_contect_address, ipc_path
