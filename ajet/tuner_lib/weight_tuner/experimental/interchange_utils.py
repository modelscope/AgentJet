import os
import time
import httpx
from typing import List
from pydantic import BaseModel
from ajet.schema.task import WorkflowOutput
from loguru import logger
from ajet.utils.networking import find_free_port


class SyncTrainConfigRequest(BaseModel):
    yaml_as_string: str

class ClaimEpisodeRequest(BaseModel):
    client_uuid: str
    episode_type: str
    allow_discard_timeout: float

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

class EndEpisodeResponse(BaseModel):
    success: bool


class EpisodeStatus(BaseModel):
    episode_uuid: str
    episode_status: str = "rolling"
    openai_base_url: str = ""
    openai_api_key: str = ""
    client_uuid: str = ""
    zmq_listen_result_addr: str = ""
    latest_activity_timestamp: float = time.time()
    allow_discard_timeout: float

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


def get_interchange_server_url(config):
    port = os.getenv("AJET_DAT_INTERCHANGE_PORT")
    if config.ajet.interchange_server.interchange_server_port != 'auto':
        port = str(int(config.ajet.interchange_server.interchange_server_port))
    assert port is not None, "AJET_DAT_INTERCHANGE_PORT env var must be set"
    master_node_ip = os.getenv("MASTER_NODE_IP", "localhost")
    base_url = f"http://{master_node_ip}:{port}"
    return base_url


def http_change_engine_status(config: str, new_status: str):
    if new_status not in [
        "ENGINE.OFF",
        "ENGINE.BOOTING",
        "ENGINE.ROLLING",
        "ENGINE.WEIGHT_SYNCING",
        "ENGINE.WEIGHT_EXPORTING"
    ]:
        raise ValueError(f"Invalid engine status: {new_status}")

    resp = httpx.post(
        f"{get_interchange_server_url(config)}/update_engine_status",
        json={"engine_status": new_status},
        timeout=10
    )
    resp.raise_for_status()
    logger.info(f"Changed engine status to {new_status}")



def http_register_episode(config, episode_uuid: str,
                            openai_base_url: str, openai_api_key: str,
                            zmq_listen_result_addr: str):

    # parse episode_uuid, openai_base_url, openai_api_key
    interchange_http_addr = get_interchange_server_url(config)
    rer = RegisterEpisodeRequest(
        episode_uuid=episode_uuid,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        zmq_listen_result_addr=zmq_listen_result_addr,
    )
    # send http request to tinkerscript server to register episode
    while True:
        try:
            response = httpx.post(
                f"{interchange_http_addr}/register_episode",
                json=rer.model_dump(),  # 或者 rer.model_dump() 如果使用 Pydantic v2
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            if not result.get('success'):
                raise RuntimeError(f"Failed to register episode {episode_uuid}")
            logger.info(f"Successfully registered episode {episode_uuid}")
            break
        except httpx.HTTPError as e:
            logger.error(f"Error registering episode {episode_uuid}: {e}. Retrying...")
            time.sleep(5)

    return rer


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
