import uuid
import time
import httpx
import yaml
from typing import List, Tuple
from loguru import logger
from ajet.schema.task import WorkflowOutput
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.weight_tuner.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.tuner_lib.weight_tuner.experimental.interchange_utils import (
    SyncTrainConfigRequest,
    ClaimEpisodeRequest,
    ClaimEpisodeResponse,
    CanContinueEpisodeRequest,
    CanContinueEpisodeResponse,
    EndEpisodeRequest,
    EndEpisodeResponse,
    EpisodeStatus,
    EpisodeBufferResponse,
)


class TinkerScriptClient(object):

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client_uuid = str(uuid.uuid4())


    def begin_episode(self, allow_discard_timeout=60) -> Tuple[str, OpenaiBaseUrlAndApiKey]:
        """
        Block until an episode is claimed.
        Return (episode_uuid, openai_base_url, openai_api_key)
        """
        while True:
            try:
                req_obj = ClaimEpisodeRequest(
                    client_uuid=self.client_uuid,
                    episode_type="default",
                    allow_discard_timeout=allow_discard_timeout,
                )
                resp = httpx.post(
                    f"{self.server_url}/claim_episode",
                    json=req_obj.model_dump(),
                    timeout=30
                )
                resp.raise_for_status()
                data = ClaimEpisodeResponse.model_validate(resp.json())
                episode_uuid = data.episode_uuid

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
                    logger.info(f"Failed to claim episode: {data.fail_cause}. Retrying in 5s...")
                    time.sleep(5)
            except Exception as e:
                logger.error(f"Error claiming episode: {e}. Retrying in 5s...")
                time.sleep(5)

    def end_episode(self, episode_uuid: str, workflow_output: WorkflowOutput):
        if not episode_uuid:
            logger.error("No episode to end.")
            return

        try:
            req_obj = EndEpisodeRequest(
                client_uuid=self.client_uuid,
                episode_uuid=episode_uuid,
                workflow_output=workflow_output
            )

            resp = httpx.post(
                f"{self.server_url}/end_episode",
                json=req_obj.model_dump(),
                timeout=30
            )
            resp.raise_for_status()
            data = EndEpisodeResponse.model_validate(resp.json())

            if data.success:
                logger.info(f"Ended episode {episode_uuid}")
            else:
                logger.error(f"Failed to end episode {episode_uuid}")

        except Exception as e:
            logger.error(f"Error ending episode: {e}")

    def sync_train_config(self, agent_jet_job: AgentJetJob):
        """
        Sync training configuration to the TinkerScript server.
        This sends the AgentJetJob config as YAML to the remote server.
        """
        try:
            config_dict = agent_jet_job.config.to_dict()
            yaml_str = yaml.safe_dump(config_dict, sort_keys=False)

            req_obj = SyncTrainConfigRequest(yaml_as_string=yaml_str)

            resp = httpx.post(
                f"{self.server_url}/sync_train_config",
                json=req_obj.model_dump(),
                timeout=30
            )
            resp.raise_for_status()
            logger.info("Synced train config to TinkerScript server")
        except Exception as e:
            logger.error(f"Error syncing train config: {e}")
            raise

    def start_engine(self):
        """
        Start the training engine on the TinkerScript server.
        This triggers the server to begin the training process.
        Polls until engine status is "ENGINE.ROLLING".
        """
        try:
            resp = httpx.post(
                f"{self.server_url}/start_engine",
                json={},
                timeout=600
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("success"):
                logger.info("Successfully started training engine on TinkerScript server")
            else:
                logger.error("Failed to start training engine")
                raise RuntimeError("Failed to start training engine")
        except Exception as e:
            logger.error(f"Error starting engine: {e}")
            raise

        # Poll until engine status is "ENGINE.ROLLING"
        logger.info("Polling engine status until ENGINE.ROLLING...")
        last_report_time = time.time()

        while True:
            try:
                current_status = self.get_engine_status()
                current_time = time.time()

                # Report status every 5 seconds
                if current_time - last_report_time >= 5:
                    logger.info(f"Current engine status: {current_status}")
                    last_report_time = current_time

                # Check if engine has reached the desired status
                if current_status == "ENGINE.ROLLING":
                    logger.info("Engine status is ENGINE.ROLLING - engine is ready")
                    break

                # Wait a bit before next poll
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error polling engine status: {e}")
                time.sleep(5)

    def get_engine_status(self) -> str:
        try:
            resp = httpx.get(
                f"{self.server_url}/get_engine_status",
                timeout=10
            )
            resp.raise_for_status()
            return resp.json().get("engine_status", "unknown")
        except Exception as e:
            logger.error(f"Error getting engine status: {e}")
            return "unknown"

    def can_continue_episode(self, episode_uuid: str) -> bool:
        if not episode_uuid:
            return False

        try:
            req_obj = CanContinueEpisodeRequest(
                client_uuid=self.client_uuid,
                episode_uuid=episode_uuid
            )
            resp = httpx.post(
                f"{self.server_url}/can_continue_episode",
                json=req_obj.model_dump(),
                timeout=10
            )
            resp.raise_for_status()
            data = CanContinueEpisodeResponse.model_validate(resp.json())
            return data.can_continue
        except Exception as e:
            logger.error(f"Error checking can_continue_episode: {e}")
            return False

    def get_episode_buffer(self) -> List[EpisodeStatus]:
        try:
            resp = httpx.post(
                f"{self.server_url}/get_episode_buffer",
                json={},
                timeout=10
            )
            resp.raise_for_status()
            data = EpisodeBufferResponse.model_validate(resp.json())
            return data.buffer
        except Exception as e:
            logger.error(f"Error getting episode buffer: {e}")
            return []
