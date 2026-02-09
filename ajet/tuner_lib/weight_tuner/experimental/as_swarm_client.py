import uuid
import time
import httpx
import yaml
from typing import List, Tuple
from loguru import logger
from ajet.schema.task import WorkflowOutput, Task
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


class SwarmClient(object):

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client_uuid = str(uuid.uuid4())
        self.previous_warning_time = 0
        self.record_episode_expire_time = {}


    def begin_episode(self, allow_discard_timeout=60, episode_type="train") -> Tuple[str, OpenaiBaseUrlAndApiKey]:
        """
        Block until an episode is claimed.
        Return (episode_uuid, openai_base_url, openai_api_key)
        """
        while True:
            try:
                req_obj = ClaimEpisodeRequest(
                    client_uuid=self.client_uuid,
                    episode_type=episode_type,
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
                self.record_episode_expire_time[episode_uuid] = time.time() + allow_discard_timeout

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
                    need_wait_scenarios =[
                        "Engine is syncing weights",
                        "Engine is in post-rolling phase",
                        "No available episodes to claim.",
                    ]
                    if any(scenario in data.fail_cause for scenario in need_wait_scenarios):
                        if time.time() - self.previous_warning_time > 60:
                            logger.info(f"{data.fail_cause}. Retrying in 30s...")
                            self.previous_warning_time = time.time()
                        time.sleep(30)
                    else:
                        logger.warning(f"Failed to claim episode: {data.fail_cause}. Retrying in 5s...")
                        time.sleep(5)
            except Exception as e:
                logger.error(f"Error claiming episode: {e}. Retrying in 5s...")
                time.sleep(5)

    def end_episode(self, task:Task, episode_uuid: str, workflow_output: WorkflowOutput):
        if not episode_uuid:
            logger.error("No episode to end.")
            return

        remain_time = self.record_episode_expire_time.get(episode_uuid, 0) - time.time()
        if remain_time < 0:
            logger.warning(f"Episode {episode_uuid} has expired (expired {remain_time} seconds ago). Please use a larger `allow_discard_timeout` when `begin_episode`. Skipping end_episode.")
            return

        try:
            task_id = task.task_id
            workflow_output.metadata["task_id"] = task_id
            req_obj = EndEpisodeRequest(
                client_uuid=self.client_uuid,
                episode_uuid=episode_uuid,
                workflow_output=workflow_output,
                task_id=task_id
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

    def abort_episode(self, episode_uuid: str):
        if not episode_uuid:
            logger.error("No episode to end.")
            return

        try:
            workflow_output = WorkflowOutput(reward=0.0, metadata={})
            req_obj = EndEpisodeRequest(
                client_uuid=self.client_uuid,
                episode_uuid=episode_uuid,
                workflow_output=workflow_output,
                task_id=""
            )

            resp = httpx.post(
                f"{self.server_url}/abort_episode",
                json=req_obj.model_dump(),
                timeout=30
            )
            resp.raise_for_status()
            data = EndEpisodeResponse.model_validate(resp.json())

            if data.success:
                logger.info(f"Aborted episode {episode_uuid}")
            else:
                logger.error(f"Failed to end episode {episode_uuid}")

        except Exception as e:
            logger.error(f"Error ending episode: {e}")

    def sync_train_config(self, agent_jet_job: AgentJetJob):
        """
        Sync training configuration to the Swarm server.
        This sends the AgentJetJob config as YAML to the remote server.
        """
        # try get init status
        current_status = self.get_engine_status()
        if current_status != "ENGINE.OFFLINE":
            raise RuntimeError(f"Cannot sync train config when engine is NOT ENGINE.OFFLINE. (current status: {current_status})")

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
            logger.info("Synced train config to Swarm server")
        except Exception as e:
            logger.error(f"Error syncing train config: {e}")
            raise

    def start_engine(self):
        """
        Start the training engine on the Swarm server.
        This triggers the server to begin the training process.
        Polls until engine status is "ENGINE.ROLLING".
        """
        # try get init status
        current_status = self.get_engine_status()
        if current_status != "ENGINE.OFFLINE":
            raise RuntimeError(f"Cannot start engine when engine is NOT ENGINE.OFFLINE. (current status: {current_status})")

        # Send start engine request
        try:
            resp = httpx.post(
                f"{self.server_url}/start_engine",
                json={},
                timeout=600
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("success"):
                logger.info("Successfully started training engine on Swarm server")
            else:
                logger.error("Failed to start training engine")
                raise RuntimeError("Failed to start training engine")
        except Exception as e:
            logger.error(f"Error starting engine: {e}")
            raise

        # Poll until engine status is "ENGINE.ROLLING"
        self._wait_until_status_change_to(desired_status="ENGINE.ROLLING")
        logger.success("Training engine is now ROLLING and ready.")

    def _wait_until_status_change_to(self, desired_status="ENGINE.ROLLING"):
        """
        Poll engine status until it reaches desired_status.
        Reports status every 5 seconds while waiting.
        """
        logger.info(f"Polling engine status until {desired_status}...")
        last_report_time = time.time()
        init_poll_time = last_report_time

        while True:
            try:
                current_status = self.get_engine_status()
                current_time = time.time()

                # Report status every 5 seconds
                if current_time - last_report_time >= 10:
                    logger.info(f"Current engine status (already waited {current_time - init_poll_time:.1f}s): {current_status}")
                    last_report_time = current_time

                # Check if engine has reached the desired status
                if current_status == desired_status:
                    logger.info(f"Engine status is {desired_status}.")
                    break

                # Wait a bit before next poll
                time.sleep(5)

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
            result = resp.json().get("engine_status", "unknown")
            if result == "unknown":
                logger.warning("get_engine_status: " + resp.json())
            return result
        except Exception as e:
            logger.error(f"Error getting engine status: {e}")
            return "ENGINE.CANNOT_CONNECT"

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

    def auto_sync_train_config_and_start_engine(self, agent_jet_job: AgentJetJob, force_restart=False):
        """
        Automatically sync training configuration and start the engine if needed.
        This checks the current engine status and performs actions accordingly.

        Args:
            - agent_jet_job: The AgentJetJob configuration to sync.
            - force_restart: If True, forces a restart of the engine.
        """
        if force_restart:
            logger.warning("Force restarting the engine...")
            self.stop_engine()
            time.sleep(8)
        current_status = self.get_engine_status()
        if current_status == "ENGINE.OFFLINE":
            logger.info("Engine is OFFLINE. Syncing train config and starting engine...")
            self.sync_train_config(agent_jet_job)
            self.start_engine()
        elif current_status == "ENGINE.ROLLING":
            logger.info("Engine is already ROLLING. No action needed.")
        elif current_status == "ENGINE.ROLLING_POST":
            logger.info("Engine is already ROLLING. No action needed.")
        elif current_status == "ENGINE.BOOTING":
            logger.info("Engine is BOOTING. Waiting until it becomes ROLLING...")
            self._wait_until_status_change_to(desired_status="ENGINE.ROLLING")
            logger.success("Training engine is now ROLLING and ready.")
        elif current_status == "ENGINE.CANNOT_CONNECT":
            logger.error("Cannot connect to the engine. Please check the network.")
            self._wait_until_status_change_to(desired_status="ENGINE.ROLLING")
            logger.success("Training engine is now ROLLING and ready.")
        else:
            raise RuntimeError(f"Cannot sync train config or start engine when engine is in status: {current_status}")

    def stop_engine(self):
        """
        Stop the training engine on the Swarm server.
        This triggers the server to stop the training process.
        """
        current_status = self.get_engine_status()
        if current_status == "ENGINE.OFFLINE":
            logger.info("Engine is already OFFLINE. No action needed.")
            return

        try:
            resp = httpx.post(
                f"{self.server_url}/stop_engine",
                json={},
                timeout=600
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("success"):
                logger.info("Successfully stopped training engine on Swarm server")
            else:
                logger.error("Failed to stop training engine")
            self._wait_until_status_change_to(desired_status="ENGINE.OFFLINE")
        except Exception as e:
            logger.error(f"Error stopping engine: {e}")
