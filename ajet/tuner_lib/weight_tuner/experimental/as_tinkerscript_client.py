import uuid
import time
import httpx
import yaml
from loguru import logger
from pydantic import BaseModel
from ajet.schema.task import WorkflowOutput
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.weight_tuner.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey

# --- Schema Definitions ---

class SyncTrainConfigRequest(BaseModel):
    yaml_as_string: str

class ClaimEpisodeRequest(BaseModel):
    client_uuid: str
    episode_type: str

class ClaimEpisodeResponse(BaseModel):
    success: bool
    client_uuid: str
    episode_uuid: str
    openai_base_url: str = ""
    openai_api_key: str = ""
    fail_cause: str = ""

class EndEpisodeRequest(BaseModel):
    client_uuid: str
    episode_uuid: str
    workflow_output: WorkflowOutput

class EndEpisodeResponse(BaseModel):
    success: bool

class TinkerScriptClient(object):

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client_uuid = str(uuid.uuid4())
        self.episode_uuid = None
        self.openai_base_url = None
        self.openai_api_key = None

    def begin_episode(self) -> OpenaiBaseUrlAndApiKey:
        """
        Block until an episode is claimed.
        Return (episode_uuid, openai_base_url, openai_api_key)
        """
        while True:
            try:
                req_obj = ClaimEpisodeRequest(
                    client_uuid=self.client_uuid,
                    episode_type="default"
                )
                resp = httpx.post(
                    f"{self.server_url}/claim_episode",
                    json=req_obj.model_dump(),
                    timeout=30
                )
                resp.raise_for_status()
                data = ClaimEpisodeResponse.model_validate(resp.json())

                if data.success:
                    self.episode_uuid = data.episode_uuid
                    self.openai_base_url = data.openai_base_url
                    self.openai_api_key = data.openai_api_key
                    logger.info(f"Claimed episode {self.episode_uuid}")
                    return OpenaiBaseUrlAndApiKey(
                        base_url=self.openai_base_url,
                        api_key=self.openai_api_key,
                    )
                else:
                    logger.info(f"Failed to claim episode: {data.fail_cause}. Retrying in 5s...")
                    time.sleep(5)
            except Exception as e:
                logger.error(f"Error claiming episode: {e}. Retrying in 5s...")
                time.sleep(5)

    def end_episode(self, workflow_output: WorkflowOutput):
        if not self.episode_uuid:
            logger.error("No episode to end.")
            return

        try:
            req_obj = EndEpisodeRequest(
                client_uuid=self.client_uuid,
                episode_uuid=self.episode_uuid,
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
                logger.info(f"Ended episode {self.episode_uuid}")
                self.episode_uuid = None
            else:
                 logger.error(f"Failed to end episode {self.episode_uuid}")

        except Exception as e:
            logger.error(f"Error ending episode: {e}")

    def sync_train_config(self, agent_jet_job: AgentJetJob):
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
            logger.info("Synced train config")
        except Exception as e:
            logger.error(f"Error syncing train config: {e}")
