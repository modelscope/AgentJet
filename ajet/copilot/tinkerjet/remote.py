from ajet import AgentJetJob
from ajet.schema.task import WorkflowOutput
from ajet.tuner_lib.weight_tuner.as_oai_baseurl_apikey import AgentJetAsOpenAI


class TinkerJetRemote(object):
    def __init__(self, remote_url: str = "http://localhost:10086"):
        ...


    def sync_train_config(self, job: AgentJetJob):
        ...


    def begin_episode(self) -> AgentJetAsOpenAI:
        ...


    def end_episode(self, workflow_output: WorkflowOutput):
        ...


    def download_tuned_model(self):
        ...