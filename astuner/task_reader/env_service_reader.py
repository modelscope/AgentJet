from astuner.schema.task import Task
from astuner.task_reader.task_reader_base import BaseTaskReader
from astuner.utils.env_service_client.env_client_ng import EnvClient


class EnvServiceTaskReader(BaseTaskReader):
    def __init__(self, reader_config):
        super().__init__(reader_config)
        self.reader_config = reader_config

    def get_tasks(self, split):
        env_url = self.reader_config.env_service.env_url
        env_type = self.reader_config.env_service.env_type
        env_service_client = EnvClient(base_url=env_url)
        task_id_array = env_service_client.get_env_profile(env_type, split=split)
        if len(task_id_array) == 0:
            raise ValueError(
                f"No task_id found for env_type: {env_type}, split: {split}, Please check connection to {env_url}"
            )
        tasks = [
            Task(
                main_query="[not defined]",
                init_messages=[],
                task_id=str(task_id),
                env_type=env_type,
                metadata={},
            )
            for task_id in task_id_array
        ]
        return tasks

    def get_validation_tasks(self):
        split = self.reader_config.env_service.validation_split
        return self.get_tasks(split=split)

    def get_training_tasks(self):
        split = self.reader_config.env_service.training_split
        return self.get_tasks(split=split)
