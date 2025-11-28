from typing import List

from astuner.schema.task import Task


class TaskReaderBase:
    def __init__(self, reader_config):
        self.reader_config = reader_config

    def get_training_tasks(self) -> List[Task]:
        raise NotImplementedError

    def get_validation_tasks(self) -> List[Task]:
        raise NotImplementedError
