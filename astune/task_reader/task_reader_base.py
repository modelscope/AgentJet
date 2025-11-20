import json
import uuid
import torch
import datasets
from typing import List, Dict, Optional
from astune.schema.task import Task


class TaskReaderBase:
    def __init__(self, config):
        self.config = config

    def get_training_tasks(self) -> List[Task]:
        raise NotImplementedError

    def get_validation_tasks(self) -> List[Task]:
        raise NotImplementedError
