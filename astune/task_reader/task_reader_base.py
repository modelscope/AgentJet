import json
import uuid
from typing import Dict, List, Optional

import datasets
import torch

from astune.schema.task import Task


class TaskReaderBase:
    def __init__(self, reader_config):
        self.reader_config = reader_config

    def get_training_tasks(self) -> List[Task]:
        raise NotImplementedError

    def get_validation_tasks(self) -> List[Task]:
        raise NotImplementedError
