import json
import uuid
import torch
import datasets
from typing import List, Dict, Optional
from astune.schema.task import Task
from astune.utils.process_dataset import create_rl_dataset, create_rl_sampler
from astune.env_service_client.env_client_ng import EnvClient


class TaskReaderBase:
    def __init__(self, config):
        self.config = config

    def get_training_tasks(self)->List[Task]:
        raise NotImplementedError

    def get_validation_tasks(self)->List[Task]:
        raise NotImplementedError

