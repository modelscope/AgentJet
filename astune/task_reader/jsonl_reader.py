import json
import uuid
import torch
import datasets
from typing import List, Dict, Optional
from astune.schema.task import Task
from astune.utils.process_dataset import create_rl_dataset, create_rl_sampler
from astune.env_service_client.env_client_ng import EnvClient
from astune.task_reader.task_reader_base import TaskReaderBase



class TaskReaderJsonl(TaskReaderBase):
    def __init__(self, config):
        super().__init__(config)

    def _read_jsonl_file(self, file_path):
        """
        Read tasks from a JSONL file.

        Args:
            file_path (str): Path to the JSONL file.

        Returns:
            List[Task]: List of Task objects.
        """
        tasks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        task_data = json.loads(line)
                        # Create a Task object from the JSON data
                        task = Task(
                            main_query=task_data.get('main_query', '[not defined]'),
                            init_messages=task_data.get('init_messages', []),
                            task_id=task_data.get('task_id', ''),
                            env_type=task_data.get('env_type', 'no_env'),
                            metadata=task_data.get('metadata', {})
                        )
                        tasks.append(task)
        except FileNotFoundError:
            raise ValueError(f"JSONL file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {str(e)}")

        if len(tasks) == 0:
            raise ValueError(f"No tasks found in file: {file_path}")

        return tasks

    def get_training_tasks(self) -> List[Task]:
        """
        Get training tasks from the JSONL file specified in the config.

        Returns:
            List[Task]: List of training Task objects.
        """
        file_path = self.config.astune.task_reader.dataset_file.training.file_path
        return self._read_jsonl_file(file_path)

    def get_validation_tasks(self) -> List[Task]:
        """
        Get validation tasks from the JSONL file specified in the config.

        Returns:
            List[Task]: List of validation Task objects.
        """
        file_path = self.config.astune.task_reader.dataset_file.validation.file_path
        return self._read_jsonl_file(file_path)
