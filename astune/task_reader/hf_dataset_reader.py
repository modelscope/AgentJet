import json
import uuid
import torch
import datasets
from typing import List, Dict, Optional
from astune.schema.task import Task
from astune.env_service_client.env_client_ng import EnvClient
from astune.task_reader.task_reader_base import TaskReaderBase



class TaskReaderHuggingFace(TaskReaderBase):
    """
    Task reader that reads tasks from Hugging Face datasets.

    This class allows loading tasks directly from Hugging Face dataset repositories.
    It supports configuring the dataset name and split names for training and validation.
    """

    def __init__(self, config):
        super().__init__(config)


    def _load_dataset_split(self, dataset_name: str, split: str) -> List[Task]:
        """
        Load a dataset split from Hugging Face datasets.

        Args:
            dataset_name: Name of the dataset in Hugging Face format (e.g., 'gsm8k')
            split: Name of the split to load (e.g., 'train', 'validation')

        Returns:
            List[Task]: List of Task objects created from the dataset.
        """
        try:
            dataset = datasets.load_dataset(dataset_name, split=split)
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{dataset_name}' with split '{split}': {str(e)}")

        # if len(dataset) == 0:
        #     raise ValueError(f"No examples found in dataset '{dataset_name}' with split '{split}'")

        tasks = []
        for idx, example in enumerate(dataset):
            # Create Task object
            task = Task(
                main_query=example['question'],
                init_messages=[],  # Dataset examples typically don't have init messages
                task_id=str(idx),
                env_type=f"no_env",
                metadata=example,
            )
            tasks.append(task)

        return tasks

    def get_training_tasks(self) -> List[Task]:
        """
        Get training tasks from the Hugging Face dataset specified in the config.

        Returns:
            List[Task]: List of training Task objects.
        """
        dataset_name = self.config.astune.task_reader.huggingface_dat_repo.dataset_path
        split = self.config.astune.task_reader.huggingface_dat_repo.training_split
        return self._load_dataset_split(dataset_name, split)

    def get_validation_tasks(self) -> List[Task]:
        """
        Get validation tasks from the Hugging Face dataset specified in the config.

        Returns:
            List[Task]: List of validation Task objects.
        """
        dataset_name = self.config.astune.task_reader.huggingface_dat_repo.dataset_path
        split = self.config.astune.task_reader.huggingface_dat_repo.validation_split
        return self._load_dataset_split(dataset_name, split)
