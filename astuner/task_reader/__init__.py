from typing import List

import datasets
import numpy as np

from astuner.schema.task import Task
from astuner.task_reader.data_generator_reader import TaskReaderDataGenerator
from astuner.task_reader.env_service_reader import TaskReaderEnvService
from astuner.task_reader.hf_dataset_reader import TaskReaderHuggingFace
from astuner.task_reader.jsonl_reader import TaskReaderJsonl
from astuner.task_reader.task_reader_base import TaskReaderBase
from astuner.task_reader.tracing_reader import TracingReader


class RandomDummyGenerator(TaskReaderBase):
    def __init__(self, reader_config):
        super().__init__(reader_config)

    def _load_dataset_split(self, dataset_name: str, split: str) -> List[Task]:
        tasks = []
        # Save the current random state
        original_state = np.random.get_state()
        np.random.seed(42)
        random_number = [x for x in range(1000)]
        # shuffle
        np.random.shuffle(random_number)
        for idx in random_number:
            task = Task(
                main_query=f"[dummy task @ {idx}]",
                init_messages=[],
                task_id=str(idx),
                env_type="no_env",
                metadata={"random_number": idx},
            )
            tasks.append(task)
        # Restore the original random state
        np.random.set_state(original_state)
        return tasks

    def get_training_tasks(self) -> List[Task]:
        return self._load_dataset_split("dataset_name", "split")

    def get_validation_tasks(self) -> List[Task]:
        return self._load_dataset_split("dataset_name", "split")


class TaskReaderRouter(TaskReaderBase):
    def __init__(self, config):
        super().__init__(config)
        task_reader_type = config.astuner.task_reader.type
        reader_config = config.astuner.task_reader
        if task_reader_type == "env_service":
            self.task_reader = TaskReaderEnvService(reader_config)
        elif task_reader_type == "dataset_file":
            self.task_reader = TaskReaderJsonl(reader_config)
        elif task_reader_type == "huggingface_dat_repo":
            self.task_reader = TaskReaderHuggingFace(reader_config)
        elif self.task_reader == "tracing":
            self.task_reader = TracingReader(reader_config)
        elif self.task_reader_type == "data_generation":
            self.task_reader = TaskReaderDataGenerator(reader_config)
        elif task_reader_type == "random_dummy":
            self.task_reader = RandomDummyGenerator(reader_config)
        else:
            raise ValueError(f"Unsupported task reader type: {task_reader_type}")

    def get_training_tasks(self) -> List[Task]:
        return self.task_reader.get_training_tasks()

    def get_validation_tasks(self) -> List[Task]:
        return self.task_reader.get_validation_tasks()


class TaskReaderRouterV2(TaskReaderBase):
    def __init__(self, reader_type, reader_config):
        super().__init__(None)

        task_reader_type = reader_type
        if task_reader_type == "env_service":
            self.task_reader = TaskReaderEnvService(reader_config)
        elif task_reader_type == "dataset_file":
            self.task_reader = TaskReaderJsonl(reader_config)
        elif task_reader_type == "huggingface_dat_repo":
            self.task_reader = TaskReaderHuggingFace(reader_config)
        elif self.task_reader == "tracing":
            self.task_reader = TracingReader(reader_config)
        elif self.task_reader_type == "data_generation":
            self.task_reader = TaskReaderDataGenerator(reader_config)
        elif task_reader_type == "random_dummy":
            self.task_reader = RandomDummyGenerator(reader_config)
        else:
            raise ValueError(f"Unsupported task reader type: {task_reader_type}")

    def get_training_tasks(self) -> List[Task]:
        result = self.task_reader.get_training_tasks()
        np.random.shuffle(result)
        return result

    def get_validation_tasks(self) -> List[Task]:
        result = self.task_reader.get_validation_tasks()
        np.random.shuffle(result)
        return result


def task_to_standard_dataset(tasks: List[Task]) -> datasets.Dataset:
    """
    Convert a list of Task objects to a standard Hugging Face Dataset.

    Args:
        tasks (List[Task]): List of Task objects.

    Returns:
        datasets.Dataset: Hugging Face Dataset containing the tasks.
    """
    data = {
        "task_id": [],
        "main_query": [],
        "init_messages": [],
        "env_type": [],
        "metadata": [],
    }

    for task in tasks:
        data["task_id"].append(task.task_id)
        data["main_query"].append(task.main_query)
        data["init_messages"].append(task.init_messages)
        data["env_type"].append(task.env_type)
        data["metadata"].append(task.metadata)

    return datasets.Dataset.from_dict(data)
