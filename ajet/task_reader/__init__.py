from typing import List

import datasets
import numpy as np

from ajet.schema.task import Task
from ajet.task_reader.data_generator_reader import DataGeneratorTaskReader
from ajet.task_reader.env_service_reader import EnvServiceTaskReader
from ajet.task_reader.hf_dataset_reader import HuggingFaceTaskReader
from ajet.task_reader.jsonl_reader import JsonlTaskReader
from ajet.task_reader.task_reader_base import BaseTaskReader
from ajet.task_reader.tracing_reader import TracingReader


class RandomDummyTaskReader(BaseTaskReader):
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


class RouterTaskReader(BaseTaskReader):
    def __init__(self, reader_type, reader_config):
        super().__init__(None)

        task_reader_type = reader_type
        if task_reader_type == "env_service":
            self.task_reader = EnvServiceTaskReader(reader_config)
        elif task_reader_type == "jsonl_dataset_file":
            self.task_reader = JsonlTaskReader(reader_config)
        elif task_reader_type == "huggingface_dat_repo":
            self.task_reader = HuggingFaceTaskReader(reader_config)
        elif task_reader_type == "tracing":
            self.task_reader = TracingReader(reader_config)
        elif task_reader_type == "data_generation":
            self.task_reader = DataGeneratorTaskReader(reader_config)
        elif task_reader_type == "random_dummy":
            self.task_reader = RandomDummyTaskReader(reader_config)
        elif task_reader_type == "finworld":
            # FinWorld 专用: 数据从 JSON 文件加载并组装 init_messages，工具调用走 env_service
            from tutorial.example_finworld.finworld_reader import FinworldReader
            self.task_reader = FinworldReader(reader_config)
        else:
            raise ValueError(f"Unsupported task reader type: {task_reader_type}")

    def get_training_tasks(self) -> List[Task]:
        result = self.task_reader.get_training_tasks()
        np.random.shuffle(result)  # type: ignore
        return result

    def get_validation_tasks(self) -> List[Task]:
        result = self.task_reader.get_validation_tasks()
        np.random.shuffle(result)  # type: ignore
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


def dict_to_ajet_task(task_dict: dict) -> Task:
    """
    Convert a dictionary to a Task object.

    Args:
        task_dict (dict): Dictionary containing task fields.

    Returns:
        Task: Task object created from the dictionary.
    """
    for vip_key in ["main_query", "task_id", "env_type", "metadata", "init_messages"]:
        if vip_key not in task_dict:
            raise ValueError(f"Key {vip_key} not found in task.raw_task")

    return Task(
        main_query=task_dict.get("main_query", ""),
        init_messages=task_dict.get("init_messages", []),
        task_id=task_dict.get("task_id", ""),
        env_type=task_dict.get("env_type", ""),
        metadata=task_dict.get("metadata", {}),
    )
