import json
import os
import random
from typing import Any, List, Mapping, TypedDict

from loguru import logger

from astuner.task_reader.tracing_reader.filters.base import Filter
from astuner.task_reader.tracing_reader.filters.factory import build_filters
from astuner.schema.task import Task

from ..task_reader_base import BaseTaskReader


class Config(TypedDict):
    base_url: str
    train_output_path: str
    filters: List[Mapping[str, Any]]


class TracingReader(BaseTaskReader):
    def __init__(
        self,
        reader_config,
        train_ratio: float = 0.7,
        split_seed: int = 42,
    ) -> None:
        from astuner.task_reader.tracing_reader.connector import LocalSqliteConnectorV1

        super().__init__(reader_config)
        # config patch
        # print("*********", config, "**********")
        self.reader_config = reader_config.feedback_tracing

        logger.info(
            f"reading tasks from {self.reader_config.get('base_url')}, #filter {len(self.reader_config.get('filters', []))}"
        )
        self._connector = LocalSqliteConnectorV1(self.reader_config.get("base_url"))
        filters_config = self.reader_config.get("filters")
        built_filters = build_filters(filters_config)
        self._filters: List[Filter] = built_filters

        self._train_ratio = train_ratio
        self._split_seed = split_seed

        self._train_tasks: List[Task] = []
        self._val_tasks: List[Task] = []

        self._init_tasks()

    def _load_existing_tasks(self, path: str) -> List[Task]:
        if not os.path.exists(path):
            return []
        tasks: List[Task] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tasks.append(Task(**obj))
        return tasks

    def _append_tasks(self, path: str, tasks: List[Task]) -> None:
        if not tasks:
            return
        mode = "a" if os.path.exists(path) else "w"
        with open(path, mode) as f:
            for task in tasks:
                obj = task.model_dump()
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _apply_filters(self, tasks: List[Task]) -> List[Task]:
        filtered = tasks
        for flt in self._filters:
            filtered = flt.filter_sync(filtered)
        return filtered

    def _init_tasks(self) -> None:
        output_path = self.reader_config.get("train_output_path")

        tasks = self._connector.load_tasks_from_conversation()
        logger.info(f"Loaded {len(tasks)} tasks from conversation")

        existing_tasks = self._load_existing_tasks(output_path)
        existing_hashes = {
            t.metadata.get("qa_hash")
            for t in existing_tasks
            if t.metadata.get("qa_hash") is not None
        }

        new_tasks = [
            t
            for t in tasks
            if t.metadata.get("qa_hash") is not None
            and t.metadata["qa_hash"] not in existing_hashes
        ]

        new_tasks_filtered = self._apply_filters(new_tasks)

        self._append_tasks(output_path, new_tasks_filtered)

        all_tasks: List[Task] = existing_tasks + new_tasks_filtered

        if not all_tasks:
            self._train_tasks = []
            self._val_tasks = []
            return

        shuffled_tasks = list(all_tasks)
        rnd = random.Random(self._split_seed)
        rnd.shuffle(shuffled_tasks)

        total = len(shuffled_tasks)
        train_size = int(total * self._train_ratio)

        if total == 1:
            train_size = 1
        else:
            if train_size <= 0:
                train_size = 1
            if train_size >= total:
                train_size = total - 1

        self._train_tasks = shuffled_tasks[:train_size]
        self._val_tasks = shuffled_tasks[train_size:]
        logger.info(f"Shuffled {total} tasks into {train_size} train and {total - train_size} val")

    def get_training_tasks(self) -> List[Task]:
        return self._train_tasks

    def get_validation_tasks(self) -> List[Task]:
        return self._val_tasks
