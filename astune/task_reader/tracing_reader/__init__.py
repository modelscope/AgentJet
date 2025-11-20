

from typing import Any, List, Mapping, MutableMapping, Sequence, TypedDict

import json
import os
import random

from astune.schema.task import Task
from astune.task_reader.tracing_reader.filters.base import Filter
from astune.task_reader.tracing_reader.filters.factory import build_filters
from astune.task_reader.tracing_reader.filters.llm_evaluate_filter import (
    LlmEvaluateFilter,
)
from ..task_reader_base import TaskReaderBase
from .connector import LocalSqliteConnectorV1, PhoenixConnector



class Config(TypedDict):
    base_url: str
    train_output_path: str
    filters: List[Mapping[str, Any]]


class TracingReader(TaskReaderBase):
    config: Config

    def __init__(
        self,
        config,
        train_ratio: float = 0.7,
        split_seed: int = 42,
    ) -> None:
        super().__init__(config)
        # config patch
        print('*********', config, '**********')
        self.config = config.astune.tracing

        self._connector = LocalSqliteConnectorV1(self.config.get("base_url"))
        filters_config = self.config.get("filters")
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
        with open(path, "a") as f:
            for task in tasks:
                obj = task.model_dump()
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _apply_filters(self, tasks: List[Task]) -> List[Task]:
        filtered = tasks
        for flt in self._filters:
            filtered = flt.filter(filtered)
        return filtered

    def _init_tasks(self) -> None:
        output_path = self.config.get("train_output_path")

        tasks = self._connector.load_tasks_from_conversation()

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

    def get_training_tasks(self) -> List[Task]:
        return self._train_tasks

    def get_validation_tasks(self) -> List[Task]:
        return self._val_tasks
