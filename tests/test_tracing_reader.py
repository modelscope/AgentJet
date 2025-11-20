import pytest
from pathlib import Path
from typing import List
from astune.task_reader.tracing_reader import TracingReader
from astune.schema.task import Task


class DummyConnector:
    def __init__(self, tasks: List[Task]):
        self._tasks = tasks
        self.called = 0

    def load_tasks_from_conversation(
        self, projects_limit: int = 100, spans_limit: int = 100
    ) -> List[Task]:
        self.called += 1
        return self._tasks


class DummyFilter:
    def __init__(self, kept: List[Task]):
        self._kept = kept
        self.last_input: List[Task] | None = None

    def filter(self, tasks: List[Task]) -> List[Task]:
        self.last_input = list(tasks)
        return self._kept


def _make_task(query: str, answer: str, qa_hash: str | None) -> Task:
    metadata = {"answer": answer}
    if qa_hash is not None:
        metadata["qa_hash"] = qa_hash
    return Task(
        main_query=query,
        task_id="tid",
        env_type="env",
        metadata=metadata,
    )


# @pytest.fixture
# def config(tmp_path: Path) -> dict:
#     from astune.utils.config_utils import read_astune_config
#     return read_astune_config('tutorial/example_math_agent/math_agent.yaml')   # type: ignore


# def test_get_training_tasks_new_file(config: dict):
#     from types import SimpleNamespace
#     from astune.task_reader.tracing_reader import TracingReader, Config
#     from astune.schema.task import Task
#     from dotenv import load_dotenv; load_dotenv()

#     # t=SimpleNamespace()
#     # t.astune=SimpleNamespace()

#     tr=TracingReader(config)
#     print(tr.get_training_tasks())


# prepare tests/database.sqlite from agentscope first

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from astune.task_reader.tracing_reader import TracingReader
from astune.schema.task import Task


@pytest.fixture
def config(tmp_path: Path) -> SimpleNamespace:
    a = SimpleNamespace()
    a.astune = SimpleNamespace()
    a.astune.tracing = {
        "base_url": "./.trash/database.sqlite",
        "train_output_path": str(tmp_path / "tasks.jsonl"),
        "filters": [],
    }
    return a


@pytest.fixture
def config_with_filter(tmp_path: Path) -> SimpleNamespace:
    a = SimpleNamespace()
    a.astune = SimpleNamespace()
    a.astune.tracing = {
        "base_url": "./.trash/database.sqlite",
        "train_output_path": str(tmp_path / "tasks.jsonl"),
        "filters": [
            {
                "type": "llm_evaluate",
                "enabled": True,
                "params": {
                    "custom_rubrics": "If the answer claims that it has written the output to a file, consider it an invalid response.",
                    "temperature": 0.5,
                    "max_tokens": 2048,
                    "print_reason": False,
                },
            }
        ],
    }
    return a


def test_get_training_tasks_new_file(config: SimpleNamespace):
    from dotenv import load_dotenv

    load_dotenv()
    # prepare tasks returned from connector
    reader = TracingReader(config, train_ratio=0.7)  # type: ignore

    result = reader.get_training_tasks()
    # the number of tasks in tests/database.sqlite
    assert len(result) == int(7 * 0.7)

    # file should be created with one json per line
    out_path = Path(config.astune.tracing["train_output_path"])
    assert out_path.exists()
    with out_path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 7  # the number of tasks in tests/database.sqlite


def test_get_training_tasks_with_filter(
    config_with_filter: SimpleNamespace, config: SimpleNamespace
):
    from dotenv import load_dotenv

    load_dotenv()
    reader = TracingReader(config_with_filter)  # type: ignore

    result = reader.get_training_tasks()
    assert len(result) < int(7 * 0.7)

    reader_full = TracingReader(config)
    result_full = reader_full.get_training_tasks()

    # find the diff
    delta = []
    for task in result_full:
        if task not in result:
            delta.append(task)

    assert len(delta) > 1
    print("these tasks are filtered:", delta)
