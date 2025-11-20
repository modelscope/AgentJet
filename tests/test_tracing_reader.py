import pytest
from pathlib import Path
from typing import List
from astune.task_reader.tracing_reader import TracingReader
from astune.schema.task import Task


class DummyConnector:
    def __init__(self, tasks: List[Task]):
        self._tasks = tasks
        self.called = 0

    def load_tasks_from_conversation(self, projects_limit: int = 100, spans_limit: int = 100) -> List[Task]:
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


@pytest.fixture
def config(tmp_path: Path) -> dict:
    from astune.utils.config_utils import read_astune_config
    return read_astune_config('launcher/math_agent/git-math-agentscope.yaml')   # type: ignore



def test_get_training_tasks_new_file(config: dict):
    from types import SimpleNamespace
    from astune.task_reader.tracing_reader import TracingReader, Config
    from astune.schema.task import Task
    from dotenv import load_dotenv; load_dotenv()

    # t=SimpleNamespace()
    # t.astune=SimpleNamespace()

    tr=TracingReader(config)
    print(tr.get_training_tasks())

