import json
from pathlib import Path
from typing import List

import pytest

from astune.schema.task import Task
from astune.task_reader.tracing_reader import TracingReader


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


@pytest.fixture
def config(tmp_path: Path) -> dict:
    return {
        "base_url": "http://example.com",
        "train_output_path": str(tmp_path / "tasks.jsonl"),
    }


def test_get_training_tasks_new_file(config: dict):
    # prepare tasks returned from connector
    t1 = _make_task("q1", "a1", "h1")
    t2 = _make_task("q2", "a2", "h2")
    tasks = [t1, t2]

    connector = DummyConnector(tasks)
    flt = DummyFilter(kept=tasks)

    reader = TracingReader(config)  # type: ignore
    reader._connector = connector  # type: ignore[attr-defined]
    reader._filters = [flt]  # type: ignore[attr-defined]

    result = reader.get_training_tasks()

    # connector should be called once
    assert connector.called == 1

    # filter should receive all new tasks
    assert flt.last_input == tasks

    # returned tasks should be exactly the filtered ones
    assert result == tasks

    # file should be created with one json per line
    out_path = Path(config["train_output_path"])
    assert out_path.exists()
    with out_path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert {obj["metadata"]["qa_hash"] for obj in lines} == {"h1", "h2"}


def test_get_training_tasks_dedup_and_missing_hash_ignored(config: dict):
    out_path = Path(config["train_output_path"])

    # existing task with hash h1
    existing = _make_task("q_exist", "a_exist", "h1")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(existing.model_dump(), ensure_ascii=False) + "\n")

    # connector returns: duplicate (h1), new (h2), and one without qa_hash
    dup = _make_task("q_dup", "a_dup", "h1")
    new = _make_task("q_new", "a_new", "h2")
    no_hash = _make_task("q_nohash", "a_nohash", None)
    connector_tasks = [dup, new, no_hash]

    # filter will keep everything it receives so we can test input to filter
    flt = DummyFilter(kept=[new])
    connector = DummyConnector(connector_tasks)

    reader = TracingReader(config)  # type: ignore
    reader._connector = connector  # type: ignore[attr-defined]
    reader._filters = [flt]  # type: ignore[attr-defined]

    result = reader.get_training_tasks()

    # existing task plus new filtered task should be returned
    assert len(result) == 2
    assert existing in result
    assert new in result

    # filter should see only new tasks with non-duplicate hashes => [new]
    assert flt.last_input == [new]

    # output file should now contain existing + new filtered
    with out_path.open("r", encoding="utf-8") as f:
        objs = [json.loads(line) for line in f if line.strip()]

    hashes = [obj["metadata"].get("qa_hash") for obj in objs]
    assert "h1" in hashes
    assert "h2" in hashes
    # no record without hash should be written
    assert None not in hashes
