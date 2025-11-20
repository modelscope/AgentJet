import json
from pathlib import Path
from typing import List

import pytest

from astune.task_reader.tracing_reader import TracingReader
from astune.schema.task import Task

@pytest.fixture
def config(tmp_path: Path) -> dict:
    return {
        "astune":{
            "base_url": ".trash/database.sqlite",
            "train_output_path": str(tmp_path / "tasks.jsonl"),
        }
    }


reader = TracingReader(config) # type: ignore
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
