from typing import List

import requests
import json
import hashlib
from datetime import datetime

from astune.schema.task import Task


class PhoneixConnector:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def _get(self, path: str, **params):
        url = f"{self._base_url}{path}"
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()["data"]

    def load_spans(self, projects_limit: int = 100, spans_limit: int = 100) -> list:
        projects = self._get(
            "/v1/projects",
            limit=projects_limit,
            include_experiment_projects="false",
        )

        all_spans: list = []
        for project in projects:
            pid = project["id"]
            spans = self._get(f"/v1/projects/{pid}/spans", limit=spans_limit)
            all_spans.extend(spans)
        return all_spans

    def load_tasks_from_conversation(
        self, projects_limit: int = 100, spans_limit: int = 100
    ) -> List[Task]:
        all_spans = self.load_spans(projects_limit=projects_limit, spans_limit=spans_limit)
        all_spans.sort(key=lambda x: datetime.fromisoformat(x["end_time"]))
        all_spans = list(filter(lambda x: x["name"].startswith("invoke_agent"), all_spans))

        qa: list = []
        for span in all_spans:
            inp = json.loads(span["attributes"]["gen_ai.input.messages"])
            out = json.loads(span["attributes"]["gen_ai.output.messages"])
            if "parts" in inp and "parts" in out:
                qa.append({
                    "query": inp["parts"][0]["content"],
                    "answer": out["parts"][0]["content"],
                })

        tasks: List[Task] = []
        for item in qa:
            raw = (item["query"] or "") + "\n" + (item["answer"] or "")
            qa_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            task = Task(
                main_query=item["query"],
                task_id="no_id",
                env_type="no_env",
                metadata={
                    "answer": item["answer"],
                    "qa_hash": qa_hash,
                },
            )
            tasks.append(task)
        return tasks


