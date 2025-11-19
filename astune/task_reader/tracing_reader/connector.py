import os
import sqlite3
import ast
import re
import requests
import json
import hashlib
from typing import List, Protocol
from datetime import datetime

from astune.schema.task import Task

class TracingConnector(Protocol):
    def load_tasks_from_conversation(self) -> List[Task]: ...


class PhoenixConnector:
    """
    PhoneixConnector is a class that connects to the Phoneix API.

    Args:
        base_url (str): The base URL of the Phoneix API.
        projects_limit (int): The maximum number of projects to load.
        spans_limit (int): The maximum number of spans to load from each project.

    Methods:
        load_spans(self, projects_limit: int = 100, spans_limit: int = 100) -> list:
            Load all spans from all projects.

        load_tasks_from_conversation(self) -> List[Task]:
            Load all tasks from the conversation spans.

    Attributes:
        _base_url (str): The base URL of the Phoneix API.
        _projects_limit (int): The maximum number of projects to load.
        _spans_limit (int): The maximum number of spans to load from each project.
    """

    def __init__(self, base_url: str, projects_limit: int = 100, spans_limit: int = 100) -> None:
        self._base_url = base_url.rstrip("/")
        self._projects_limit = projects_limit
        self._spans_limit = spans_limit

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
        self
    ) -> List[Task]:
        all_spans = self.load_spans(
            projects_limit=self._projects_limit, spans_limit=self._spans_limit)
        all_spans.sort(key=lambda x: datetime.fromisoformat(x["end_time"]))
        all_spans = list(
            filter(lambda x: x["name"].startswith("invoke_agent"), all_spans))

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


def parse_msg_line(line: str):
    """
    Extract role and content from Msg(...).
    """
    match = re.search(r"Msg\((.*)\)", line, re.DOTALL)
    if not match:
        return None

    inner = match.group(1)

    kv_pairs = []
    for item in re.findall(r"(\w+)=((?:'.*?'|None))", inner):
        key, val = item
        kv_pairs.append(f"'{key}': {val}")
    dict_like = "{" + ", ".join(kv_pairs) + "}"

    try:
        data = ast.literal_eval(dict_like)
    except Exception as e:
        print("解析失败:", e)
        return None

    role = data.get("role")
    content = data.get("content")
    return {"role": role, "content": content}


class LocalSqliteConnector:
    """
    A connector that loads tasks from a SQLite database file.

    Args:
        db_path (str): Path to the SQLite database file.

    Attributes:
        _db_path (str): Path to the SQLite database file.

    Methods:
        load_tasks_from_conversation (self) -> List[Task]:
            Load tasks from a conversation in the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        assert os.path.exists(
            self._db_path), f"DB file {self._db_path} does not exist"

    def load_tasks_from_conversation(self) -> List[Task]:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT attributes FROM span_table where name='ReActAgent.reply'").fetchall()

        qa = []
        for row in rows:
            js = json.loads(row[0])
            query = js['input']['kwargs']['msg']
            output = js['output'] if 'output' in js else None
            if query is not None and output is not None:
                query = parse_msg_line(query)
                output = parse_msg_line(output)
                if query is not None and output is not None:
                    if query['role']=='user' and output['role']=='assistant':
                        if query['content'] is not None and output['content'] is not None:
                            qa.append(
                                {"query": query['content'], "answer": output['content']})
        
        conn.close()

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


__all__ = ["LocalSqliteConnector", "PhoenixConnector"]