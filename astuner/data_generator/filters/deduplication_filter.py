import os
import shutil
from typing import Iterable, List

from astuner.schema.task import Task
from astuner.utils.embedding_client import EmbeddingClient

from .base import Filter


class DeduplicationFilter(Filter):
    def __init__(
        self,
        similarity_threshold: float,
        db_path: str,
        model: str,
        api_key: str | None,
        base_url: str,
    ):
        # remove old db
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        self._client = EmbeddingClient(
            similarity_threshold=similarity_threshold,
            base_url=base_url,
            api_key=api_key,
            model=model,
            chroma_db_path=db_path,
        )

        self._similarity_threshold = similarity_threshold
        self._db_path = db_path

    def filter(self, tasks: Iterable[Task]) -> List[Task]:
        res = []
        for task in tasks:
            similar = self._client.find_top_k_by_text(task.main_query, k=1)
            if len(similar) != 0 and similar[0][1] >= self._similarity_threshold:
                continue
            res.append(task)
            self._client.add(task.main_query, hash(task.main_query))
            

        return res
