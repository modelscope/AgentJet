from abc import ABC, abstractmethod
from typing import Iterable, List

from astuner.schema.task import Task


class Filter(ABC):
    @abstractmethod
    def filter(self, tasks: Iterable[Task]) -> List[Task]:
        """Filter a collection of Task objects and return the kept ones."""
        raise NotImplementedError
