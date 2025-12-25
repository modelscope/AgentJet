import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Iterable, List

from agentscope_tuner.schema.task import Task


class Filter(ABC):
    @abstractmethod
    async def filter(self, tasks: Iterable[Task]) -> List[Task]:
        """Filter a collection of Task objects and return the kept ones."""
        raise NotImplementedError

    def filter_sync(self, tasks: Iterable[Task]) -> List[Task]:
        """This is a temp fix for async filter being called in a sync context."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.filter(tasks))

        res_holder: dict[str, object] = {}
        err_holder: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                res_holder["res"] = asyncio.run(self.filter(tasks))
            except BaseException as e:
                err_holder["err"] = e

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()

        if "err" in err_holder:
            raise err_holder["err"]

        res = res_holder.get("res")
        assert isinstance(res, list)
        return res
