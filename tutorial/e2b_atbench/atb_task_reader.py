# -*- coding: utf-8 -*-
"""ATB v3 任务读取器: 遍历 coding-agent-material 的 Clean_Tasks 目录树.

目录布局: <Clean_Tasks>/<family>/<instance_result>/, 每个 instance 目录内含
solver 任务文件 + judge 材料 (被 generate_claudecode._run_claudecode_in_sandbox
直接消费). 每个实例产出一个 ajet.schema.task.Task, metadata["task_dir"] = 实例
宿主机绝对路径.
"""

import os
from typing import Iterator, Optional

from ajet.schema.task import Task

# 判定一个子目录是否为合法任务实例的标记 (任一存在即可)
_INSTANCE_MARKERS = ("solver", "judge", "instruction.md", "task.toml", "environment")


class AtbDirTaskReader:
    def __init__(self, clean_tasks_root: str, limit: Optional[int] = None):
        self.root = clean_tasks_root
        self.limit = limit

    def _read_instruction(self, inst_dir: str) -> str:
        for cand in ("instruction.md", "solver/instruction.md"):
            p = os.path.join(inst_dir, cand)
            if os.path.isfile(p):
                try:
                    with open(p, encoding="utf-8", errors="ignore") as f:
                        return f.read()
                except Exception:
                    pass
        return ""

    @staticmethod
    def _is_instance(d: str) -> bool:
        return any(os.path.exists(os.path.join(d, m)) for m in _INSTANCE_MARKERS)

    def _make_task(self, inst_dir: str, name: str) -> Task:
        return Task(
            task_id=name,
            main_query=self._read_instruction(inst_dir) or name,
            metadata={"task_dir": os.path.abspath(inst_dir), "task_name": name},
        )

    def generate_training_tasks(self) -> Iterator[Task]:
        yielded = 0
        if not os.path.isdir(self.root):
            return
        for family in sorted(os.listdir(self.root)):
            fam_dir = os.path.join(self.root, family)
            if not os.path.isdir(fam_dir):
                continue
            if self._is_instance(fam_dir):
                yield self._make_task(fam_dir, family)
                yielded += 1
                if self.limit and yielded >= self.limit:
                    return
                continue
            for inst in sorted(os.listdir(fam_dir)):
                inst_dir = os.path.join(fam_dir, inst)
                if not os.path.isdir(inst_dir) or not self._is_instance(inst_dir):
                    continue
                yield self._make_task(inst_dir, inst)
                yielded += 1
                if self.limit and yielded >= self.limit:
                    return
