from typing import Iterable, List

from astune.schema.task import Task

from ..fn import Fn
from ..llm_client import DashScopeClient
from .base import Filter


EVALUATE_PROMPT = """你现在扮演一个**严格的 QA 质量审查员**。给你一条数据样本，包含一个“query”（用户提问/任务）和一个“answer”（助手回复）。请只依据文本本身进行静态评估，不要编造事实或外部检索。

一、判定目标
判定该「query–answer」是否为**高质量数据（GOOD）**，并给出分数与理由。若不满足标准，则标为**BAD**。

二、判断标准（任一命中即 BAD）
1. **缺失项**：query 为空、answer 为空、或两者皆空。
2. **非答复**：answer 只有“收到/明白/请提供更多信息/占位寒暄”，没有实质回答或行动结果。
3. **牛头不对马嘴**：answer 与 query 主题明显不相关。
4. **流程借口**：answer 主要描述“无法搜索/被限流/遇到验证码/换设备”等流程困难，却**没有**给出替代性信息、总结、或可执行的下一步。
5. **明显自相矛盾或逻辑不通**：同一回答内部互相打架（如先说能做又说不能做）。
6. **安全/合规红线**：含违法、仇恨、个人隐私泄露等明显不当内容。
7. **语言极不匹配**：query 的语言与 answer 完全不匹配，影响理解（如中文提问，answer 用不通顺的法语且与内容无关）。

三、特殊情况与其他标准
{custom_rubrics}

以上信息有任何一个命中，最终结果即为BAD，否则是GOOD。
"""


class LlmEvaluateFilter(Filter):
    def __init__(
        self,
        *,
        custom_rubrics: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        print_reason: bool = True,
    ) -> None:
        """Filter that evaluates the quality of tasks using LLM.
        """

        self._print_reason = print_reason
        self._llm = DashScopeClient()
        self._fn = Fn(
            name="evaluate_quality",
            description=EVALUATE_PROMPT.format(custom_rubrics=custom_rubrics),
            llm_client=self._llm,
            input_schema={
                "query": "用户提问/任务",
                "answer": "助手回复",
            },
            output_schema={
                "reason": "判断理由，简明说明理由",
                "result": "GOOD/BAD",
            },
            sampling_params={"temperature": temperature, "max_tokens": max_tokens},
        )

    def filter(self, tasks: Iterable[Task]) -> List[Task]:
        kept: List[Task] = []
        for task in tasks:
            payload = {
                "query": task.main_query,
                "answer": task.metadata.get("answer", ""),
            }
            res = self._fn(payload)
            assert isinstance(res, dict)
            if self._print_reason:
                print(res["reason"])
            if res.get("result") == "GOOD":
                kept.append(task)
        return kept
