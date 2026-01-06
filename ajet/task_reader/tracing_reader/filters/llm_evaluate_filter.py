import os
from typing import Iterable, List

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from pydantic import BaseModel, Field

from ajet.schema.task import Task
from ajet.task_rollout.dashscope_llm_bridge import create_external_llm_fn

from .base import Filter

EVALUATE_PROMPT = """You are now acting as a **strict QA quality reviewer**. You will be given a data sample containing a “query” (user question/task) and an “answer” (assistant reply). Evaluate it **only based on the text itself**, without inventing facts or performing external retrieval.

---

## 1. Evaluation Goal
Determine whether the given “query-answer” pair is **high-quality data (GOOD)** and provide a score and justification.
If it does not meet the criteria, label it as **BAD**.

---

## 2. BAD Criteria (if any are met → BAD)
1. **Missing elements**: The query is empty, the answer is empty, or both are empty.
2. **Non-answer**: The answer contains only acknowledgments such as “Received / OK / Please provide more information,” without substantive content or actionable results.
3. **Irrelevant**: The answer is clearly unrelated to the query.
4. **Process excuses**: The answer mainly describes process issues (“cannot search / rate-limited / captcha / try another device”), **without** providing alternative information, summaries, or next steps.
5. **Self-contradiction or illogical**: The answer contradicts itself or contains major logical inconsistencies.
6. **Safety or compliance violations**: Includes illegal content, hate speech, personal privacy leaks, or other clearly inappropriate material.
7. **Severe language mismatch**: The answer is in a completely different language from the query in a way that breaks comprehension (e.g., Chinese query but irrelevant and incoherent French reply).

---

## 3. Special Cases & Additional Rules
{custom_rubrics}

---

If **any** of the above conditions are triggered, the final result must be **BAD**. Otherwise, it is **GOOD**.
"""


class EvalResModel(BaseModel):
    reason: str = Field(
        description="judgment reason, briefly explain the reason",
    )
    result: str = Field(
        description="GOOD/BAD",
    )


class LlmEvaluateFilter(Filter):
    def __init__(
        self,
        *,
        custom_rubrics: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        print_reason: bool = True,
    ) -> None:
        """Filter that evaluates the quality of tasks using LLM."""

        self._print_reason = print_reason
        self.external_llm_fn = create_external_llm_fn(
            alien_llm_model="qwen3-235b-a22b-instruct-2507",
            alien_llm_response_length=512,
        )
        self._fn = ReActAgent(
            name="agent",
            sys_prompt=EVALUATE_PROMPT.format(custom_rubrics=custom_rubrics),
            model=DashScopeChatModel(
                "qwen3-235b-a22b-instruct-2507", os.environ["DASHSCOPE_API_KEY"]
            ),
            formatter=DashScopeMultiAgentFormatter(),
            max_iters=1,
        )

    async def filter(self, tasks: Iterable[Task]) -> List[Task]:
        kept: List[Task] = []
        for task in tasks:
            payload = "query: " + task.main_query + "\n" "answer: " + task.metadata.get(
                "answer", ""
            )

            res = await self._fn(
                Msg("user", content=payload, role="user"), structured_model=EvalResModel
            )
            assert isinstance(res, EvalResModel)
            if self._print_reason:
                print(res.reason)
            if res.result == "GOOD":
                kept.append(task)
        return kept
