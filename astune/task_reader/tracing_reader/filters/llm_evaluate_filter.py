from typing import Iterable, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from astune.schema.task import Task
from astune.task_rollout.dashscope_llm_bridge import (
    construct_alien_llm_chat_fn,
)

from ..fn import Fn
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


class LlmEvaluateFilter(Filter):
    def __init__(
        self,
        *,
        custom_rubrics: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        print_reason: bool = True,
        max_thread: int = 1,
    ) -> None:
        """Filter that evaluates the quality of tasks using LLM."""

        self._print_reason = print_reason
        self._max_thread = max_thread
        self.alien_llm_chat_fn = construct_alien_llm_chat_fn(
            alien_llm_model="qwen3-235b-a22b-instruct-2507",
            alien_llm_response_length=512,
        )
        self._fn = Fn(
            name="evaluate_quality",
            description=EVALUATE_PROMPT.format(custom_rubrics=custom_rubrics),
            alien_llm_chat_fn=self.alien_llm_chat_fn,
            input_schema={
                "query": "user query/task",
                "answer": "assistant answer",
            },
            output_schema={
                "reason": "judgment reason, briefly explain the reason",
                "result": "GOOD/BAD",
            },
            sampling_params={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    def filter(self, tasks: Iterable[Task]) -> List[Task]:
        tasks_list = list(tasks)
        kept: List[Task] = []

        if not tasks_list:
            return kept

        max_workers = max(self._max_thread, 1)

        def _evaluate(task: Task) -> dict:
            payload = {
                "query": task.main_query,
                "answer": task.metadata.get("answer", ""),
            }
            res = self._fn(payload)
            assert isinstance(res, dict)
            return res

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(_evaluate, task): task for task in tasks_list
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                res = future.result()
                if self._print_reason:
                    print(res["reason"])
                if res.get("result") == "GOOD":
                    kept.append(task)

        return kept
