# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderScore

try:
    from openjudge.models import OpenAIChatModel
except Exception:  # pragma: no cover
    from openjudge.models.openai_chat_model import OpenAIChatModel

from .prompt import TRACEABILITY_SYSTEM_PROMPT, TRACEABILITY_USER_PROMPT_TEMPLATE
from .json_utils import strict_load_json, validate_shape, coerce_to_messages_list, construct_traceability_prompt, count_digit_tokens


class TraceabilityRewardGrader(BaseGrader):
    """
    Traceability & Verifiability Reward (TVR)

    Input: traj (trajectory / record) - supports:
      - list[dict]
      - list[list[dict]]
      - dict with {"traj": ...} etc.

    Output: GraderScore(name="traceability", score in [0,1], reason includes stats + brief examples)
    """

    def __init__(
        self,
        model: Optional[OpenAIChatModel] = None,
        name: str = "traceability",
        temperature: float = 0.0,
        max_tokens: int = 2200,
    ) -> None:
        super().__init__(name=name)
        self.model = model or OpenAIChatModel(
            model_name="gpt-4.1-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

    async def aevaluate(self, traj: Any, **kwargs: Any) -> GraderScore:
        messages = coerce_to_messages_list(traj)

        user_prompt, report_plain = construct_traceability_prompt(
            messages,
            TRACEABILITY_USER_PROMPT_TEMPLATE,
        )

        judge_messages = [
            {"role": "system", "content": TRACEABILITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        resp = await self.model.achat(judge_messages)
        text = resp.get("content", "")

        try:
            obj = strict_load_json(text)
            norm = validate_shape(obj)
            score = self._compute_score(norm, report_plain)
            reason = self._build_reason(norm, report_plain, score)
            return GraderScore(name=self.name, score=score, reason=reason)
        except Exception as e:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=f"TVR judge output invalid: {e}",
            )

    def _compute_score(self, norm: Dict[str, Any], report_plain: str) -> float:
        stats = norm["stats"]
        total = max(1, int(stats.get("total_claims", 0)))

        supported = int(stats.get("supported", 0))
        contradicted = int(stats.get("contradicted", 0))
        no_evidence = int(stats.get("no_evidence", 0))
        speculative_ok = int(stats.get("speculative_ok", 0))
        unclear = int(stats.get("unclear", 0))

        # Positive contribution
        pos = supported + 0.6 * speculative_ok + 0.3 * unclear
        # Negative contribution (contradiction is harsh)
        neg = 1.0 * contradicted + 0.8 * no_evidence

        base = (pos - neg) / total  # can be negative
        base = max(0.0, min(1.0, base))

        # Coverage factor (deterministic) based on digits/dates in report body
        real_digit_tokens = count_digit_tokens(report_plain)
        expected_min_claims = min(25, max(6, real_digit_tokens // 2))
        claim_count = int(stats.get("total_claims", total))

        selection_factor = min(1.0, claim_count / expected_min_claims) if expected_min_claims > 0 else 1.0

        # If the judge reports digit coverage, blend it in (but keep deterministic as the main)
        reported_total_digits = int(stats.get("report_digit_tokens", 0))
        reported_covered_digits = int(stats.get("covered_digit_tokens", 0))
        if reported_total_digits > 0:
            reported_cov = min(1.0, max(0.0, reported_covered_digits / reported_total_digits))
        else:
            reported_cov = 1.0

        cov_factor = 0.7 + 0.3 * reported_cov  # [0.7, 1.0]

        score = base * selection_factor * cov_factor
        score = max(0.0, min(1.0, score))
        return float(score)

    def _build_reason(self, norm: Dict[str, Any], report_plain: str, score: float) -> str:
        stats = norm["stats"]
        ex = norm.get("examples", {})
        best = ex.get("best_supported", [])
        worst = ex.get("worst_failed", [])

        real_digit_tokens = count_digit_tokens(report_plain)

        parts = []
        parts.append(
            f"score={score:.3f}; "
            f"claims={stats['total_claims']}; "
            f"supported={stats['supported']}; "
            f"spec_ok={stats['speculative_ok']}; "
            f"unclear={stats['unclear']}; "
            f"no_ev={stats['no_evidence']}; "
            f"contradicted={stats['contradicted']}; "
            f"report_digits≈{real_digit_tokens}"
        )

        if best:
            parts.append(f"best_supported={best[:1]}")
        if worst:
            parts.append(f"worst_failed={worst[:1]}")
        return " | ".join(parts)
