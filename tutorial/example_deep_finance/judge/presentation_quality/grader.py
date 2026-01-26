from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderScore

# import path 兼容两种写法（文档里两种都出现过）
try:
    from openjudge.models import OpenAIChatModel
except Exception:  # pragma: no cover
    from openjudge.models.openai_chat_model import OpenAIChatModel

from .prompt import (
    QUALITY_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    ALL_KEYS,
    A_KEYS,
    B_KEYS,
    C_KEYS,
)
from .json_utils import strict_load_json, validate_shape, get_bool_pass, get_note


class PresentationQualityGrader(BaseGrader):
    """
    - 输入：report_content（研究报告文本）
    - 输出：GraderScore(name, score, reason)
    - score：8项(pass)均分，范围[0,1]
    - determinism：建议用 temperature=0 + disable thinking 等（见 create_default_model）
    - 解析失败：score=0，并在 reason 显示报错
    """

    def __init__(
        self,
        model: OpenAIChatModel,
        name: str = "presentation_quality",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.model = model

    @staticmethod
    def create_default_model(
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        deterministic: bool = True,
        enable_thinking: bool = False,
        seed: int = 0,
    ) -> OpenAIChatModel:
        """
        你也可以不调用这个工厂，自己在外面 new OpenAIChatModel。
        QuickStart 文档确认 OpenAIChatModel 会从 OPENAI_API_KEY/OPENAI_BASE_URL 读取。
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")

        extra_body: Dict[str, Any] = {}
        if deterministic:
            # OpenAI兼容接口常见字段；DashScope/Qwen 常用 enable_thinking
            extra_body.update(
                {
                    "temperature": 0,
                    "top_p": 1,
                    "seed": seed,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                }
            )
        if enable_thinking is False:
            extra_body["enable_thinking"] = False

        kwargs: Dict[str, Any] = {"model": model_name}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if extra_body:
            kwargs["extra_body"] = extra_body

        return OpenAIChatModel(**kwargs)

    async def aevaluate(
        self,
        report_content: str,
        user_query: str | None = None,
        **_: Any,
    ) -> GraderScore:
        """
        入口：直接喂 report_content（研究报告文本）
        - user_query 可选：用于填充 prompt；不提供则用 "(unknown)"
        """
        report = (report_content or "").strip()
        if not report:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="BadInput: empty report_content",
            )

        uq = (user_query or "").strip() or "(unknown)"

        user_content = USER_PROMPT_TEMPLATE.format(
            user_query=uq,
            report_content=report,
        )
        messages = [
            {"role": "system", "content": QUALITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # 核心：OpenJudge 的 OpenAIChatModel 支持 await model.achat([...])，并返回 .content
        try:
            resp = await self.model.achat(messages)
            raw_text = getattr(resp, "content", None)
            if raw_text is None:
                raw_text = str(resp)
        except Exception as e:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=f"ModelCallError: {type(e).__name__}: {e}",
            )

        obj, jerr = strict_load_json(str(raw_text))
        if obj is None:
            snippet = str(raw_text)[:200].replace("\n", " ")
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=f"ParseError: {jerr}; raw[:200]={snippet}",
            )

        obj, serr = validate_shape(obj)
        if obj is None:
            snippet = str(raw_text)[:200].replace("\n", " ")
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=f"SchemaError: {serr}; raw[:200]={snippet}",
            )

        score, reason = self._score_and_reason(obj)
        return GraderScore(name=self.name, score=score, reason=reason)

    def _score_and_reason(self, obj: Dict[str, Any]) -> Tuple[float, str]:
        scan = obj["scan"]
        structuring = obj["structuring"]
        editorial = obj["editorial"]
        top_fixes = obj.get("top_fixes", [])

        # 8项均分（强确定性：完全由Python算）
        pass_map: Dict[str, bool] = {}
        note_map: Dict[str, str] = {}

        def take(section: Dict[str, Any], key: str):
            item = section.get(key)
            pass_map[key] = get_bool_pass(item)
            note_map[key] = get_note(item)

        for k in A_KEYS:
            take(scan, k)
        for k in B_KEYS:
            take(structuring, k)
        for k in C_KEYS:
            take(editorial, k)

        passed = sum(1 for k in ALL_KEYS if pass_map.get(k) is True)
        total = len(ALL_KEYS)  # 8
        score = passed / float(total)

        # reason：不加额外字段，只给紧凑总结
        failed_items = [k for k in ALL_KEYS if not pass_map.get(k, False)]
        failed_str = ", ".join(f"{k}({note_map.get(k,'')})" for k in failed_items[:4])
        fixes_str = " | ".join(str(x) for x in (top_fixes or [])[:3])

        parts: List[str] = []
        parts.append(f"Pass {passed}/{total}")
        if failed_items:
            parts.append(f"Fail: {failed_str}")
        if fixes_str:
            parts.append(f"TopFixes: {fixes_str}")

        reason = " ; ".join(parts)
        return round(score, 6), reason[:800]
