"""
CGCV JSON Utilities
JSON 解析和验证工具
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

class ClaimStatus(str, Enum):
    """断言验证状态枚举"""
    VERIFIED = "verified"
    CITATION_MISSING = "citation_missing"
    CITATION_BROKEN = "citation_broken"
    SUBJECT_MISALIGN = "subject_misalign"
    PREDICATE_MISALIGN = "predicate_misalign"
    OBJECT_MISALIGN = "object_misalign"
    QUALIFIER_MISALIGN = "qualifier_misalign"


# 所有有效的 status 值
VALID_STATUSES = {s.value for s in ClaimStatus}

# JSON 提取正则
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# =============================================================================
# JSON Repair Helper
# =============================================================================

def _repair_json(js: str) -> str:
    """
    尝试修复常见的JSON格式错误
    1. 修复字符串中未转义的换行符
    2. 修复trailing comma
    3. 修复不完整的JSON（截断）
    """
    # 1. 替换字符串值中的未转义换行符
    def escape_newlines_in_strings(s: str) -> str:
        result = []
        in_string = False
        escape_next = False
        i = 0
        while i < len(s):
            c = s[i]
            if escape_next:
                result.append(c)
                escape_next = False
            elif c == '\\':
                result.append(c)
                escape_next = True
            elif c == '"':
                result.append(c)
                in_string = not in_string
            elif in_string and c == '\n':
                result.append('\\n')
            elif in_string and c == '\r':
                result.append('\\r')
            elif in_string and c == '\t':
                result.append('\\t')
            else:
                result.append(c)
            i += 1
        return ''.join(result)

    js = escape_newlines_in_strings(js)

    # 2. 移除trailing comma: ",}" -> "}" 和 ",]" -> "]"
    js = re.sub(r',\s*}', '}', js)
    js = re.sub(r',\s*]', ']', js)

    # 3. 尝试修复截断的JSON - 补全缺失的括号
    open_braces = js.count('{')
    close_braces = js.count('}')
    open_brackets = js.count('[')
    close_brackets = js.count(']')

    if open_braces > close_braces:
        # 先关闭可能未闭合的字符串
        in_string = False
        escape_next = False
        for c in js:
            if escape_next:
                escape_next = False
            elif c == '\\':
                escape_next = True
            elif c == '"':
                in_string = not in_string
        if in_string:
            js += '"'

        # 补全缺失的括号
        js += ']' * (open_brackets - close_brackets)
        js += '}' * (open_braces - close_braces)

    return js


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClaimVerification:
    """单个断言的验证结果"""
    subject: str
    predicate: str
    object: str
    qualifier: str
    citation: Optional[str]
    status: str
    source_id: Optional[str]
    note: str

    def is_verified(self) -> bool:
        return self.status == ClaimStatus.VERIFIED.value

    def is_citation_issue(self) -> bool:
        return self.status in {
            ClaimStatus.CITATION_MISSING.value,
            ClaimStatus.CITATION_BROKEN.value
        }

    def is_alignment_issue(self) -> bool:
        return self.status in {
            ClaimStatus.SUBJECT_MISALIGN.value,
            ClaimStatus.PREDICATE_MISALIGN.value,
            ClaimStatus.OBJECT_MISALIGN.value,
            ClaimStatus.QUALIFIER_MISALIGN.value
        }


@dataclass
class CGCVResult:
    """CGCV 验证结果汇总"""
    claims: List[ClaimVerification]
    total: int
    verified: int
    citation_missing: int
    citation_broken: int
    alignment_issues: int

    @property
    def score(self) -> float:
        """计算验证通过率"""
        if self.total == 0:
            return 0.0
        return self.verified / self.total

    def get_summary(self) -> Dict[str, int]:
        """获取统计摘要"""
        return {
            "total": self.total,
            "verified": self.verified,
            "citation_missing": self.citation_missing,
            "citation_broken": self.citation_broken,
            "alignment_issues": self.alignment_issues
        }


# =============================================================================
# JSON Parsing Functions
# =============================================================================

def extract_first_json_object(text: str) -> Optional[str]:
    """
    从文本中提取第一个 JSON 对象

    Args:
        text: 原始文本

    Returns:
        JSON 字符串，如果未找到返回 None
    """
    if not text:
        return None

    # 先尝试找 ```json ... ``` 代码块
    json_block_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if json_block_match:
        return json_block_match.group(1).strip()

    # 再尝试找第一个 {...}
    m = _JSON_RE.search(text.strip())
    if not m:
        return None
    return m.group(0)


def strict_load_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    严格解析 JSON（带容错修复）

    Args:
        text: 原始文本

    Returns:
        (解析结果, 错误信息) 元组
    """
    js = extract_first_json_object(text)
    if js is None:
        return None, "No JSON object found in model output"

    # 第一次尝试：直接解析
    try:
        obj = json.loads(js)
        if not isinstance(obj, dict):
            return None, f"Top-level JSON is not an object: {type(obj).__name__}"
        return obj, None
    except json.JSONDecodeError:
        pass  # 继续尝试修复

    # 第二次尝试：修复后解析
    try:
        repaired = _repair_json(js)
        obj = json.loads(repaired)
        if not isinstance(obj, dict):
            return None, f"Top-level JSON is not an object: {type(obj).__name__}"
        return obj, None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def validate_cgcv_schema(obj: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    验证 CGCV JSON 结构

    期望格式:
    {
      "claims": [
        {
          "subject": str,
          "predicate": str,
          "object": str,
          "qualifier": str,
          "citation": str | null,
          "status": str (one of VALID_STATUSES),
          "source_id": str | null,
          "note": str
        }
      ]
    }

    Args:
        obj: JSON 对象

    Returns:
        (规范化后的对象, 错误信息) 元组
    """
    # claims 必须存在且为 list
    if "claims" not in obj:
        return None, "Missing field: claims"

    claims = obj["claims"]
    if not isinstance(claims, list):
        return None, f"Field 'claims' must be list, got {type(claims).__name__}"

    # 验证并规范化每个 claim
    normalized_claims = []
    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue  # 跳过非字典项

        # 提取并规范化字段
        normalized = {
            "subject": str(claim.get("subject", "未明确"))[:200],
            "predicate": str(claim.get("predicate", "未明确"))[:200],
            "object": str(claim.get("object", "未明确"))[:500],
            "qualifier": str(claim.get("qualifier", "未明确"))[:200],
            "citation": claim.get("citation"),
            "status": str(claim.get("status", "")).lower(),
            "source_id": claim.get("source_id"),
            "note": str(claim.get("note", ""))[:500]
        }

        # 规范化 citation
        if normalized["citation"] is not None:
            normalized["citation"] = str(normalized["citation"])
            if normalized["citation"].lower() in ("null", "none", ""):
                normalized["citation"] = None

        # 规范化 source_id
        if normalized["source_id"] is not None:
            normalized["source_id"] = str(normalized["source_id"])
            if normalized["source_id"].lower() in ("null", "none", ""):
                normalized["source_id"] = None

        # 验证 status
        if normalized["status"] not in VALID_STATUSES:
            # 尝试模糊匹配
            status_lower: str = normalized["status"]
            matched = False
            for valid_status in VALID_STATUSES:
                if valid_status in status_lower or status_lower in valid_status:
                    normalized["status"] = valid_status
                    matched = True
                    break
            if not matched:
                # 默认标记为 citation_missing
                normalized["status"] = ClaimStatus.CITATION_MISSING.value

        normalized_claims.append(normalized)

    obj["claims"] = normalized_claims
    return obj, None


def parse_cgcv_result(obj: Dict[str, Any]) -> CGCVResult:
    """
    解析 CGCV 结果为结构化对象

    Args:
        obj: 经过 validate_cgcv_schema 验证的 JSON 对象

    Returns:
        CGCVResult 对象
    """
    claims = []
    verified_count = 0
    citation_missing_count = 0
    citation_broken_count = 0
    alignment_issues_count = 0

    for claim_dict in obj.get("claims", []):
        claim = ClaimVerification(
            subject=claim_dict.get("subject", ""),
            predicate=claim_dict.get("predicate", ""),
            object=claim_dict.get("object", ""),
            qualifier=claim_dict.get("qualifier", ""),
            citation=claim_dict.get("citation"),
            status=claim_dict.get("status", ""),
            source_id=claim_dict.get("source_id"),
            note=claim_dict.get("note", "")
        )
        claims.append(claim)

        # 统计
        if claim.is_verified():
            verified_count += 1
        elif claim.status == ClaimStatus.CITATION_MISSING.value:
            citation_missing_count += 1
        elif claim.status == ClaimStatus.CITATION_BROKEN.value:
            citation_broken_count += 1
        elif claim.is_alignment_issue():
            alignment_issues_count += 1

    return CGCVResult(
        claims=claims,
        total=len(claims),
        verified=verified_count,
        citation_missing=citation_missing_count,
        citation_broken=citation_broken_count,
        alignment_issues=alignment_issues_count
    )


# =============================================================================
# Trajectory 处理辅助函数
# =============================================================================

def _extract_text_content(content) -> str:
    """统一提取纯文本内容"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                out.append(item.get("text", ""))
            elif isinstance(item, str):
                out.append(item)
        return "\n".join(out)
    return str(content)


def _strip_think(text: str) -> str:
    """去除 <think>...</think> 标签"""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


def _strip_markdown_fences(text: str) -> str:
    """
    清理 markdown 代码块标记
    - 移除开头的 ```markdown / ```md / ``` 等
    - 移除结尾的 ```
    """
    text = text.strip()
    # 移除开头的 ```xxx
    text = re.sub(r'^```(?:markdown|md)?\s*\n?', '', text, flags=re.IGNORECASE)
    # 移除结尾的 ```
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


def _normalize_traj(trajectory):
    """兼容 [[...]] 格式"""
    if isinstance(trajectory, list) and trajectory and isinstance(trajectory[0], list):
        return trajectory[0]
    return trajectory


def _extract_tool_call_json(text: str) -> str:
    """提取工具调用 JSON"""
    m = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", text)
    if m:
        return m.group(1).strip()
    l, r = text.find("["), text.rfind("]")
    if l != -1 and r != -1 and r > l:
        cand = text[l:r+1].strip()
        if ("tool_name" in cand) and ("tool_args" in cand):
            return cand
    return ""


def _looks_like_tool_result(text: str) -> bool:
    """判断是否为工具返回结果"""
    t = text.strip()
    # 匹配常见的工具返回格式
    if t.startswith("Tool:") or t.startswith("Result:"):
        return True
    # 匹配 [Tool: xxx] 格式
    if t.startswith("[Tool:"):
        return True
    # 匹配 <tool_response> 格式
    if "<tool_response>" in t or "</tool_response>" in t:
        return True
    # 匹配 dashscope_search 等工具的返回结果
    if t.startswith("{") and ("query" in t) and ("search_results" in t or "response_content" in t):
        return True
    # 匹配爬取工具返回的结构化数据
    if ("股票代码 |" in t) or ("单位：" in t) or t.startswith("### "):
        return True
    # 匹配同花顺工具返回的来源标记
    if "> 以下内容来自：" in t:
        return True
    return False


def _is_probably_final_report(text: str) -> bool:
    """判断是否为最终报告"""
    t = text.strip()
    return ("## References" in t) or ("[TASK_COMPLETED]" in t) or t.lstrip().startswith("# ")


def _split_tool_responses(text: str) -> List[str]:
    """
    分割多个工具响应

    处理格式如：
    [Tool: xxx]
...
</tool_response>
<tool_response>
[Tool: yyy]
...
    """
    # 先尝试按 </tool_response>\n<tool_response> 分割
    if "</tool_response>" in text and "<tool_response>" in text:
        parts = re.split(r'</tool_response>\s*<tool_response>', text)
        # 清理每个部分的标签
        cleaned = []
        for p in parts:
            p = re.sub(r'^\s*<tool_response>\s*', '', p)
            p = re.sub(r'\s*</tool_response>\s*$', '', p)
            p = p.strip()
            if p:
                cleaned.append(p)
        if cleaned:
            return cleaned

    # 尝试按 [Tool: xxx] 分割
    tool_pattern = r'(?=\[Tool:\s*[^\]]+\])'
    parts = re.split(tool_pattern, text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts

    # 无法分割，返回原文本
    return [text.strip()] if text.strip() else []


def construct_cgcv_prompt(
    trajectory: List[Dict[str, Any]],
    user_prompt_template: str
) -> str:
    """
    从 trajectory 构建 CGCV 评估 prompt

    Args:
        trajectory: 对话轨迹 [{"role": ..., "content": ...}, ...]
        user_prompt_template: 用户 prompt 模板

    Returns:
        构建好的 user prompt 字符串
    """
    traj = _normalize_traj(trajectory)
    if not traj:
        traj = []

    user_query = ""
    tool_calls: List[str] = []
    evidence: List[str] = []
    final_report = ""

    # 找到 final report（从后往前找第一个符合条件的 assistant 消息）
    for i in range(len(traj) - 1, -1, -1):
        step = traj[i]
        if step.get("role") == "assistant":
            txt = _strip_think(_extract_text_content(step.get("content")))
            if _is_probably_final_report(txt):
                final_report = txt
                break

    if not final_report:
        for i in range(len(traj) - 1, -1, -1):
            if traj[i].get("role") == "assistant":
                final_report = _strip_think(_extract_text_content(traj[i].get("content")))
                break

    # 清理 markdown 代码块标记
    final_report = _strip_markdown_fences(final_report)

    # 遍历提取 user_query, tool_calls, evidence
    evidence_idx = 0
    for idx, step in enumerate(traj):
        role = step.get("role")
        raw = _extract_text_content(step.get("content"))
        txt = _strip_think(raw)
        if not raw:
            continue

        # 跳过 system 消息
        if role == "system":
            continue

        if role == "user" and not user_query and (not _looks_like_tool_result(raw)):
            user_query = txt
            continue

        if role == "assistant":
            call_json = _extract_tool_call_json(raw)
            if call_json:
                tool_calls.append(f"【工具调用 {len(tool_calls) + 1}】\n{call_json}")

        if role == "tool":
            # 处理多工具响应的情况
            tool_parts = _split_tool_responses(raw)
            for part in tool_parts:
                if part:
                    evidence_idx += 1
                    evidence.append(f"【Evidence {evidence_idx}】\n{part}")
        elif role == "user" and user_query and _looks_like_tool_result(raw):
            # 某些情况下工具结果可能在 user 消息中
            evidence_idx += 1
            evidence.append(f"【Evidence {evidence_idx}】\n{raw}")

    # 构建 evidence_text，使用更清晰的分隔
    evidence_parts = []
    if evidence:
        evidence_parts.append("\n\n".join(evidence))

    evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "（无可用证据）"

    return user_prompt_template.format(
        user_query=user_query,
        evidence_text=evidence_text,
        report=final_report
    ).strip()


# =============================================================================
# Score Computation
# =============================================================================

def compute_cgcv_score(
    result: CGCVResult,
    citation_weight: float = 0.3,
    alignment_weight: float = 0.7
) -> Tuple[float, str]:
    """
    计算 CGCV 评分

    评分策略：
    1. 基础分：verified / total
    2. 可选：分层评分
       - citation_score: 有引用且可追溯的比例
       - alignment_score: 内容对齐的比例（在有有效引用的前提下）

    Args:
        result: CGCVResult 对象
        citation_weight: 引用分数权重（默认 0.3）
        alignment_weight: 对齐分数权重（默认 0.7）

    Returns:
        (score, reason) 元组
    """
    total = result.total

    if total == 0:
        return 0.0, "no_claims_detected"

    # 简单评分：verified / total
    base_score = result.verified / total

    # 分层统计
    citation_issues = result.citation_missing + result.citation_broken
    claims_with_valid_citation = total - citation_issues

    # 引用有效率
    citation_valid_rate = claims_with_valid_citation / total if total > 0 else 0.0

    # 对齐正确率（在有效引用中）
    if claims_with_valid_citation > 0:
        alignment_correct_rate = result.verified / claims_with_valid_citation
    else:
        alignment_correct_rate = 0.0

    # 加权分数
    weighted_score = (
        citation_weight * citation_valid_rate +
        alignment_weight * alignment_correct_rate
    )

    # 最终使用基础分数（更直观）
    final_score = base_score

    # 构建 reason
    reason_parts = [
        f"total={total}",
        f"verified={result.verified}",
        f"citation_missing={result.citation_missing}",
        f"citation_broken={result.citation_broken}",
        f"alignment_issues={result.alignment_issues}",
        f"score={final_score:.4f}",
    ]

    # 添加错误摘要
    if result.alignment_issues > 0:
        # 统计各类对齐错误
        error_counts = {}
        for claim in result.claims:
            if claim.is_alignment_issue():
                error_counts[claim.status] = error_counts.get(claim.status, 0) + 1
        error_summary = ", ".join(f"{k}:{v}" for k, v in error_counts.items())
        reason_parts.append(f"errors=[{error_summary}]")

    reason = " | ".join(reason_parts)
    return round(final_score, 6), reason[:800]
