# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


# --------------------------
# JSON parsing helpers
# --------------------------

def strict_load_json(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from model output.

    - Accept plain JSON.
    - If extra text exists, extract the first {...} block.
    """
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Invalid JSON output")


def _clip(s: str, n: int) -> str:
    s = s or ""
    s = s.replace("\u0000", "")
    return s[:n]


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int, float)):
            return int(x)
        if isinstance(x, str) and x.strip():
            return int(float(x.strip()))
    except Exception:
        return default
    return default


def _as_list_str(x: Any, max_items: int = 10, max_len: int = 60) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for item in x[:max_items]:
        if isinstance(item, str):
            out.append(_clip(item, max_len))
        else:
            out.append(_clip(str(item), max_len))
    return out


def validate_shape(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize model output for TVR.

    Returns:
      {
        "claims": [...],
        "stats": {...},
        "examples": {...}
      }
    """
    if not isinstance(obj, dict):
        raise ValueError("Output is not a JSON object")

    claims_raw = obj.get("claims", [])
    if not isinstance(claims_raw, list):
        claims_raw = []

    claims: List[Dict[str, Any]] = []
    for c in claims_raw[:25]:
        if not isinstance(c, dict):
            continue

        claim = _clip(str(c.get("claim", "")), 240)
        ctype = _clip(str(c.get("type", "other")), 24)

        sig = c.get("signature", {})
        if not isinstance(sig, dict):
            sig = {}
        entities = _as_list_str(sig.get("entities", []), max_items=10, max_len=50)
        numbers = _as_list_str(sig.get("numbers", []), max_items=10, max_len=40)
        times = _as_list_str(sig.get("times", []), max_items=10, max_len=40)

        anchors_raw = c.get("anchors", [])
        anchors: List[Dict[str, Any]] = []
        if isinstance(anchors_raw, list):
            for a in anchors_raw[:2]:
                if not isinstance(a, dict):
                    continue
                step = _as_int(a.get("step", -1), default=-1)
                quote = _clip(str(a.get("quote", "")), 120)
                if step >= 0 and quote:
                    anchors.append({"step": step, "quote": quote})

        verdict = _clip(str(c.get("verdict", "unclear")), 20)
        if verdict not in {"supported", "contradicted", "no_evidence", "speculative_ok", "unclear"}:
            verdict = "unclear"

        issue = _clip(str(c.get("issue", "none")), 20)
        allowed_issues = {
            "none", "entity_mismatch", "time_mismatch", "value_mismatch", "scope_mismatch",
            "logic_leap", "over_precision", "missing_anchor"
        }
        if issue not in allowed_issues:
            issue = "none"

        note = _clip(str(c.get("note", "")), 80)

        claims.append({
            "claim": claim,
            "type": ctype,
            "signature": {"entities": entities, "numbers": numbers, "times": times},
            "anchors": anchors,
            "verdict": verdict,
            "issue": issue,
            "note": note,
        })

    # stats
    stats_raw = obj.get("stats", {})
    if not isinstance(stats_raw, dict):
        stats_raw = {}

    # always re-count to avoid mismatch / gaming
    verdict_counts = {
        "supported": 0,
        "contradicted": 0,
        "no_evidence": 0,
        "speculative_ok": 0,
        "unclear": 0,
    }
    for c in claims:
        verdict_counts[c["verdict"]] += 1

    report_digit_tokens = max(0, _as_int(stats_raw.get("report_digit_tokens", 0), default=0))
    covered_digit_tokens = max(0, _as_int(stats_raw.get("covered_digit_tokens", 0), default=0))

    stats = {
        "total_claims": len(claims),
        "supported": verdict_counts["supported"],
        "contradicted": verdict_counts["contradicted"],
        "no_evidence": verdict_counts["no_evidence"],
        "speculative_ok": verdict_counts["speculative_ok"],
        "unclear": verdict_counts["unclear"],
        "report_digit_tokens": report_digit_tokens,
        "covered_digit_tokens": covered_digit_tokens,
    }

    # examples (small)
    examples_raw = obj.get("examples", {})
    if not isinstance(examples_raw, dict):
        examples_raw = {}

    def _normalize_example_list(x: Any, max_items: int = 2) -> List[Dict[str, Any]]:
        if not isinstance(x, list):
            return []
        out: List[Dict[str, Any]] = []
        for it in x[:max_items]:
            if isinstance(it, dict):
                out.append({k: _clip(str(v), 140) for k, v in list(it.items())[:3]})
            elif isinstance(it, str):
                out.append({"text": _clip(it, 140)})
        return out

    examples = {
        "best_supported": _normalize_example_list(examples_raw.get("best_supported", []), 2),
        "worst_failed": _normalize_example_list(examples_raw.get("worst_failed", []), 2),
    }

    return {"claims": claims, "stats": stats, "examples": examples}


# --------------------------
# Trajectory helpers
# --------------------------

def coerce_to_messages_list(traj: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
    - list[dict]
    - list[list[dict]] (take first non-empty inner list)
    - dict with keys: traj / messages / conversation / steps (best-effort)

    Returns list[dict] message objects.
    """
    if traj is None:
        return []

    if isinstance(traj, dict):
        for key in ("traj", "messages", "conversation", "steps"):
            if key in traj:
                return coerce_to_messages_list(traj[key])
        return []

    if isinstance(traj, list):
        if not traj:
            return []
        if isinstance(traj[0], list):
            for inner in traj:
                if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                    return inner
            return []
        if isinstance(traj[0], dict):
            return traj

    return []


def _extract_text_content(content: Any) -> str:
    """
    Extract textual content from different possible message formats.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join([p for p in parts if p])
    return str(content)


def strip_references(markdown: str) -> str:
    """
    Remove References section and anything after it (common Markdown headings).
    """
    if not isinstance(markdown, str):
        return ""
    m = re.search(r"\n#+\s*References\b", markdown, flags=re.IGNORECASE)
    if m:
        return markdown[: m.start()].strip()
    return markdown.strip()


def count_digit_tokens(text: str) -> int:
    """
    Rough count for digit/date tokens in text.
    """
    if not text:
        return 0
    pats = [
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",      # ISO-ish date
        r"\b\d+(?:\.\d+)?%?\b",                  # number / percent
    ]
    tokens: List[str] = []
    for p in pats:
        tokens.extend(re.findall(p, text))
    return len(tokens)


def _is_probably_final_report(text: str) -> bool:
    """
    Heuristic: final report is usually markdown-ish and contains References / TASK_COMPLETED etc.
    """
    if not text:
        return False
    # allow either TASK_COMPLETED or markdown headings + References
    has_markdown = ("#" in text) or ("|---" in text) or ("## " in text)
    has_refs = re.search(r"#+\s*References\b", text, flags=re.IGNORECASE) is not None
    has_done = "[TASK_COMPLETED]" in text
    return has_done or (has_markdown and has_refs)


def _extract_tool_calls_and_results(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract tool call and tool output blocks, loosely following the format in your existing data.
    """
    items: List[Dict[str, Any]] = []
    for i, msg in enumerate(trajectory):
        role = msg.get("role", "")
        content = _extract_text_content(msg.get("content", ""))

        if role == "assistant":
            # look for JSON code block that indicates tool calls
            if "```json" in content and ("tool_name" in content or "tool_args" in content):
                items.append({"step": i, "kind": "tool_call", "text": content})
        elif role == "tool":
            items.append({"step": i, "kind": "tool_result", "text": content})
    return items


def construct_reward_prompt(trajectory: List[Dict[str, Any]], user_prompt_template: str) -> str:
    """
    Build a user prompt with:
      - user_query: last user message
      - evidence_text: concatenated tool calls/results with step index
      - final_report: last assistant message that looks like final report
    """
    trajectory = coerce_to_messages_list(trajectory)

    user_query = ""
    for msg in reversed(trajectory):
        if msg.get("role") == "user":
            user_query = _extract_text_content(msg.get("content", ""))
            break

    final_report = ""
    for msg in reversed(trajectory):
        if msg.get("role") == "assistant":
            t = _extract_text_content(msg.get("content", ""))
            if _is_probably_final_report(t):
                final_report = t
                break
    if not final_report:
        # fallback to last assistant msg
        for msg in reversed(trajectory):
            if msg.get("role") == "assistant":
                final_report = _extract_text_content(msg.get("content", ""))
                break

    evidence_items = _extract_tool_calls_and_results(trajectory)
    evidence_lines: List[str] = []
    for it in evidence_items:
        step = it["step"]
        kind = it["kind"]
        prefix = "CALL" if kind == "tool_call" else "RESULT"
        evidence_lines.append(f"[{prefix} step={step}]\n{it['text']}".strip())
    evidence_text = "\n\n".join(evidence_lines).strip()

    return user_prompt_template.format(
        user_query=user_query,
        evidence_text=evidence_text,
        final_report=final_report,
    )


def construct_traceability_prompt(
    trajectory: List[Dict[str, Any]],
    user_prompt_template: str,
) -> Tuple[str, str]:
    """
    Returns:
      - user_prompt (for the judge model)
      - report_plain (final report without References) for deterministic coverage checks
    """
    user_prompt = construct_reward_prompt(trajectory, user_prompt_template)

    final_report = ""
    marker = "\n## AI Report\n"
    if marker in user_prompt:
        final_report = user_prompt.split(marker, 1)[1]
    # Cut at "\n\n### 审计流程" if present.
    cut = "\n\n### 审计流程"
    if cut in final_report:
        final_report = final_report.split(cut, 1)[0]

    report_plain = strip_references(final_report)
    return user_prompt, report_plain
