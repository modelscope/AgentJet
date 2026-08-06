# -*- coding: utf-8 -*-
"""可切换的 <tool_call> 工具调用解析器 (ajet 层, 不依赖 vLLM parser 接口差异).

背景: 不同模型用不同文本格式表达工具调用:
  - hermes 风格  : <tool_call>{...json...}</tool_call>   (内容为 JSON: {"name","arguments"})
  - qwen3_coder : <tool_call>\n<function=NAME>\n<parameter=K>v</parameter>...</function>\n</tool_call>
                  (XML 风格, Qwen3.6/Qwen3 Coder 系)
agentjet 里 `ajet.rollout.tool_parser` 配置选哪种 (默认 hermes, 兼容旧模型).
所有解析器返回统一结构, 与 vLLM Hermes `ExtractedToolCallInformation.model_dump()` 兼容:
    {"tools_called": bool, "tool_calls": [{"type":"function",
       "function": {"name": str, "arguments": <json-str>}}], "content": str|None}
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _extracted(tools_called: bool, tool_calls: List[Dict[str, Any]],
               content: Optional[str]) -> Dict[str, Any]:
    return {
        "tools_called": tools_called,
        "tool_calls": tool_calls,
        "content": content if content else None,
    }


def _make_tool_call(name: str, arguments: Any) -> Dict[str, Any]:
    """arguments 转 JSON 字符串, 与 Hermes parser 的 FunctionCall.arguments 一致."""
    if isinstance(arguments, str):
        # 已是 JSON 字符串 -> 校验/保留
        try:
            json.loads(arguments)
            args_str = arguments
        except json.JSONDecodeError:
            args_str = json.dumps(arguments, ensure_ascii=False)
    else:
        args_str = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": f"toolu_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


# ---------------------------------------------------------------------------
# hermes: <tool_call>{json}</tool_call>
# ---------------------------------------------------------------------------
_HERMES_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _parse_hermes(text: str) -> Dict[str, Any]:
    matches = _HERMES_RE.findall(text)
    if not matches:
        return _extracted(False, [], text)
    tool_calls: List[Dict[str, Any]] = []
    for m in matches:
        raw = m if m else ""
        try:
            fc = json.loads(raw)
            name = fc.get("name", "")
            arguments = fc.get("arguments", {})
            tool_calls.append(_make_tool_call(name, arguments))
        except Exception:
            logger.warning("[tool_parser] hermes json parse failed: %s", raw[:200])
            return _extracted(False, [], text)
    content = text[: text.find("<tool_call>")] or None
    return _extracted(True, tool_calls, content)


# ---------------------------------------------------------------------------
# qwen3_coder (XML): <tool_call>\n<function=NAME>\n<parameter=K>v</parameter>...\n</function>\n</tool_call>
# ---------------------------------------------------------------------------
_QWEN3_TOOLCALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_QWEN3_FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_QWEN3_PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)


def _parse_qwen3_coder(text: str) -> Dict[str, Any]:
    blocks = _QWEN3_TOOLCALL_RE.findall(text)
    if not blocks:
        return _extracted(False, [], text)
    tool_calls: List[Dict[str, Any]] = []
    for block in blocks:
        fns = _QWEN3_FUNCTION_RE.findall(block)
        for fname, fbody in fns:
            name = fname.strip()
            params: Dict[str, Any] = {}
            for pname, pval in _QWEN3_PARAM_RE.findall(fbody):
                key = pname.strip()
                val = pval.strip()
                # 值保持字符串 (claude 的 tool_use input 接受 str 或 dict; 这里给 str)
                params[key] = val
            # 无 <parameter> 时把 <function> body 整体当 arguments
            if not params:
                body = fbody.strip()
                if body:
                    params = body
            tool_calls.append(_make_tool_call(name, params))
    if not tool_calls:
        return _extracted(False, [], text)
    content = text[: text.find("<tool_call>")] or None
    return _extracted(True, tool_calls, content)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
_TOOL_PARSERS: Dict[str, Any] = {
    "hermes": _parse_hermes,
    "qwen3_coder": _parse_qwen3_coder,
}


def get_available_tool_parsers() -> List[str]:
    return sorted(_TOOL_PARSERS.keys())


def parse_tool_calls(text: str, parser_name: Optional[str]) -> Dict[str, Any]:
    """按 parser_name 解析 text 里的 <tool_call>; 未知名字回退 hermes."""
    name = (parser_name or "hermes").strip().lower()
    fn = _TOOL_PARSERS.get(name)
    if fn is None:
        logger.warning("[tool_parser] unknown parser %r, fallback hermes", parser_name)
        fn = _parse_hermes
    try:
        return fn(text)
    except Exception as e:
        logger.exception("[tool_parser] %s parse error: %s", name, e)
        return _extracted(False, [], text)
