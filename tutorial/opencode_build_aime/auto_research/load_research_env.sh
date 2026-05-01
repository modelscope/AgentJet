#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/data_cpfs/qingxu.fu/alpha_auto_research/research_config.jsonc"
PYTHON_BIN="${PYTHON_BIN:-python3}"

eval "$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import json
import shlex
import sys
from pathlib import Path


def strip_jsonc_comments(text: str) -> str:
    out = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


config_path = Path(sys.argv[1])
text = strip_jsonc_comments(config_path.read_text(encoding="utf-8"))
config = json.loads(text)

for key in ("SWANLAB_WEB_HOST", "SWANLAB_API_KEY", "SWANLAB_API_HOST"):
    value = config["swanlab"][key]
    print(f"export {key}={shlex.quote(value)}")
PY
)"
