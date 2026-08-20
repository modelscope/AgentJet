#!/usr/bin/env python3
"""Salvage claude-code transcript jsonl from LIVE e2b sandboxes, registry-driven.

For a run whose client predates the fixed download roots (or any run): poll the
lifecycle registry, connect to every still-alive sandbox, and pull
/root/.claude/projects ( + /root/cc_data ) *.jsonl into --out-dir. Stable file
names (no timestamp) so each poll overwrites with grown content; the last poll
before a sandbox dies is its final transcript.

Usage:
  python3 salvage_jsonl.py --registry <exp>/e2b_sandbox_registry.log --out-dir <exp>/jsonl
  ... --watch 300     # loop every N seconds until stopped
Env: E2B_API_KEY, E2B_DOMAIN (E2B_VALIDATE_API_KEY=false auto-set).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import urllib.error
import urllib.request


def http_get(url: str, headers: dict) -> int:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def alive_ids(registry: str, api: str, hdr: dict) -> list[str]:
    last: dict[str, str] = {}
    with open(registry, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                last[parts[2]] = parts[1]
    cand = [sid for sid, ev in last.items() if ev in ("CREATED", "KILL_FAILED")]
    out = []
    for sid in cand:
        if http_get(f"{api}/sandboxes/{sid}", hdr) == 200:
            out.append(sid)
    return out


async def pull_one(sid: str, out_dir: str) -> int:
    from e2b import AsyncSandbox
    sb = await AsyncSandbox.connect(sid)
    r = await sb.commands.run(
        "find /root/cc_data /root/.claude/projects -name '*.jsonl' 2>/dev/null", timeout=30)
    n = 0
    for p in (r.stdout or "").split():
        try:
            txt = await sb.files.read(p, format="text")
            name = f"{sid}-{p.replace('/', '_')}.jsonl"
            with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
                f.write(txt or "")
            n += 1
        except Exception:
            pass
    return n


async def run_once(registry: str, api: str, hdr: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ids = alive_ids(registry, api, hdr)
    ok = files = 0
    for sid in ids:
        try:
            files += await pull_one(sid, out_dir)
            ok += 1
        except Exception:
            pass  # died mid-pull / connect refused -> skip this round
    import time as _t
    print(f"[{_t.strftime('%H:%M:%S')}] alive={len(ids)} pulled_from={ok} files={files} -> {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--watch", type=int, default=0, help="loop every N seconds")
    args = ap.parse_args()

    key = os.environ.get("E2B_API_KEY")
    if not key:
        sys.exit("need E2B_API_KEY")
    os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")
    domain = os.environ.get("E2B_DOMAIN", "sandbox01.vpc.cn-hongkong.pai-eas.aliyuncs.com")
    api = f"https://api.{domain}"
    hdr = {"X-API-KEY": key}

    while True:
        try:
            asyncio.run(run_once(args.registry, api, hdr, args.out_dir))
        except Exception as e:
            print(f"round error: {e}", file=sys.stderr, flush=True)
        if args.watch <= 0:
            return
        import time as _t
        _t.sleep(args.watch)


if __name__ == "__main__":
    main()
