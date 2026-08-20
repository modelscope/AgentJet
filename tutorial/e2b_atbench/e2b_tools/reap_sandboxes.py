#!/usr/bin/env python3
"""Reap orphaned e2b sandboxes recorded in the lifecycle registry -- READ-ONLY by default.

Reads the registry written by atbench_runtime.sandbox._registry_log
(``{ts}\tCREATED|KILLED|KILL_FAILED\t{sandbox_id}\t{pid}``), takes every id whose
LAST event is CREATED/KILL_FAILED (i.e. never cleanly killed -- the signature of a
hard-killed run), probes each via gateway GET, and -- with --kill -- deletes the
ones still alive that match our template and are older than --min-age-min
(the age filter protects the LIVE run's freshly created sandboxes).

Usage:
  python3 reap_sandboxes.py                                    # dry-run
  python3 reap_sandboxes.py --kill                             # kill confirmed orphans
  python3 reap_sandboxes.py --registry /path/registry.log      # explicit registry
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request


def default_api_host() -> str:
    domain = os.environ.get("E2B_DOMAIN", "sandbox01.vpc.cn-hongkong.pai-eas.aliyuncs.com")
    return f"https://api.{domain}"


def http(method: str, url: str, headers: dict) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def parse_registry(path: str) -> dict:
    """id -> {"last": event, "created": ts-str}"""
    state: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            ts, event, sid = parts[0], parts[1], parts[2]
            rec = state.setdefault(sid, {"last": event, "created": ts})
            rec["last"] = event
            if event == "CREATED":
                rec["created"] = ts
    return state


def age_min(ts: str) -> float:
    # registry timestamps are the client box's LOCAL time (time.strftime in
    # sandbox._registry_log); compare against local now. (Treating them as UTC
    # inflated ages by +8h on the CST boxes, weakening min-age protection of a
    # live run's young sandboxes.)
    t0 = dt.datetime.fromisoformat(ts)
    return (dt.datetime.now() - t0).total_seconds() / 60


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.environ.get("E2B_SANDBOX_REGISTRY", "tmp/e2b_sandbox_registry.log"))
    ap.add_argument("--api-host", default=default_api_host())
    ap.add_argument("--template", default=os.environ.get("SLIME_AGENT_E2B_TEMPLATE"),
                    help="only delete sandboxes whose templateID equals this (default $SLIME_AGENT_E2B_TEMPLATE)")
    ap.add_argument("--min-age-min", type=float, default=10.0,
                    help="skip sandboxes younger than this (protects the live run; default 10)")
    ap.add_argument("--kill", action="store_true", help="actually DELETE (default: dry-run)")
    args = ap.parse_args()

    key = os.environ.get("E2B_API_KEY")
    if not key:
        sys.exit("need E2B_API_KEY in env (X-API-KEY header)")
    hdr = {"X-API-KEY": key}

    state = parse_registry(args.registry)
    orphans = {sid: rec for sid, rec in state.items() if rec["last"] in ("CREATED", "KILL_FAILED")}
    print(f"registry: {args.registry} | ids={len(state)} | never-killed={len(orphans)} | mode={'KILL' if args.kill else 'DRY-RUN'}")

    n_killed = n_dead = n_skip = 0
    for sid, rec in sorted(orphans.items(), key=lambda kv: kv[1]["created"]):
        status, body = http("GET", f"{args.api_host}/sandboxes/{sid}", hdr)
        if status == 404:
            n_dead += 1
            continue
        if status != 200:
            print(f"  {sid}  GET ERR {status}: {body[:60]}")
            continue
        info = json.loads(body)
        tpl = info.get("templateID", "?")
        a = age_min(rec["created"])
        why_skip = []
        if args.template and tpl != args.template:
            why_skip.append(f"template={tpl}")
        if a < args.min_age_min:
            why_skip.append(f"age={a:.0f}m<{args.min_age_min}m")
        if why_skip:
            n_skip += 1
            print(f"  {sid}  SKIP ({', '.join(why_skip)})")
            continue
        if not args.kill:
            print(f"  {sid}  ORPHAN state={info.get('state')} template={tpl} age={a:.0f}m group={info.get('sandboxGroup')}")
            continue
        st, _ = http("DELETE", f"{args.api_host}/sandboxes/{sid}", hdr)
        print(f"  {sid}  DELETE -> {st}")
        n_killed += int(st in (200, 202, 204))
    print(f"summary: killed={n_killed} already-dead={n_dead} skipped={n_skip}"
          + ("" if args.kill else "  (dry-run; pass --kill to delete)"))


if __name__ == "__main__":
    main()
