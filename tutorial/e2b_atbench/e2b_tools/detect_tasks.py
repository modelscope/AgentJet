#!/usr/bin/env python3
"""Detect and print live e2b (PAI-EAS gateway) sandbox tasks -- READ-ONLY.

Two data sources:
  1. Gateway Prometheus ``/metrics`` (NO auth): live sandbox counts per
     (sandbox_group, template_id, phase), plus created/deleted totals and
     aggregate resource requests. This is the only enumeration-level view the
     gateway exposes to a service-level key (GET /sandboxes is 401).
  2. ``--probe-ids``: per-sandbox GET (needs E2B_API_KEY, X-API-KEY header) to
     print state/age detail for known sandbox ids (registry / error logs).

This tool NEVER deletes anything. Cleanup goes through an explicit reaper flow.

Usage:
  python3 detect_tasks.py                          # counts table + orphan hint
  python3 detect_tasks.py --concurrency 32         # set expected live episodes
  python3 detect_tasks.py --interval 60            # watch mode
  python3 detect_tasks.py --probe-ids sbx-xxxx     # detail for known ids
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

RE_PHASE = re.compile(r'^sandbox_gateway_sandboxes_phase_count\{([^}]*)\}\s+([0-9.]+)$')
RE_CTD = re.compile(r'^sandbox_gateway_sandboxes_(created|deleted)_total\{([^}]*)\}\s+([0-9.]+)$')
RE_RES = re.compile(r'^sandbox_gateway_sandboxes_requested_(cpu_cores|memory_gib|storage_gib)\{([^}]*)\}\s+([0-9.]+)$')
PHASES = ("Running", "Pending", "Terminating", "Failed", "other")


def parse_labels(raw: str) -> dict:
    return dict(re.findall(r'(\w+)="([^"]*)"', raw))


def http_get(url: str, headers: dict | None = None, timeout: float = 20.0) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def default_api_host() -> str:
    domain = os.environ.get("E2B_DOMAIN", "sandbox01.vpc.cn-hongkong.pai-eas.aliyuncs.com")
    return f"https://api.{domain}"


def fetch_metrics(api: str) -> str:
    status, body = http_get(f"{api}/metrics")
    if status != 200:
        raise RuntimeError(f"GET {api}/metrics -> {status}: {body[:200]}")
    return body


def collect(body: str) -> dict:
    """rows[(group, template)] = {phase: n, created: n, deleted: n, cpu/mem/disk: n}"""
    rows: dict[tuple[str, str], dict] = {}

    def slot(g: str, t: str) -> dict:
        return rows.setdefault((g, t), {"created": 0, "deleted": 0, "cpu": 0.0, "mem": 0.0, "disk": 0.0})

    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = RE_PHASE.match(line)
        if m:
            lab, val = parse_labels(m.group(1)), float(m.group(2))
            s = slot(lab.get("sandbox_group", "?"), lab.get("template_id", "?"))
            ph = lab.get("phase") or "other"
            ph = ph if ph in PHASES else "other"
            s[ph] = s.get(ph, 0) + int(val)
            continue
        m = RE_CTD.match(line)
        if m:
            lab = parse_labels(m.group(2))
            s = slot(lab.get("sandbox_group", "?"), lab.get("template_id", "?"))
            s[m.group(1)] = int(float(m.group(3)))
            continue
        m = RE_RES.match(line)
        if m:
            lab = parse_labels(m.group(2))
            s = slot(lab.get("sandbox_group", "?"), lab.get("template_id", "?"))
            key = {"cpu_cores": "cpu", "memory_gib": "mem", "storage_gib": "disk"}[m.group(1)]
            s[key] += float(m.group(3))
    return rows


def fmt_age(started_at: str) -> str:
    try:
        t0 = dt.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        return f"{(dt.datetime.now(dt.timezone.utc) - t0).total_seconds() / 60:.0f}m"
    except Exception:
        return "?"


def print_counts(api: str, ours: str | None, concurrency: int) -> None:
    rows = collect(fetch_metrics(api))
    now = dt.datetime.now().strftime("%H:%M:%S")
    print(f"\n=== e2b sandbox tasks by template  [{now}]  ({api}/metrics, no-auth) ===")
    hdr = f"{'group':<20} {'template':<38} {'Run':>5} {'Pend':>5} {'Term':>5} {'Fail':>5} {'net':>6} {'cpu':>6} {'memG':>6} {'dskG':>6}"
    print(hdr)
    print("-" * len(hdr))
    for (g, t), s in sorted(rows.items(), key=lambda kv: -(kv[1].get("Running", 0))):
        if not any(s.get(p, 0) for p in PHASES):
            continue
        mark = "  <== OURS" if ours and t == ours else ""
        print(f"{g:<20} {t[:38]:<38} {s.get('Running', 0):>5} {s.get('Pending', 0):>5} "
              f"{s.get('Terminating', 0):>5} {s.get('Failed', 0):>5} "
              f"{s['created'] - s['deleted']:>6} {s['cpu']:>6.0f} {s['mem']:>6.0f} {s['disk']:>6.0f}{mark}")
    if ours and any(t == ours for (_, t) in rows):
        run = sum(s.get("Running", 0) for (g, t), s in rows.items() if t == ours)
        over = run - concurrency
        hint = (f"  -> ~{over} ORPHAN SUSPECTS (endAt=created+timeout reclaims them automatically)"
                if over > 0 else "  -> within expected concurrency")
        print(f"\nOURS [{ours}] Running={run}, expected<={concurrency}{hint}")
    else:
        print(f"\n(ours template {ours!r} not present in metrics)" if ours else "")


def probe_ids(api: str, ids: list[str]) -> None:
    key = os.environ.get("E2B_API_KEY")
    if not key:
        sys.exit("--probe-ids needs E2B_API_KEY in env (X-API-KEY header)")
    hdr = {"X-API-KEY": key}
    print(f"\n=== per-sandbox probe ({len(ids)} ids, GET only) ===")
    print(f"{'sandbox_id':<22} {'state':<12} {'age':>5} {'endAt(UTC)':<21} {'group':<20} template")
    for sid in ids:
        status, body = http_get(f"{api}/sandboxes/{sid}", headers=hdr)
        if status == 404:
            print(f"{sid:<22} {'DEAD(404)':<12}")
            continue
        if status != 200:
            print(f"{sid:<22} {'ERR':<12} HTTP {status}: {body[:80]}")
            continue
        j = json.loads(body)
        print(f"{sid:<22} {j.get('state', '?'):<12} {fmt_age(j.get('startedAt', '')):>5} "
              f"{j.get('endAt', '?'):<21} {j.get('sandboxGroup', '?'):<20} {j.get('templateID', '?')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-host", default=default_api_host())
    ap.add_argument("--template", default=os.environ.get("SLIME_AGENT_E2B_TEMPLATE"),
                    help="our template id (marked + orphan-checked); default $SLIME_AGENT_E2B_TEMPLATE")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="expected max live sandboxes of our template (default 32)")
    ap.add_argument("--interval", type=int, default=0, help="refetch every N seconds (watch mode)")
    ap.add_argument("--probe-ids", nargs="*", default=[], help="sandbox ids to GET-detail (needs E2B_API_KEY)")
    args = ap.parse_args()

    while True:
        try:
            print_counts(args.api_host, args.template, args.concurrency)
            if args.probe_ids:
                probe_ids(args.api_host, args.probe_ids)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
        if args.interval <= 0:
            break
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
