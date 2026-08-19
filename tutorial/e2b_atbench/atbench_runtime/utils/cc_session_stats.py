#!/usr/bin/env python3
"""claude-code 转写 jsonl -> 智能体用量统计 (v2, 中枢侧; 兼容远程 pipeline 用法)。

针对 ~/.claude/projects 下 2.1.195/2.1.197 全量转写实测出的可靠性规则:

  1. 子代理是独立文件: <project>/<session-uuid>/subagents/agent-<hash>.jsonl,
     同目录还有 agent-<hash>.meta.json = {agentType, description, toolUseId,
     spawnDepth}。嵌套子代理(depth>=2)与 depth-1 平铺在同一目录。
  2. 一次 LLM 调用会被拆成多条 assistant jsonl 条目(text 块一条 / tool_use
     块一条), 且每条重复携带同一份全量 usage -> **必须按 message.id 去重**,
     否则输出 token 直接翻倍(实测 60379 vs 30521)。
  3. 偶发(538/3715 文件)网关不回传 usage: 条目 usage 全 0。这些仍是真实
     LLM 调用, 计入调用数, 但 final_context 取"最后一条非零 usage"的消息,
     并单独记 n_zero_usage_msgs。
  4. 子代理与 spawn 调用的关联: 优先 meta.toolUseId 与主转写/父代理转写中
     Agent(旧名 Task) tool_use id 精确匹配; 无 meta 时退回时间戳就近。
     智能体数量同时给两个口径: 主转写 spawn 次数 vs 子代理转写文件数
     (正常应相等; 不等说明有 spawn 失败/后台未收尾, 输出 mismatch 标记)。
  5. 一个项目目录可能有多条主 session(重试/续跑) -> 按 session 逐条出。

统计口径:
  agent 数        = 子代理转写文件数(全深度) 与 主转写 Agent/Task tool_use 数 互验
  结束时上下文长度 = 最后一条非零 usage 消息的 input+cache_read+cache_creation
  输出 token      = 去重后各消息 output_tokens 求和
  耗时            = 首~末条 timestamp 差 (墙钟, 含工具执行)

CLI:
  python3 cc_session_stats.py <src> [-o out.json] [--md]
      src = trajectory tar.gz / 解包目录 / 单项目目录 / 扁平化 jsonl 目录
  python3 cc_session_stats.py --scan <projects_root> [-o OUT_DIR]
      遍历 ~/.claude/projects 全部项目 -> sessions.csv / subagents.csv /
      agent_scan_summary.md (中枢批分析用)

本模块只依赖标准库。
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import re
import sys
import tarfile
from datetime import datetime, timezone

# 旧名 Task / 新名 Agent (v2.1.63 起) 都认, 统计 spawn 次数时二者等价。
_SPAWN_TOOL_NAMES = ("Agent", "Task")
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _parse_ts(s):
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _fmt_ts(epoch):
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _iter_entries(text):
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if isinstance(e, dict):
            yield e


def _usage_of(entry):
    u = (entry.get("message") or {}).get("usage") or {}
    return u if isinstance(u, dict) else {}


def _ctx_tokens(u):
    return ((u.get("input_tokens") or 0)
            + (u.get("cache_read_input_tokens") or 0)
            + (u.get("cache_creation_input_tokens") or 0))


def _usage_nonzero(u):
    return bool((u.get("input_tokens") or 0) or (u.get("cache_read_input_tokens") or 0)
                or (u.get("cache_creation_input_tokens") or 0) or (u.get("output_tokens") or 0))


def _analyze_one(entries_iter, *, role, agent_id=None):
    """单个 agent (主会话或一个子代理) 的转写条目 -> 统计。

    关键: 按 message.id 去重 (一次调用拆多条, usage 重复), 无 id 时退回条目 uuid。
    """
    st = {
        "role": role, "agent_id": agent_id, "session_id": None,
        "n_llm_calls": 0, "n_assistant_entries": 0, "n_zero_usage_msgs": 0,
        "n_tool_calls": 0, "tool_calls": {}, "n_subagents_spawned": 0,
        "input_tokens_sum": 0, "cache_read_sum": 0, "cache_creation_sum": 0,
        "output_tokens": 0,
        "final_context_tokens": None, "final_output_tokens": None,
        "final_total_tokens": None, "end_reason": None,
        "started_at": None, "ended_at": None, "duration_s": None,
    }
    first_ts = last_ts = None
    msg_usage = {}        # message.id -> usage (重复条目覆盖为相同值, 取到即全量)
    msg_stop = {}         # message.id -> stop_reason (取末条)
    msg_order = []        # 保持消息出现顺序, 定位"最后一条"
    for e in entries_iter:
        ts = _parse_ts(e.get("timestamp"))
        if ts is not None:
            first_ts = ts if first_ts is None else first_ts
            last_ts = ts
        if st["session_id"] is None and e.get("sessionId"):
            st["session_id"] = e.get("sessionId")
        if e.get("type") != "assistant":
            continue
        st["n_assistant_entries"] += 1
        m = e.get("message") or {}
        mid = m.get("id") or f"_noid_{e.get('uuid')}"
        u = _usage_of(e)
        if mid not in msg_usage:
            msg_order.append(mid)
        msg_usage[mid] = u
        msg_stop[mid] = m.get("stop_reason") or msg_stop.get(mid)
        for c in (m.get("content") or []) if isinstance(m.get("content"), list) else []:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                name = c.get("name") or "?"
                st["n_tool_calls"] += 1
                st["tool_calls"][name] = st["tool_calls"].get(name, 0) + 1
                if name in _SPAWN_TOOL_NAMES:
                    st["n_subagents_spawned"] += 1

    for mid in msg_order:
        u = msg_usage.get(mid) or {}
        st["n_llm_calls"] += 1
        if not _usage_nonzero(u):
            st["n_zero_usage_msgs"] += 1
            continue
        st["input_tokens_sum"] += u.get("input_tokens") or 0
        st["cache_read_sum"] += u.get("cache_read_input_tokens") or 0
        st["cache_creation_sum"] += u.get("cache_creation_input_tokens") or 0
        st["output_tokens"] += u.get("output_tokens") or 0
        st["final_context_tokens"] = _ctx_tokens(u)   # 最后一条非零 usage
        st["final_output_tokens"] = u.get("output_tokens")
        st["final_total_tokens"] = st["final_context_tokens"] + (u.get("output_tokens") or 0)
    if msg_order:
        st["end_reason"] = msg_stop.get(msg_order[-1])
    if first_ts is not None:
        st["started_at"] = _fmt_ts(first_ts)
        st["ended_at"] = _fmt_ts(last_ts)
        st["duration_s"] = round(last_ts - first_ts, 1)
    return st


def _collect_spawn_events(entries):
    """Agent/Task tool_use: (ts, tool_use_id, subagent_type, description)。"""
    out = []
    for e in entries:
        if e.get("type") != "assistant":
            continue
        ts = _parse_ts(e.get("timestamp"))
        m = e.get("message") or {}
        for c in (m.get("content") or []) if isinstance(m.get("content"), list) else []:
            if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") in _SPAWN_TOOL_NAMES:
                inp = c.get("input") or {}
                out.append((ts, c.get("id"), inp.get("subagent_type"),
                            (inp.get("description") or "")[:80]))
    return out


class _SubLinker:
    """子代理 -> spawn tool_use 关联: toolUseId 精确匹配优先, 时间戳就近兜底。"""

    def __init__(self):
        self.by_id = {}      # tool_use_id -> (ts, source_agent, subagent_type, description)
        self.unmatched = []  # [(ts, tool_use_id, subagent_type, description)]

    def add_source(self, spawn_events, source_agent):
        for ts, tid, stype, desc in spawn_events:
            if tid:
                self.by_id[tid] = (ts, source_agent, stype, desc)
            self.unmatched.append((ts, tid, stype, desc))

    def link(self, tool_use_id, first_ts):
        if tool_use_id and tool_use_id in self.by_id:
            ts, src, stype, desc = self.by_id[tool_use_id]
            if (tid := tool_use_id) in [u[1] for u in self.unmatched]:
                self.unmatched = [u for u in self.unmatched if u[1] != tool_use_id or u[0] != ts]
            return src, stype, desc, tool_use_id
        # 兜底: 时间上最近且早于子代理首条(+2s)的未匹配 spawn
        best = None
        for i, (ts, tid, stype, desc) in enumerate(self.unmatched):
            if ts is None or first_ts is None or ts > first_ts + 2.0:
                continue
            if best is None or ts > self.unmatched[best][0]:
                best = i
        if best is None:
            return None, None, None, None
        ts, tid, stype, desc = self.unmatched.pop(best)
        return "main?", stype, desc, tid


def analyze_session_group(main_text, sub_files):
    """一个主会话 + subagents 目录下的全部子代理(含嵌套) -> 完整统计 dict。

    sub_files: {文件名: text} (只传 agent-*.jsonl, meta 单独经 sub_metas 传)。
    """
    main_entries = list(_iter_entries(main_text))
    main = _analyze_one(main_entries, role="main")

    sub_texts, sub_metas = {}, {}
    for fname, payload in sub_files.items():
        if fname.endswith(".meta.json"):
            try:
                sub_metas[fname[:-len(".meta.json")]] = json.loads(payload)
            except Exception:
                pass
        elif fname.endswith(".jsonl"):
            sub_texts[fname] = payload

    # 两遍: 先给每个子代理算基础统计; 再做关联(需知道所有子代理的 spawn 事件,
    # 因为 depth>=2 的父代理是另一个子代理)。
    sub_base = {}
    for fname, text in sub_texts.items():
        entries = list(_iter_entries(text))
        agent_id = next((e.get("agentId") for e in entries if e.get("agentId")), None) \
            or fname[:-len(".jsonl")]
        sub_base[fname] = {"entries": entries, "stat": _analyze_one(
            entries, role="subagent", agent_id=agent_id), "agent_id": agent_id}

    linker = _SubLinker()
    linker.add_source(_collect_spawn_events(main_entries), "main")
    for fname, info in sub_base.items():
        linker.add_source(_collect_spawn_events(info["entries"]), info["agent_id"])

    subagents = []
    for fname, info in sorted(sub_base.items()):
        s = info["stat"]
        meta = sub_metas.get(fname[:-len(".jsonl")]) or {}
        first_ts = next((_parse_ts(e.get("timestamp")) for e in info["entries"]
                         if _parse_ts(e.get("timestamp")) is not None), None)
        parent, stype, desc, tu_id = linker.link(meta.get("toolUseId"), first_ts)
        s.update({
            "subagent_type": meta.get("agentType") or stype,
            "description": (meta.get("description") or desc or "")[:80],
            "spawn_tool_use_id": meta.get("toolUseId") or tu_id,
            "spawn_depth": meta.get("spawnDepth"),
            "parent_agent": parent,
        })
        subagents.append(s)

    # 自洽校验: 总 spawn 数(主 + 各子代理自己 spawn 的嵌套) vs 子代理转写文件数。
    # 主转写只记 depth-1 的 spawn, depth>=2 的 spawn 记在父(子代理)转写里。
    spawn_total = main["n_subagents_spawned"] + sum(
        s["n_subagents_spawned"] for s in subagents)
    return {
        "session_id": main["session_id"],
        "n_subagents_spawned": main["n_subagents_spawned"],
        "n_subagents_spawned_total": spawn_total,
        "n_subagent_transcripts": len(subagents),
        "spawn_transcript_mismatch": spawn_total != len(subagents),
        "main": main,
        "subagents": subagents,
        "totals": {
            "output_tokens_all": main["output_tokens"] + sum(s["output_tokens"] for s in subagents),
            "output_tokens_main": main["output_tokens"],
            "output_tokens_subagents": sum(s["output_tokens"] for s in subagents),
            "agent_active_s_sum": round((main["duration_s"] or 0)
                                        + sum(s["duration_s"] or 0 for s in subagents), 1),
        },
    }


# ── 输入收集: 三种形态统一成 {path: text} ──

def _collect_from_tar(path):
    files = {}
    with tarfile.open(path, "r:gz") as tar:
        for m in tar:
            if not m.isfile() or m.size > 200 * 1024 * 1024:
                continue
            if m.name.endswith(".jsonl") or (m.name.endswith(".meta.json") and "subagents" in m.name):
                fh = tar.extractfile(m)
                if fh:
                    files[m.name] = io.TextIOWrapper(fh, encoding="utf-8", errors="replace").read()
    return files


def _collect_from_dir(path):
    files = {}
    for root, _dirs, names in os.walk(path):
        for n in names:
            if n.endswith(".jsonl") or (n.endswith(".meta.json") and "subagents" in root):
                p = os.path.join(root, n)
                try:
                    with open(p, encoding="utf-8", errors="replace") as f:
                        files[p] = f.read()
                except OSError:
                    pass
    return files


def _group_by_session(files):
    """{path: text} -> [(session_uuid, main_text, {sub 文件名: text})]。

    只认 .claude/projects 下的会话转写 (cc_data 是 driver 副本, 防口径重复;
    一个都没有才兜底); .mobius.jsonl 旁路流一律排除。兼容: 原生树
    (<uuid>.jsonl + <uuid>/subagents/agent-*.jsonl)、解包目录
    (root/.claude/projects/...)、_download_jsonl 扁平化文件名
    (<ts>-<tag>-root_.claude.projects_..._<uuid>.jsonl / ..._subagents_agent-x)。
    """
    mains, subs, cc_data = {}, {}, {}
    for p, text in files.items():
        base = os.path.basename(p)
        if base.endswith(".mobius.jsonl"):
            continue
        if base.endswith(".meta.json"):        # 子代理元数据, 归入该 session 的 sub 集
            m_uuid = None
            for u in _UUID_RE.finditer(p):
                m_uuid = u.group(0)
            if m_uuid and ("/subagents/" in p.replace("\\", "/") or "_subagents_" in base):
                subs.setdefault(m_uuid, {})[base] = text
            continue
        if not base.endswith(".jsonl"):
            continue
        m_uuid = None
        for u in _UUID_RE.finditer(p):
            m_uuid = u.group(0)  # 取最后一个 (项目目录名不含 uuid)
        if "/subagents/" in p.replace("\\", "/") or "_subagents_" in base:
            if m_uuid:
                subs.setdefault(m_uuid, {})[base] = text
        elif ".claude/projects" in p.replace("\\", "/") or ".claude.projects" in base:
            if m_uuid:
                mains[m_uuid] = text
        elif "cc_data" in p:
            cc_data[m_uuid or base] = text
    if not mains:
        mains = cc_data
    return [(sid, mains[sid], subs.get(sid, {})) for sid in sorted(mains)]


def analyze_path(path):
    """tar.gz / 目录 / 扁平化 jsonl 目录 -> {sessions: [...], primary: ...}。"""
    path = os.path.abspath(path)
    if os.path.isfile(path) and (path.endswith(".tar.gz") or path.endswith(".tgz")):
        files = _collect_from_tar(path)
    elif os.path.isdir(path):
        files = _collect_from_dir(path)
    else:
        raise FileNotFoundError(path)
    sessions = [analyze_session_group(mt, st) for _sid, mt, st in _group_by_session(files)]
    sessions.sort(key=lambda s: -(s["totals"]["output_tokens_all"] or 0))
    return {"source": path, "n_sessions": len(sessions),
            "primary": sessions[0] if sessions else None, "sessions": sessions}


def summarize_line(stats):
    if not stats or not stats.get("primary"):
        return "(no claude session found)"
    p = stats["primary"]
    m = p["main"]
    parts = [f"sessions={stats['n_sessions']} subagents={p['n_subagents_spawned']}"
             f"(transcripts={p['n_subagent_transcripts']})",
             f"main: {m['duration_s']}s out={m['output_tokens']}tok"
             f" fin_ctx={m['final_context_tokens']}tok calls={m['n_llm_calls']}"]
    for s in p["subagents"][:4]:
        parts.append(f"sub[{s.get('subagent_type')}] {s['duration_s']}s"
                     f" out={s['output_tokens']}tok fin_ctx={s['final_context_tokens']}tok")
    if len(p["subagents"]) > 4:
        parts.append(f"... +{len(p['subagents']) - 4} more")
    parts.append(f"total_out={p['totals']['output_tokens_all']}tok")
    return " | ".join(str(x) for x in parts)




# ── --agg: 批跑 RUN_DIR 汇总 (trajectories/*.tar.gz 优先, jsonl/<tag>/ 兜底) ──

def _verdict_of(path):
    try:
        m = re.search(r"\*\*Verdict:\*\*\s*(PASS|FAIL)",
                      open(path, encoding="utf-8", errors="replace").read(), re.IGNORECASE)
        return m.group(1).upper() if m else ""
    except OSError:
        return ""


def aggregate_run(run_dir, csv_out=None, md_out=None):
    """扫 RUN_DIR: trajectories/*.tar.gz (缺失则 jsonl/<tag>/) 逐 job 统计,
    关联 verdicts/<tag>.md 的 PASS/FAIL, 落 csv + md 并返回行列表。"""
    run_dir = os.path.abspath(run_dir)
    rows = []

    def _row_from(tag, prim):
        m = (prim or {}).get("main") or {}
        subs = (prim or {}).get("subagents") or []
        return {
            "tag": tag,
            "verdict": _verdict_of(os.path.join(run_dir, "verdicts", tag + ".md")),
            "n_subagents": (prim or {}).get("n_subagents_spawned_total"),
            "main_dur_s": m.get("duration_s"),
            "main_out_tok": m.get("output_tokens"),
            "main_fin_ctx_tok": m.get("final_context_tokens"),
            "main_llm_calls": m.get("n_llm_calls"),
            "sub_out_tok_sum": sum(s.get("output_tokens") or 0 for s in subs),
            "sub_dur_s_sum": round(sum(s.get("duration_s") or 0 for s in subs), 1),
            "all_out_tok": ((prim or {}).get("totals") or {}).get("output_tokens_all"),
        }

    for t in sorted(glob.glob(os.path.join(run_dir, "trajectories", "*.tar.gz"))):
        tag = os.path.basename(t)[:-len(".tar.gz")]
        try:
            rows.append(_row_from(tag, analyze_path(t)["primary"]))
        except Exception as e:
            rows.append({"tag": tag, "error": str(e)[:80]})
    have = {r["tag"] for r in rows}
    for d in sorted(glob.glob(os.path.join(run_dir, "jsonl", "*"))):
        tag = os.path.basename(d)
        if not os.path.isdir(d) or tag in have:
            continue
        try:
            rows.append(_row_from(tag, analyze_path(d)["primary"]))
        except Exception as e:
            rows.append({"tag": tag, "error": str(e)[:80]})
    rows.sort(key=lambda r: r["tag"])

    fields = ["tag", "verdict", "n_subagents", "main_dur_s", "main_out_tok",
              "main_fin_ctx_tok", "main_llm_calls", "sub_out_tok_sum",
              "sub_dur_s_sum", "all_out_tok", "error"]
    if csv_out:
        os.makedirs(os.path.dirname(os.path.abspath(csv_out)) or ".", exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
    if md_out:
        ok = [r for r in rows if not r.get("error") and r.get("n_subagents") is not None]
        n = len(ok) or 1
        avg = lambda k: round(sum(r[k] or 0 for r in ok) / n, 1)
        lines = ["# solver agent 用量汇总", "",
                 f"- jobs: {len(rows)} (有效 {len(ok)})",
                 f"- 平均子代理数: {avg('n_subagents')}",
                 f"- 主 agent: 平均耗时 {avg('main_dur_s')}s, 平均输出 {avg('main_out_tok')} tok,"
                 f" 平均结束上下文 {avg('main_fin_ctx_tok')} tok",
                 f"- 子代理输出合计平均: {avg('sub_out_tok_sum')} tok", "",
                 "| tag | verdict | subs | main_dur_s | main_out | main_fin_ctx | sub_out_sum | all_out |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r.get('tag','')} | {r.get('verdict','')} | {r.get('n_subagents','')}"
                         f" | {r.get('main_dur_s','')} | {r.get('main_out_tok','')}"
                         f" | {r.get('main_fin_ctx_tok','')} | {r.get('sub_out_tok_sum','')}"
                         f" | {r.get('all_out_tok','')} |"
                         + (f" ERR:{r['error']}" if r.get("error") else ""))
        os.makedirs(os.path.dirname(os.path.abspath(md_out)) or ".", exist_ok=True)
        with open(md_out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return rows


# ── --scan: 遍历 projects 根 (中枢批分析) ──

def scan_projects(root, out_dir):
    """遍历 <projects_root>/<project-dir>/<uuid>.jsonl(+<uuid>/subagents/),
    每 session 一行 -> sessions.csv; 每子代理一行 -> subagents.csv; 汇总 md。"""
    root = os.path.abspath(root)
    os.makedirs(out_dir, exist_ok=True)
    sess_rows, sub_rows = [], []
    n_proj = 0
    for d in sorted(os.listdir(root)):
        pdir = os.path.join(root, d)
        if not os.path.isdir(pdir):
            continue
        mains = [f for f in glob.glob(os.path.join(pdir, "*.jsonl"))
                 if not f.endswith(".mobius.jsonl")]
        if not mains:
            continue
        n_proj += 1
        for mf in mains:
            sid = os.path.basename(mf)[:-len(".jsonl")]
            try:
                with open(mf, encoding="utf-8", errors="replace") as f:
                    main_text = f.read()
                sub_files = {}
                sdir = os.path.join(pdir, sid, "subagents")
                if os.path.isdir(sdir):
                    for sf in sorted(os.listdir(sdir)):
                        if sf.endswith(".jsonl") or sf.endswith(".meta.json"):
                            try:
                                with open(os.path.join(sdir, sf), encoding="utf-8",
                                          errors="replace") as f:
                                    sub_files[sf] = f.read()
                            except OSError:
                                pass
                g = analyze_session_group(main_text, sub_files)
            except Exception as e:
                sess_rows.append({"project": d, "session_id": sid, "error": str(e)[:100]})
                continue
            m = g["main"]
            sess_rows.append({
                "project": d, "session_id": sid,
                "started_at": m["started_at"], "ended_at": m["ended_at"],
                "duration_s": m["duration_s"],
                "n_subagents_spawn": g["n_subagents_spawned"],
                "n_subagents_spawn_total": g["n_subagents_spawned_total"],
                "n_subagent_transcripts": g["n_subagent_transcripts"],
                "mismatch": int(g["spawn_transcript_mismatch"]),
                "main_out_tok": m["output_tokens"],
                "main_fin_ctx_tok": m["final_context_tokens"],
                "main_llm_calls": m["n_llm_calls"],
                "main_zero_usage_msgs": m["n_zero_usage_msgs"],
                "sub_out_tok_sum": sum(s["output_tokens"] or 0 for s in g["subagents"]),
                "all_out_tok": g["totals"]["output_tokens_all"],
                "end_reason": m["end_reason"],
            })
            for s in g["subagents"]:
                sub_rows.append({
                    "project": d, "session_id": sid, "agent_id": s["agent_id"],
                    "depth": s.get("spawn_depth"), "parent": s.get("parent_agent"),
                    "type": s.get("subagent_type"), "description": s.get("description"),
                    "started_at": s["started_at"], "duration_s": s["duration_s"],
                    "out_tok": s["output_tokens"], "fin_ctx_tok": s["final_context_tokens"],
                    "llm_calls": s["n_llm_calls"],
                    "zero_usage_msgs": s["n_zero_usage_msgs"],
                    "end_reason": s["end_reason"],
                })
    sess_f = os.path.join(out_dir, "sessions.csv")
    sub_f = os.path.join(out_dir, "subagents.csv")
    _write_csv(sess_f, sess_rows, ["project", "session_id", "started_at", "ended_at",
                "duration_s", "n_subagents_spawn", "n_subagents_spawn_total",
                "n_subagent_transcripts", "mismatch",
                "main_out_tok", "main_fin_ctx_tok", "main_llm_calls",
                "main_zero_usage_msgs", "sub_out_tok_sum", "all_out_tok",
                "end_reason", "error"])
    _write_csv(sub_f, sub_rows, ["project", "session_id", "agent_id", "depth", "parent",
                "type", "description", "started_at", "duration_s", "out_tok",
                "fin_ctx_tok", "llm_calls", "zero_usage_msgs", "end_reason"])
    _scan_summary(os.path.join(out_dir, "agent_scan_summary.md"), sess_rows, sub_rows, n_proj)
    return sess_rows, sub_rows


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _scan_summary(path, sess_rows, sub_rows, n_proj):
    ok = [r for r in sess_rows if not r.get("error")]
    n = len(ok) or 1
    avg = lambda k: round(sum(r.get(k) or 0 for r in ok) / n, 1)
    import statistics
    sub_counts = [r.get("n_subagent_transcripts") or 0 for r in ok]
    lines = ["# claude-code 转写智能体用量扫描汇总", "",
             f"- 项目目录: {n_proj}, 主 session: {len(sess_rows)} (有效 {len(ok)}),"
             f" 子代理转写: {len(sub_rows)}",
             f"- 子代理数/会话: mean={round(sum(sub_counts)/n, 2)}"
             f" median={statistics.median(sub_counts)} max={max(sub_counts) if sub_counts else 0}",
             f"- 主 agent: 平均耗时 {avg('duration_s')}s, 平均输出 {avg('main_out_tok')} tok,"
             f" 平均结束上下文 {avg('main_fin_ctx_tok')} tok, 平均 LLM 调用 {avg('main_llm_calls')}",
             f"- spawn数≠转写数 的会话: {sum(1 for r in ok if r.get('mismatch'))}",
             f"- 含零 usage 消息的会话: {sum(1 for r in ok if r.get('main_zero_usage_msgs'))}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="claude-code 转写 -> 智能体用量统计")
    ap.add_argument("src", nargs="?", help="tar.gz / 目录 / 项目目录 / 扁平化 jsonl 目录")
    ap.add_argument("--scan", metavar="PROJECTS_ROOT", help="遍历 projects 根批量扫描")
    ap.add_argument("--agg", metavar="RUN_DIR", help="批跑 RUN_DIR 汇总 (trajectories+verdicts+jsonl)")
    ap.add_argument("-o", "--out", help="--scan: 输出目录; 单源: 统计 json 路径")
    ap.add_argument("--md", action="store_true", help="单源模式额外打印 markdown 表")
    a = ap.parse_args(argv)
    if a.agg:
        csv_out = a.out or os.path.join(a.agg, "agent_stats.csv")
        rows = aggregate_run(a.agg, csv_out=csv_out,
                             md_out=os.path.join(a.agg, "agent_stats.md"))
        print(f"[agg] {len(rows)} jobs -> {csv_out} (+ agent_stats.md)")
        return 0
    if a.scan:
        out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(a.scan)),
                                        "agent_scan")
        sess_rows, _sub = scan_projects(a.scan, out_dir)
        print(f"[scan] {len(sess_rows)} sessions -> {out_dir}/sessions.csv"
              f" + subagents.csv + agent_scan_summary.md")
        return 0
    if not a.src:
        ap.error("需要 <src> 或 --scan PROJECTS_ROOT")
    stats = analyze_path(a.src)
    print(summarize_line(stats))
    if a.md and stats.get("primary"):
        p = stats["primary"]
        print("\n| agent | type | depth | dur_s | out_tok | fin_ctx_tok | llm_calls | end |")
        print("|---|---|---|---|---|---|---|---|")
        m = p["main"]
        print(f"| main | - | 0 | {m['duration_s']} | {m['output_tokens']}"
              f" | {m['final_context_tokens']} | {m['n_llm_calls']} | {m['end_reason']} |")
        for s in p["subagents"]:
            print(f"| {str(s['agent_id'])[:14]} | {s.get('subagent_type')} | {s.get('spawn_depth')}"
                  f" | {s['duration_s']} | {s['output_tokens']} | {s['final_context_tokens']}"
                  f" | {s['n_llm_calls']} | {s['end_reason']} |")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[out] -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
