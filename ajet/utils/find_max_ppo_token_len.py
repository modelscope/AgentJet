# -*- coding: utf-8 -*-
"""[AJET] ppo_max_token_len_per_gpu limit finder.

Extracted from ``AjetDataParallelPPOActor`` to keep ``dp_actor.py`` lean. Enabled by the env var
``AGENTJET_FIND_MAX_PPO_TOKEN_LEN``; ``update_policy`` calls
``find_max_ppo_token_len_per_gpu(self, data)`` on the *first* PPO update and the call never returns
(it always raises ``RuntimeError`` to terminate the run with the discovered value).
"""

import gc
import math
import os
import time as _time

import torch
import torch.distributed as dist

from verl.utils.device import get_device_id
from ajet.tuner_lib.experimental.interchange_utils import http_push_verbose_log


def find_max_ppo_token_len_per_gpu(actor, data):
    """Estimate the largest ``ppo_max_token_len_per_gpu`` that fits on this GPU, then raise.

    Mechanism (enabled by env ``AGENTJET_FIND_MAX_PPO_TOKEN_LEN``):

    * For a per-GPU token budget ``L`` we synthesise a *worst-case* PPO micro-batch holding
      ``L`` valid (non-pad) tokens, then run a real ``forward + backward + optimizer_step`` on
      ``actor`` — the same code path & dominant memory (the ``[total_nnz, vocab]`` logits tensor)
      as a real update — and record the peak GPU memory.
    * Peak memory grows ~linearly with ``L``. We probe only *safe* sizes (ramping until the peak
      reaches ``SAFE_FRAC`` of the card), least-squares-fit the ``peak_GB(L)`` line, and
      **extrapolate** to the OOM ceiling (``CEIL_FRAC`` of the card) to predict the maximum ``L``
      that fits. We deliberately *avoid* triggering OOM: under FSDP + param/optimizer offload a
      caught OOM tends to leave the worker unrecoverable, so we never push past the safe zone (if
      an OOM does occur it is used only as a hard upper bound and the search stops immediately).
    * Every probe is identical across ranks (``L`` derives from config, not data) and the OK/peak
      verdict is all-reduced, so FSDP collectives stay in lock-step.

    Tunable via env vars (all optional):
      ``AGENTJET_FIND_MAX_START``      initial probe size (default: current value)
      ``AGENTJET_FIND_MAX_CAP``        hard upper bound on probe size (default: 16x current)
      ``AGENTJET_FIND_MAX_BUDGET_S``   wall-clock budget in seconds (default: 600)
      ``AGENTJET_FIND_MAX_SEQ``        cap on a single synthetic sequence (default: current value)
      ``AGENTJET_FIND_MAX_SAFE_FRAC``  stop ramping at this mem fraction (default: 0.80)
      ``AGENTJET_FIND_MAX_CEIL_FRAC``  extrapolated OOM ceiling fraction (default: 0.92)
      ``AGENTJET_FIND_MAX_MARGIN``     head-room applied to the result (default: 0.90)

    Args:
        actor: the ``AjetDataParallelPPOActor`` instance (provides ``config``, ``scaler``,
            ``ulysses_sequence_parallel_size``, ``actor_optimizer``, ``_forward_micro_batch``,
            ``_optimizer_step``).
        data: the ``DataProto`` of the intercepted update (only ``meta_info`` is used).
    """
    if not actor.config.use_dynamic_bsz:
        raise RuntimeError(
            "[FIND_MAX_PPO_TOKEN_LEN] requires actor.use_dynamic_bsz=True "
            "(ppo_max_token_len_per_gpu only governs dynamic batching)."
        )

    device = get_device_id()
    usp = max(1, int(actor.ulysses_sequence_parallel_size))
    temperature = data.meta_info["temperature"]
    pad_token_id = data.meta_info.get("pad_token_id", 0)
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1

    def _envint(name, default):
        v = os.environ.get(name)
        try:
            return int(v) if v not in (None, "") else int(default)
        except ValueError:
            return int(default)

    cur = int(actor.config.ppo_max_token_len_per_gpu)
    start = max(512, _envint("AGENTJET_FIND_MAX_START", max(1024, cur)))
    hard_cap = _envint("AGENTJET_FIND_MAX_CAP", cur * 16)
    budget_s = _envint("AGENTJET_FIND_MAX_BUDGET_S", 600)
    max_seq_cap = max(8, _envint("AGENTJET_FIND_MAX_SEQ", cur))
    calc_entropy = bool(actor.config.calculate_entropy or (actor.config.entropy_coeff != 0))

    t0 = _time.time()
    trials = []  # list of (L, ok, peak_gb)

    def _log(msg):
        line = f"[FIND_MAX_PPO_TOKEN_LEN][rank{rank}] {msg}"
        print(line, flush=True)
        if rank == 0:
            try:
                http_push_verbose_log(line, tag="find_max_ppo_token_len")
            except Exception:
                pass

    def _build_micro_batch(total_tokens: int) -> dict:
        # `_forward_micro_batch` consumes a plain dict of tensors; pack `total_tokens` valid
        # (non-pad) tokens into B fully-attended sequences each <= max_seq_cap.
        total_tokens = max(8, int(total_tokens))
        bsz = max(1, math.ceil(total_tokens / max_seq_cap))
        seq_len = max(8, math.ceil(total_tokens / bsz))  # <= max_seq_cap
        ids = torch.full((bsz, seq_len), 1, dtype=torch.long, device=device)
        attn = torch.ones((bsz, seq_len), dtype=torch.long, device=device)
        pos = (
            torch.arange(seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(bsz, -1)
            .contiguous()
        )
        resp_len = max(1, seq_len // 2)
        responses = ids[:, -resp_len:].contiguous()
        response_mask = torch.ones((bsz, resp_len), dtype=torch.long, device=device)
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "position_ids": pos,
            "responses": responses,
            "response_mask": response_mask,
        }

    def _probe(L: int):
        # Run ONE real forward+backward+optimizer_step on a synthetic micro-batch of L tok/gpu
        # and return (global_ok, peak_gb). With ulysses SP the forward slices the sequence
        # across `usp` ranks, so the synthetic batch holds L*usp tokens.
        total_tokens = int(L) * usp
        outputs = loss = mb = model_inputs = None
        local_ok = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            actor.actor_optimizer.zero_grad(set_to_none=True)
            mb = _build_micro_batch(total_tokens)
            model_inputs = {**mb, "pad_token_id": pad_token_id}
            outputs = actor._forward_micro_batch(
                model_inputs, temperature=temperature, calculate_entropy=calc_entropy
            )
            log_prob = outputs["log_probs"]
            # Surrogate loss over the full forward graph => realistic backward memory.
            loss = (log_prob * model_inputs["response_mask"].to(log_prob.dtype)).sum()
            if actor.scaler is not None:
                actor.scaler.scale(loss).backward()
            else:
                loss.backward()
            actor._optimizer_step()
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            local_ok = False
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA error" in str(e):
                local_ok = False
            else:
                raise
        finally:
            outputs = loss = mb = model_inputs = None
            actor.actor_optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        global_ok = local_ok
        if distributed:
            okt = torch.tensor([1.0 if local_ok else 0.0], device=device)
            dist.all_reduce(okt, op=dist.ReduceOp.MIN)
            global_ok = okt.item() > 0.5
            pkt = torch.tensor([peak_gb], device=device)
            dist.all_reduce(pkt, op=dist.ReduceOp.MAX)
            peak_gb = float(pkt.item())
        trials.append((int(L), global_ok, peak_gb))
        _log(
            f"L={L} tok/gpu (synth_total={total_tokens}) -> "
            f"{'OK ' if global_ok else 'OOM'} peak={peak_gb:.2f}GB"
        )
        return global_ok, peak_gb

    def _fit(samples):
        # least-squares line peak_gb ~= a + b * L over OK samples; returns (a, b)
        n = len(samples)
        sx = sum(s[0] for s in samples)
        sy = sum(s[1] for s in samples)
        sxx = sum(s[0] * s[0] for s in samples)
        sxy = sum(s[0] * s[1] for s in samples)
        denom = n * sxx - sx * sx
        if denom == 0:
            return (sy / n, 0.0)
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        return (a, b)

    def _budget_left():
        return (_time.time() - t0) < budget_s

    # Memory model: peak GPU mem grows ~linearly with per-gpu token count L (logits[L,vocab]
    # + checkpointed activations + constant model/optim state). We probe only SAFE sizes,
    # fit the line, and EXTRAPOLATE to the GPU ceiling -- deliberately triggering OOM is
    # avoided because a caught OOM under FSDP+offload tends to leave the worker unrecoverable.
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    safe_frac = float(os.environ.get("AGENTJET_FIND_MAX_SAFE_FRAC", "0.80"))   # stop ramping here
    ceil_frac = float(os.environ.get("AGENTJET_FIND_MAX_CEIL_FRAC", "0.92"))   # OOM ceiling estimate
    margin = float(os.environ.get("AGENTJET_FIND_MAX_MARGIN", "0.90"))         # head-room on result
    safe_gb = safe_frac * total_gb
    ceil_gb = ceil_frac * total_gb

    _log(
        f"start={start} cap={hard_cap} budget={budget_s}s usp={usp} world={world} "
        f"cur_value={cur} max_seq_cap={max_seq_cap} total_mem={total_gb:.1f}GB "
        f"safe={safe_gb:.1f}GB ceil={ceil_gb:.1f}GB margin={margin}"
    )

    samples = []   # OK (L, peak_gb)
    oom_L = None
    L = max(512, min(start, hard_cap))
    while _budget_left():
        ok, peak = _probe(L)
        if not ok:
            oom_L = L            # unexpected OOM: hard upper bound, stop (no further fwd).
            break
        samples.append((L, peak))
        if peak >= safe_gb or L >= hard_cap:
            break               # close enough to the ceiling; stop before risking OOM.
        # choose next L
        if len(samples) >= 2:
            a, b = _fit(samples)
            nxt = int((safe_gb - a) / b) if b > 1e-12 else L * 2
            nxt = min(nxt, L * 2, hard_cap)           # never more than double per step
            # guard: keep predicted peak strictly below the ceiling to avoid OOM
            while b > 1e-12 and (a + b * nxt) > ceil_gb and nxt > L:
                nxt = (nxt + L) // 2
            # diminishing returns: once the next safe step is <2% bigger, the line is
            # well-determined -> stop ramping and extrapolate (avoids wasting the budget
            # inching toward the safe ceiling).
            if nxt <= L or (nxt - L) < max(256, int(0.02 * L)):
                break
            L = nxt
        else:
            L = min(L * 2, hard_cap)

    elapsed = _time.time() - t0
    table = "\n".join(
        f"    L={Lp:>9d} tok/gpu  {'OK ' if okp else 'OOM'}  peak={pg:.2f} GB"
        for (Lp, okp, pg) in trials
    )

    if len(samples) < 1:
        raise RuntimeError(
            "[FIND_MAX_PPO_TOKEN_LEN] FAILED: no safe probe succeeded (even the smallest OOM'd). "
            f"Reduce max_model_len / model size.\nProbes:\n{table}"
        )

    # Extrapolate the fitted line to the OOM ceiling => predicted max L that fits.
    if len(samples) >= 2:
        a, b = _fit(samples)
        if b > 1e-12:
            L_ceiling = int((ceil_gb - a) / b)
        else:
            L_ceiling = samples[-1][0]
        fit_desc = f"peak_GB ~= {a:.2f} + {b*1000:.4f}*(L/1000)"
    else:
        # single point: assume proportional through origin
        L0, p0 = samples[0]
        L_ceiling = int(L0 * ceil_gb / max(p0, 1e-6))
        fit_desc = f"single-point proportional from (L={L0}, {p0:.1f}GB)"

    if oom_L is not None:
        L_ceiling = min(L_ceiling, int(oom_L * 0.97))
    L_ceiling = max(L_ceiling, samples[-1][0])
    recommended = int(L_ceiling * margin)

    raise RuntimeError(
        "\n==================== FIND_MAX_PPO_TOKEN_LEN RESULT ====================\n"
        f"  Predicted MAX ppo_max_token_len_per_gpu (peak hits {ceil_frac:.0%} of "
        f"{total_gb:.0f}GB) ~= {L_ceiling} tok/gpu\n"
        f"  Recommended (x{margin} head-room): ajet.rollout.ppo_max_token_len_per_gpu = {recommended}\n"
        f"  Largest safely-measured probe: {samples[-1][0]} tok/gpu @ {samples[-1][1]:.1f}GB"
        + (f"; observed OOM at {oom_L} tok/gpu\n" if oom_L else "\n")
        + f"  Fit: {fit_desc}\n"
        f"  (current value was {cur}; world_size={world}, ulysses_sp={usp}, elapsed={elapsed:.1f}s)\n"
        f"  Probe log:\n{table}\n"
        "  >> Set ajet.rollout.ppo_max_token_len_per_gpu to the recommended value and rerun "
        "WITHOUT AGENTJET_FIND_MAX_PPO_TOKEN_LEN.\n"
        "======================================================================="
    )
