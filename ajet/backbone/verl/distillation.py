# Copyright 2025 Alibaba Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
On-Policy Distillation (OPD) loss for AgentJet.

Ported from verl >=0.8.0 ``verl/trainer/distillation/losses.py`` (commit 455e44c6, PR #5041)
into the AgentJet override layer so we can stay on the pinned ``verl==0.7.1`` backend.

OPD = On-Policy Distillation. The student samples responses on-policy; a frozen **teacher**
scores those exact student tokens; the resulting divergence drives the update either as a
**supervised loss** (``use_policy_gradient=False``, arXiv:2306.13649) or as a **reward/advantage**
for a policy-gradient step (``use_policy_gradient=True``, advantage = -KL — the "Thinking Machines"
recipe https://thinkingmachines.ai/blog/on-policy-distillation/). It can be blended with the
normal task PPO loss via ``use_task_rewards`` + ``distillation_loss_coef``.

Two divergence families are supported (selected by ``loss_mode``):
  * **single-sample KL estimator** (``kl, k1, abs, mse, k2, low_var_kl, k3`` and the ``+``
    straight-through variants) — needs only the teacher's logprob of the *sampled* token,
    shape ``[B, T, 1]``. Reuses ``ajet.backbone.verl.core_algos.kl_penalty``. No logits needed.
  * **top-k forward KL** (``forward_kl_topk``) — needs the teacher's top-k ``(ids, logprobs)``,
    shape ``[B, T, K]``, plus the student logits (computed inside the forward pass via a
    logits-processor hook — see ``compute_forward_kl_topk``).

This module is self-contained: it owns the loss registry, the per-family loss functions, the
top-k logits-processor math, the unified dispatcher and the PPO-loss combiner.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from ajet.backbone.verl.core_algos import agg_loss, get_policy_loss_fn, kl_penalty

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# =====================================================================================
# Config (ajet-native; verl 0.7.1 has no DistillationConfig, so we define our own)
# =====================================================================================
@dataclass
class OpdLossConfig:
    """Runtime OPD loss config.

    Built from the ``ajet.opd`` block (+ master switch ``ajet.teacher_model.teacher_opd_enabled``)
    by :func:`opd_loss_config_from_ajet`. ``actor_config`` / ``loss_agg_mode`` are bound at runtime
    from the live actor config.

    loss_mode:
        Distillation divergence. Estimator family: ``kl|k1|abs|mse|k2|low_var_kl|k3`` (and ``+``
        straight-through: ``k1+|k3+``). Top-k: ``forward_kl_topk``.
    topk:
        Teacher top-k size. ``0`` (default) => estimator mode (teacher returns only the sampled
        token's logprob). ``>0`` => top-k forward KL (teacher returns top-k ids+logprobs).
    use_policy_gradient:
        True  => OPD-PG: use ``-KL`` as advantage in a vanilla PPO loss (the on-policy-distillation
                 reward signal). Recommended ``loss_mode=k1``.
        False => supervised: backprop the divergence directly. Recommended ``loss_mode=k3`` or
                 ``forward_kl_topk``.
    use_task_rewards:
        Blend with the task PPO loss. If False, the task policy loss is dropped and only the
        distillation objective is optimized.
    distillation_loss_coef:
        Weight of the (already-mode-applied) distillation loss when combined with the task loss.
    loss_max_clamp / log_prob_min_clamp:
        Numerical stability clamps.
    """

    enabled: bool = False
    loss_mode: str = "k3"
    topk: int = 0
    use_policy_gradient: bool = False
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0
    loss_agg_mode: str = "token-mean"
    # bound at runtime from the live ActorConfig
    actor_config: Any = None
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    @property
    def use_topk(self) -> bool:
        return self.loss_mode == "forward_kl_topk" or self.topk > 0

    def __post_init__(self):
        if self.use_policy_gradient and self.loss_mode in ("mse", "k2"):
            # k2 already gives the correct gradient; PG-on-k2 is redundant but not wrong.
            pass
        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            logger.warning(
                "forward_kl_topk is most effective as a supervised loss (use_policy_gradient=False). "
                "With PG, the update uses only the sampled token's ∇log π(a), so the top-k "
                "distributional signal is largely unused."
            )
        if (not self.use_policy_gradient) and self.loss_mode in ("k1", "kl"):
            raise ValueError(
                f"Directly backpropagating {self.loss_mode} is incorrect: the gradient wrt weights "
                f"does not depend on teacher logprobs. Use a '+' variant (e.g. k1+) or another mode."
            )


def opd_loss_config_from_ajet(ajet_cfg: Any, actor_config: Any) -> OpdLossConfig:
    """Build an :class:`OpdLossConfig` from the merged ajet config.

    Reads the master switch ``ajet.teacher_model.teacher_opd_enabled`` and the knobs under
    ``ajet.opd`` (with ``ajet.teacher_model.teacher_topk`` as the top-k source). Tolerates a
    plain dict or an OmegaConf object.
    """
    def get(obj, key, default=None):
        if obj is None:
            return default
        if hasattr(obj, "get"):  # dict-like / OmegaConf
            return obj.get(key, default)
        return getattr(obj, key, default)

    teacher = get(ajet_cfg, "teacher_model")
    opd = get(ajet_cfg, "opd")
    enabled = bool(get(teacher, "teacher_opd_enabled", False))

    cfg = OpdLossConfig(
        enabled=enabled,
        loss_mode=get(opd, "loss_mode", "k3"),
        topk=int(get(teacher, "teacher_topk", get(opd, "topk", 0)) or 0),
        use_policy_gradient=bool(get(opd, "use_policy_gradient", False)),
        use_task_rewards=bool(get(opd, "use_task_rewards", True)),
        distillation_loss_coef=float(get(opd, "distillation_loss_coef", 1.0)),
        loss_max_clamp=get(opd, "loss_max_clamp", 10.0),
        log_prob_min_clamp=get(opd, "log_prob_min_clamp", -10.0),
        loss_agg_mode=getattr(actor_config, "loss_agg_mode", "token-mean"),
        actor_config=actor_config,
    )
    # bind clip ratios from the live actor config when available
    cfg.clip_ratio = getattr(actor_config, "clip_ratio", 0.2)
    cfg.clip_ratio_low = getattr(actor_config, "clip_ratio_low", getattr(actor_config, "clip_ratio", 0.2))
    cfg.clip_ratio_high = getattr(actor_config, "clip_ratio_high", getattr(actor_config, "clip_ratio", 0.2))
    # propagate resolved topk into loss_mode if the user only set teacher_topk
    if cfg.topk > 0 and cfg.loss_mode != "forward_kl_topk":
        # estimator mode requested but teacher returns top-k: keep estimator (use sampled-token col)
        pass
    if cfg.loss_mode == "forward_kl_topk" and cfg.topk <= 0:
        cfg.topk = max(cfg.topk, 16)
    return cfg


# =====================================================================================
# Loss registry
# =====================================================================================
# family is one of: "estimator" (needs only sampled-token teacher logprob) | "topk" (needs logits)
OpdLossFn = Callable[[OpdLossConfig, dict, dict], tuple[torch.Tensor, dict]]
_OPD_LOSS_REGISTRY: dict[str, OpdLossFn] = {}
_OPD_FAMILY_REGISTRY: dict[str, str] = {}


def _register(names, family: str):
    if isinstance(names, str):
        names = [names]

    def deco(fn):
        for n in names:
            if n in _OPD_LOSS_REGISTRY:
                raise ValueError(f"OPD loss '{n}' already registered")
            _OPD_LOSS_REGISTRY[n] = fn
            _OPD_FAMILY_REGISTRY[n] = family
        return fn

    return deco


def get_opd_family(loss_mode: str) -> str:
    if loss_mode not in _OPD_FAMILY_REGISTRY:
        raise ValueError(
            f"Unsupported OPD loss_mode {loss_mode!r}. Supported: {sorted(_OPD_LOSS_REGISTRY)}"
        )
    return _OPD_FAMILY_REGISTRY[loss_mode]


def supported_loss_modes() -> list[str]:
    return sorted(_OPD_LOSS_REGISTRY)


# =====================================================================================
# Estimator family (kl / k1 / abs / mse / k2 / low_var_kl / k3 + '+' variants)
# Needs: student log_prob [B,T] + teacher logprob of the sampled token [B,T,1]
# =====================================================================================
@_register(["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3", "k1+", "k3+", "kl+"], "estimator")
def _estimator_distillation_loss(cfg: OpdLossConfig, model_output: dict, data: dict):
    """Single-sample KL estimator between student and teacher over the sampled tokens.

    model_output["log_probs"]: [B, T] student logprob of each response token.
    data["teacher_logprobs"]:  [B, T, 1] (or [B, T]) teacher logprob of the SAME sampled token.
    """
    student_log_probs = model_output["log_probs"]
    teacher_log_probs = data["teacher_logprobs"]
    if teacher_log_probs.dim() == student_log_probs.dim() + 1:
        teacher_log_probs = teacher_log_probs.squeeze(-1)
    assert teacher_log_probs.shape == student_log_probs.shape, (
        f"teacher_logprobs {teacher_log_probs.shape} != student log_probs {student_log_probs.shape}"
    )

    # NOTE: verl 0.7.1's ``kl_penalty_forward`` does NOT strip the ``+`` suffix, so passing
    # ``k1+``/``k3+``/``kl+`` directly raises NotImplementedError. We strip it ourselves and apply
    # the straight-through estimator: forward = base estimator (correct KL expectation), backward
    # via 0.5*(logp-reflogp)^2 (correct gradient direction) — mirroring verl's ``kl_penalty`` '+'
    # handling without depending on it.
    base = cfg.loss_mode[:-1] if cfg.loss_mode.endswith("+") else cfg.loss_mode
    forward_score = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=base
    )
    if cfg.loss_mode.endswith("+") and base not in ("mse", "k2"):
        backward_score = 0.5 * (student_log_probs - teacher_log_probs).square()
        distillation_losses = backward_score - backward_score.detach() + forward_score.detach()
    else:
        distillation_losses = forward_score
    response_mask = data["response_mask"].bool()
    valid = distillation_losses[response_mask]
    metrics = {
        "opd/abs_loss": valid.abs().mean().item() if valid.numel() else 0.0,
        "opd/raw_loss_mean": valid.mean().item() if valid.numel() else 0.0,
    }
    return distillation_losses, metrics


# =====================================================================================
# Top-k forward KL (loss_mode="forward_kl_topk")
# Needs: student logits [B,T,V] (via logits-processor hook) + teacher top-k (ids[B,T,K], lp[B,T,K])
# =====================================================================================
def _chunked_topk_log_probs(logits: torch.Tensor, topk_ids: torch.Tensor, chunk_size: int = 4096):
    """log_softmax(logits).gather(topk_ids) without materializing [B,T,V] log_softmax buffer."""
    B, T, V = logits.shape
    K = topk_ids.shape[-1]
    flat_logits = logits.reshape(-1, V)
    flat_topk = topk_ids.reshape(-1, K)
    N = flat_logits.shape[0]
    if N == 0:
        return torch.empty((B, T, K), dtype=logits.dtype, device=logits.device)
    out = torch.empty((N, K), dtype=logits.dtype, device=logits.device)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        chunk = flat_logits[s:e].float()
        log_z = torch.logsumexp(chunk, dim=-1, keepdim=True)
        out[s:e] = (torch.gather(chunk, -1, flat_topk[s:e]) - log_z).to(logits.dtype)
    return out.reshape(B, T, K)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    cfg: OpdLossConfig,
    response_mask: torch.Tensor,
) -> dict:
    """Forward-KL distillation between the student distribution and the teacher's top-k support.

    Called as a *logits processor* inside the student forward (see ``AjetDataParallelPPOActor``)
    while the response-token logits are materialized. Returns per-token quantities; aggregation
    into a scalar happens later via :func:`compute_distillation_loss`.

    All tensors are dense (padded), shape ``[B, T, ...]`` (T = response length). For
    ``ulysses_sequence_parallel_size > 1`` the caller is responsible for slicing ``student_logits``
    and the teacher tensors along T before calling.
    """
    # student logprob at the teacher's top-k ids
    if getattr(cfg, "use_chunked_topk", False):
        student_topk_log_probs = _chunked_topk_log_probs(
            student_logits, teacher_topk_ids, chunk_size=getattr(cfg, "chunked_topk_chunk_size", 4096)
        )
    else:
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids).to(student_logits.dtype)

    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    if cfg.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(cfg.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(cfg.log_prob_min_clamp)

    # KL(teacher || student) restricted to the teacher's top-k support
    t = teacher_topk_log_probs.float()
    s = student_topk_log_probs.float()
    distillation_losses = (t.exp() * (t - s)).sum(dim=-1)

    # diagnostics: teacher/student top-k overlap (per "Rethinking OPD", arXiv:2604.13016)
    with torch.no_grad():
        student_topk_ids = torch.topk(student_logits, k=teacher_topk_ids.shape[-1], dim=-1).indices
        overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
        overlap_count = overlap_mask.sum(dim=-1)
        rm = response_mask.bool()
        if rm.any():
            overlap_ratio = (overlap_count[rm].float().mean() / teacher_topk_ids.shape[-1]).item()
        else:
            overlap_ratio = 0.0

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_ratio": overlap_ratio,
    }


@_register(["forward_kl_topk"], "topk")
def _topk_distillation_loss(cfg: OpdLossConfig, model_output: dict, data: dict):
    """Top-k forward KL. The per-token loss is expected to have been precomputed in the logits
    processor and stashed in ``model_output["distillation_losses"]`` (see ``compute_forward_kl_topk``)."""
    distillation_losses = model_output["distillation_losses"]
    response_mask = data["response_mask"].bool()
    rm = response_mask
    metrics = {
        "opd/student_mass": model_output["student_mass"][rm].mean().item() if rm.any() else 0.0,
        "opd/teacher_mass": model_output["teacher_mass"][rm].mean().item() if rm.any() else 0.0,
        "opd/overlap_ratio": float(model_output.get("overlap_ratio", 0.0)),
    }
    # top-k distributions don't sum to 1 => divergence can be slightly negative
    distillation_losses = distillation_losses.clamp_min(0.0)
    return distillation_losses, metrics


# =====================================================================================
# Unified dispatcher: divergence -> (scalar distillation loss, metrics)
# =====================================================================================
def compute_distillation_loss(
    cfg: OpdLossConfig,
    model_output: dict,
    data: dict,
) -> tuple[torch.Tensor, dict]:
    """Compute the scalar distillation loss under the configured gradient mode.

    model_output: {"log_probs": [B,T], optionally "distillation_losses","student_mass",...}
    data:         {"teacher_logprobs":[B,T,1], "teacher_ids":[B,T,K], "response_mask":[B,T],
                   "old_log_probs":[B,T], optional "rollout_is_weights":[B,T]}

    Returns (distill_loss_scalar, metrics). The scalar already encodes the gradient mode:
      * supervised => agg_loss(divergence, mask)
      * PG         => vanilla PPO loss with advantage = -divergence.detach()
    """
    assert cfg.actor_config is not None, "OpdLossConfig.actor_config must be bound at runtime"
    response_mask = data["response_mask"]
    if response_mask.is_floating_point():
        response_mask_bool = response_mask.bool()
    else:
        response_mask_bool = response_mask.bool()

    loss_fn = _OPD_LOSS_REGISTRY[cfg.loss_mode]
    distillation_losses, distill_metrics = loss_fn(cfg, model_output, data)

    if cfg.loss_max_clamp is not None:
        # k1 can be negative; clamp symmetrically
        distillation_losses = distillation_losses.clamp(
            min=-cfg.loss_max_clamp, max=cfg.loss_max_clamp
        )

    if cfg.use_policy_gradient:
        # OPD-PG: use the (negated) divergence as the advantage in a vanilla PPO loss.
        policy_loss_fn = get_policy_loss_fn("vanilla")
        old_log_prob = data["old_log_probs"]
        rollout_is_weights = data.get("rollout_is_weights", None)
        # Build a lightweight actor-config view that exposes the clip knobs for vanilla PPO.
        actor_cfg = _ActorClipView(cfg, cfg.actor_config)
        distill_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=model_output["log_probs"],
            advantages=-distillation_losses.detach(),
            response_mask=response_mask_bool,
            loss_agg_mode=cfg.loss_agg_mode,
            config=actor_cfg,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"opd/{k}": v for k, v in pg_metrics.items()}
        distill_metrics.update(pg_metrics)
    else:
        # Supervised: directly backprop the aggregated divergence.
        distill_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask_bool,
            loss_agg_mode=cfg.loss_agg_mode,
        )

    distill_metrics["opd/distill_loss"] = distill_loss.detach().item() if torch.is_tensor(distill_loss) else float(distill_loss)
    return distill_loss, distill_metrics


class _ActorClipView:
    """Config view for ``compute_policy_loss_vanilla``: delegates everything (``.get``,
    ``global_batch_info``, entropy knobs, ...) to the real actor config, but overrides the
    PPO clip ratios with the OPD-resolved ones.
    """

    def __init__(self, cfg: OpdLossConfig, actor_config: Any):
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_base", actor_config)

    @property
    def clip_ratio(self):
        return self._cfg.clip_ratio

    @property
    def clip_ratio_low(self):
        return self._cfg.clip_ratio_low

    @property
    def clip_ratio_high(self):
        return self._cfg.clip_ratio_high

    def __getattr__(self, name):
        # Any attribute not defined here (get, global_batch_info, kl_loss_coef, ...) -> real actor config.
        return getattr(self._base, name)


def opd_loss_config_from_meta(meta: dict, actor_config: Any) -> OpdLossConfig:
    """Build an :class:`OpdLossConfig` from a flat dict carried in ``DataProto.meta_info["opd"]``.

    The trainer serializes the resolved OPD knobs into ``batch.meta_info["opd"]`` so they reach the
    Ray worker actors (which only see the ``actor_rollout_ref`` subtree, not the ``ajet`` namespace).
    """
    if not meta:
        return OpdLossConfig(enabled=False, actor_config=actor_config)
    cfg = OpdLossConfig(
        enabled=bool(meta.get("enabled", False)),
        loss_mode=meta.get("loss_mode", "k3"),
        topk=int(meta.get("topk", 0) or 0),
        use_policy_gradient=bool(meta.get("use_policy_gradient", False)),
        use_task_rewards=bool(meta.get("use_task_rewards", True)),
        distillation_loss_coef=float(meta.get("distillation_loss_coef", 1.0)),
        loss_max_clamp=meta.get("loss_max_clamp", 10.0),
        log_prob_min_clamp=meta.get("log_prob_min_clamp", -10.0),
        loss_agg_mode=getattr(actor_config, "loss_agg_mode", "token-mean"),
        actor_config=actor_config,
    )
    cfg.clip_ratio = getattr(actor_config, "clip_ratio", 0.2)
    cfg.clip_ratio_low = getattr(actor_config, "clip_ratio_low", getattr(actor_config, "clip_ratio", 0.2))
    cfg.clip_ratio_high = getattr(actor_config, "clip_ratio_high", getattr(actor_config, "clip_ratio", 0.2))
    if cfg.loss_mode == "forward_kl_topk" and cfg.topk <= 0:
        cfg.topk = 16
    return cfg


def build_opd_meta(ajet_cfg: Any) -> Optional[dict]:
    """Build a plain (JSON-serializable) OPD meta dict for ``DataProto.meta_info["opd"]``.

    Returns None when OPD is disabled. The trainer puts this on the batch so the Ray worker
    actors can rebuild an :class:`OpdLossConfig` via :func:`opd_loss_config_from_meta` (they only
    receive the actor_rollout_ref subtree, not the ajet namespace).
    """
    teacher = getattr(ajet_cfg, "teacher_model", None) if not hasattr(ajet_cfg, "get") else ajet_cfg.get("teacher_model")
    if teacher is None:
        return None

    def g(obj, key, default=None):
        if obj is None:
            return default
        if hasattr(obj, "get"):
            return obj.get(key, default)
        return getattr(obj, key, default)

    if not bool(g(teacher, "teacher_opd_enabled", False)):
        return None
    opd = g(ajet_cfg, "opd")
    topk_teacher = int(g(teacher, "teacher_topk", 0) or 0)
    topk_opd = int(g(opd, "topk", 0) or 0)
    loss_mode = g(opd, "loss_mode", "k3")
    topk = topk_teacher or topk_opd
    if loss_mode == "forward_kl_topk" and topk <= 0:
        topk = 16
    if loss_mode == "simct" and topk <= 0:
        topk = int(g(opd, "simct_topk", 20) or 20)  # SimCT overlap-candidate top-K
    return {
        "enabled": True,
        "loss_mode": loss_mode,
        "topk": topk,
        "use_policy_gradient": bool(g(opd, "use_policy_gradient", False)),
        "use_task_rewards": bool(g(opd, "use_task_rewards", True)),
        "distillation_loss_coef": float(g(opd, "distillation_loss_coef", 1.0)),
        "loss_max_clamp": g(opd, "loss_max_clamp", 10.0),
        "log_prob_min_clamp": g(opd, "log_prob_min_clamp", -10.0),
    }


def is_opd_enabled(ajet_cfg: Any) -> bool:
    """Master switch check from the merged ajet config."""
    teacher = getattr(ajet_cfg, "teacher_model", None) if not hasattr(ajet_cfg, "get") else ajet_cfg.get("teacher_model")
    if teacher is None:
        return False
    if hasattr(teacher, "get"):
        return bool(teacher.get("teacher_opd_enabled", False))
    return bool(getattr(teacher, "teacher_opd_enabled", False))
