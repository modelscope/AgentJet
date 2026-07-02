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
SimCT cross-tokenizer OPD loss — the scoring + loss half (companion to ``simct_align.py``).

Bridges the gap between the SimCT paper/repo (which assumes a WHITE-BOX teacher with full
``teacher_lm_head`` logits) and our setting where the teacher is a REMOTE vLLM server exposing
only top-K ``prompt_logprobs``.

Common supervision space per sample = [overlap candidates | spans] (a "virtual common vocab"),
exactly as in the SimCT repo ``span_ctkd._build_virtual_vocab_logits``:
  * overlap candidates: the teacher's top-K tokens at the segments' first positions, intersected
    with the shared vocabulary (we can only score what the remote teacher returns; the student is
    white-box so it gathers its own logits at the same candidate ids).
  * spans: the multi-token minimal aligned units from ``simct_align``.

Teacher virtual logits are built from ``prompt_logprobs`` (top-K dicts); student virtual logits
from white-box student logits. Reverse-KL is then taken over the virtual vocab, per segment.

This module is CPU-testable: the teacher side consumes plain ``prompt_logprobs`` dicts (as returned
by vLLM), the student side consumes a student logit matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ajet.backbone.verl.simct_align import Alignment, align_sequences_with_spans, find_overlap_tokens

_SEG = Tuple[int, int, int, int]  # (tea_start, tea_end, stu_start, stu_end)


def build_shared_id_map(tok_student, tok_teacher) -> Dict[int, int]:
    """teacher_token_id -> student_token_id for every shared (same decoded text) token."""
    norm = lambda s: s.replace("Ġ", "▁")
    stu_vocab = {norm(k): v for k, v in tok_student.get_vocab().items()}
    tea_vocab = {norm(k): v for k, v in tok_teacher.get_vocab().items()}
    shared = set(stu_vocab.keys()) & set(tea_vocab.keys())
    return {tea_vocab[t]: stu_vocab[t] for t in shared}


@dataclass
class SimCTSampleSpec:
    """Per-sample SimCT supervision spec.

    Everything the student (actor) needs to build its virtual logits and run reverse-KL against the
    precomputed teacher virtual logits. Produced driver-side from the remote teacher's prompt_logprobs.
    """
    segments: List[_SEG]
    num_segments: int
    # overlap candidate ids (per-sample virtual vocab dims 0..num_candidates-1)
    candidate_tea_ids: List[int]
    candidate_stu_ids: List[int]
    num_candidates: int
    # spans (virtual vocab dims num_candidates..num_candidates+num_spans-1), in span-dim order
    span_seg_indices: List[int]            # segment index of each span dim
    seg_to_span_dim: Dict[int, int]        # segment index -> its span dim (only for span segments)
    # student-side pointers (indices into the STUDENT tokenization of the rollout, NOT input_ids)
    seg_first_stu_pos: List[int]           # per segment: student first-token position (ss)
    span_stu_constituents: List[List[Tuple[int, int]]]  # per span dim: [(stu_pos, stu_token_id), ...]
    # teacher virtual logits (precomputed, detached) — [num_segments, num_candidates + num_spans]
    teacher_virtual: torch.Tensor

    @property
    def num_spans(self) -> int:
        return len(self.span_seg_indices)

    @property
    def virtual_dim(self) -> int:
        return self.num_candidates + self.num_spans


def _teacher_self_logprob(tea_prompt_logprobs: List[Optional[dict]], pos: int, tok_id: int, clamp: float) -> float:
    """Teacher's logprob of the actual token ``tok_id`` at position ``pos`` (vLLM includes the actual
    prompt token in prompt_logprobs). Returns ``clamp`` if missing."""
    entry = tea_prompt_logprobs[pos] if pos < len(tea_prompt_logprobs) else None
    if not entry:
        return clamp
    # entry keyed by token_id string (vLLM) or int
    info = entry.get(str(tok_id)) or entry.get(tok_id)
    if not isinstance(info, dict):
        return clamp
    return float(info.get("logprob", clamp))


def build_teacher_simct_spec(
    alignment: Alignment,
    tea_prompt_logprobs: List[Optional[dict]],
    shared_tea_to_stu: Dict[int, int],
    topk: int,
    log_prob_min_clamp: float = -10.0,
) -> Optional[SimCTSampleSpec]:
    """Build the per-sample SimCT spec + teacher virtual logits from the remote teacher's
    ``prompt_logprobs`` (one dict per teacher-token position, as vLLM returns).

    The overlap candidate set = union over segment first-positions of (teacher top-K ids ∩ shared).
    Teacher virtual logits per segment:
      * overlap cols: teacher logprob of each candidate at the segment's first teacher position
        (if returned by the teacher, else ``log_prob_min_clamp``).
      * span cols: ``-1e9`` except the segment's own span dim = mean teacher self-logprob over the
        span's constituent teacher tokens (exact — those tokens are in the teacher-tokenized rollout).
    """
    segments = alignment.segments
    if not segments:
        return None

    # 1) overlap candidates = union of teacher top-K ∩ shared across segment first teacher positions
    seen = {}
    for (ts, te, ss, se) in segments:
        entry = tea_prompt_logprobs[ts] if ts < len(tea_prompt_logprobs) else None
        if not entry:
            continue
        # collect candidate teacher ids in this position's top-K that are shared
        for k_str, info in entry.items():
            try:
                tid = int(k_str)
            except (ValueError, TypeError):
                continue
            if tid in shared_tea_to_stu and tid not in seen:
                seen[tid] = shared_tea_to_stu[tid]
    candidate_tea_ids = list(seen.keys())
    candidate_stu_ids = [seen[t] for t in candidate_tea_ids]
    cand_index = {t: i for i, t in enumerate(candidate_tea_ids)}
    num_candidates = len(candidate_tea_ids)

    # 2) spans
    span_seg_indices = [i for i, (ts, te, ss, se) in enumerate(segments) if (te - ts) > 1 or (se - ss) > 1]
    seg_to_span_dim = {seg_i: d for d, seg_i in enumerate(span_seg_indices)}
    num_spans = len(span_seg_indices)
    virtual_dim = num_candidates + num_spans

    # student-side pointers
    seg_first_stu_pos = [ss for (ts, te, ss, se) in segments]
    span_stu_constituents: List[List[Tuple[int, int]]] = []
    for seg_i in span_seg_indices:
        ts, te, ss, se = segments[seg_i]
        span_stu_constituents.append([(ss + k, alignment.stu_ids[ss + k]) for k in range(se - ss)])

    # 3) teacher virtual logits [num_segments, virtual_dim]
    teacher_virtual = torch.full((len(segments), virtual_dim), float(log_prob_min_clamp), dtype=torch.float32)
    for ri, (ts, te, ss, se) in enumerate(segments):
        entry = tea_prompt_logprobs[ts] if ts < len(tea_prompt_logprobs) else None
        if entry:
            for k_str, info in entry.items():
                try:
                    tid = int(k_str)
                except (ValueError, TypeError):
                    continue
                ci = cand_index.get(tid)
                if ci is not None and isinstance(info, dict):
                    teacher_virtual[ri, ci] = float(info.get("logprob", log_prob_min_clamp))
        # span dims
        for seg_i, dim in seg_to_span_dim.items():
            col = num_candidates + dim
            if seg_i == ri:
                ts2, te2, ss2, se2 = segments[seg_i]
                self_lps = [_teacher_self_logprob(tea_prompt_logprobs, ts2 + k, alignment.tea_ids[ts2 + k], log_prob_min_clamp)
                            for k in range(te2 - ts2)]
                teacher_virtual[ri, col] = sum(self_lps) / max(len(self_lps), 1)
            else:
                teacher_virtual[ri, col] = -1e9

    return SimCTSampleSpec(
        segments=segments,
        num_segments=len(segments),
        candidate_tea_ids=candidate_tea_ids,
        candidate_stu_ids=candidate_stu_ids,
        num_candidates=num_candidates,
        span_seg_indices=span_seg_indices,
        seg_to_span_dim=seg_to_span_dim,
        seg_first_stu_pos=seg_first_stu_pos,
        span_stu_constituents=span_stu_constituents,
        teacher_virtual=teacher_virtual,
    )


def student_virtual_logits(spec: SimCTSampleSpec, stu_logits: torch.Tensor) -> torch.Tensor:
    """Build the student's virtual-vocab logits from white-box student logits.

    Args:
        spec: per-sample spec (defines the virtual vocab + student-side pointers).
        stu_logits: ``[T_student, V_student]`` student logits at the rollout positions (white-box).
    Returns:
        ``[num_segments, num_candidates + num_spans]`` student virtual logits.
    """
    device = stu_logits.device
    dtype = stu_logits.dtype
    V = spec.num_candidates + spec.num_spans
    out = torch.full((spec.num_segments, V), -1e9, device=device, dtype=dtype)

    if spec.num_candidates > 0:
        cand_stu = torch.tensor(spec.candidate_stu_ids, device=device, dtype=torch.long)
        for ri, first_pos in enumerate(spec.seg_first_stu_pos):
            out[ri, :spec.num_candidates] = stu_logits[first_pos, cand_stu]

    # span self-logits = mean of student logits at each constituent (pos, token_id)
    for dim, seg_i in enumerate(spec.span_seg_indices):
        col = spec.num_candidates + dim
        for ri in range(spec.num_segments):
            if ri == seg_i:
                constituents = spec.span_stu_constituents[dim]
                vals = torch.stack([stu_logits[pos, tid] for (pos, tid) in constituents])
                out[ri, col] = vals.mean()
            else:
                out[ri, col] = -1e9
    return out


def simct_reverse_kl(stu_virtual: torch.Tensor, tea_virtual: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Reverse-KL ``Σ p_s·(log p_s − log p_t)`` over the virtual vocab, per segment row.

    Returns a scalar loss = sum over all segment rows (caller normalizes).
    """
    s = stu_virtual / temperature
    t = tea_virtual / temperature
    s_lp = F.log_softmax(s.float(), dim=-1)
    t_lp = F.log_softmax(t.float(), dim=-1)
    rkl = (s_lp.exp() * (s_lp - t_lp)).sum(dim=-1)   # [num_segments]
    return rkl.sum()


# =====================================================================================
# Driver-side orchestration: turn a DataProto batch into per-sample SimCT specs by querying
# the remote teacher. This is the trainer-side half of the AgentJet integration.
# =====================================================================================
@dataclass
class SimCTSampleSpecWithOffset(SimCTSampleSpec):
    """A SimCTSampleSpec plus the student-side prompt offset (so the actor can map response-relative
    segment positions back to absolute input_ids positions)."""
    prompt_len: int = 0  # number of student prompt tokens (offset for seg_first_stu_pos / span constituents)


class SimCTDriver:
    """Builds per-sample SimCT specs from a batch by querying the remote teacher.

    Holds the (remote) teacher client + both tokenizers + the shared-id map. ``compute_specs`` decodes
    each student rollout, re-tokenizes the prompt/response with the teacher tokenizer, queries the
    teacher (teacher-tokenized prompt‖response, ``prompt_logprobs=topk``) with a FORCED prompt/response
    boundary (concatenated teacher token lists — avoids cross-boundary token merges), aligns the
    teacher vs student response tokenizations, and builds the spec. Student-side positions in the spec
    are RESPONSE-relative; the actor offsets by ``spec.prompt_len``.
    """

    def __init__(self, teacher_client, tok_teacher, tok_student, topk: int = 20,
                 log_prob_min_clamp: float = -10.0, chunk_size: int = 16):
        self.tc = teacher_client
        self.tok_t = tok_teacher
        self.tok_s = tok_student
        self.shared = build_shared_id_map(tok_student, tok_teacher)
        self.topk = topk
        self.clamp = log_prob_min_clamp
        self.chunk_size = chunk_size

    @torch.no_grad()
    def compute_specs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      response_mask: torch.Tensor) -> List[Optional[SimCTSampleSpecWithOffset]]:
        """Returns one spec per sample (None if the sample has no response / no aligned segments)."""
        input_ids = input_ids.cpu()
        am = attention_mask.bool().cpu()
        resp_mask = response_mask.cpu()
        B = input_ids.shape[0]

        # per-sample: decode, split prompt/response, build teacher sequences
        teacher_seqs, metas = [], []
        for i in range(B):
            full = input_ids[i][am[i]].tolist()                 # [prompt_real ..., resp_real ...]
            resp_real = int(resp_mask[i].sum().item())
            if resp_real <= 0 or resp_real >= len(full):
                metas.append(None); continue
            prompt_ids = full[:-resp_real]
            resp_ids = full[-resp_real:]
            prompt_text = self.tok_s.decode(prompt_ids)
            resp_text = self.tok_s.decode(resp_ids)
            tea_prompt = self.tok_t(prompt_text, add_special_tokens=False)["input_ids"]
            tea_resp = self.tok_t(resp_text, add_special_tokens=False)["input_ids"]
            teacher_seqs.append(tea_prompt + tea_resp)
            metas.append({"prompt_ids": prompt_ids, "resp_ids": resp_ids,
                          "tea_prompt": tea_prompt, "tea_resp": tea_resp,
                          "prompt_len": len(prompt_ids)})

        # query teacher (chunked) — _post_completions takes a list of token-id sequences
        specs: List[Optional[SimCTSampleSpecWithOffset]] = [None] * B
        valid_idx = [i for i, m in enumerate(metas) if m is not None]
        for s in range(0, len(valid_idx), self.chunk_size):
            batch_idx = valid_idx[s:s + self.chunk_size]
            seqs = [metas[i]["tea_prompt"] + metas[i]["tea_resp"] for i in batch_idx]
            pls = self.tc._post_completions(seqs)  # list (per seq) of prompt_logprobs lists
            for k, i in enumerate(batch_idx):
                m = metas[i]
                full_pl = pls[k]
                if full_pl is None:
                    continue
                resp_pl = full_pl[len(m["tea_prompt"]):]   # response-portion teacher prompt_logprobs
                al = align_sequences_with_spans(m["tea_resp"], m["resp_ids"], self.tok_t, self.tok_s)
                spec = build_teacher_simct_spec(al, resp_pl, self.shared, self.topk, self.clamp)
                if spec is None:
                    continue
                spec.prompt_len = m["prompt_len"]
                specs[i] = spec
        return specs

