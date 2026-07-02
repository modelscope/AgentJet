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
Remote vLLM teacher client for On-Policy Distillation (OPD).

Instead of launching the teacher as an in-cluster Ray rollout (as verl >=0.8.0 does), AgentJet
talks to an **external, already-running vLLM OpenAI-compatible server** configured via
``ajet.teacher_model.{teacher_model_vllm_url, teacher_model_name, teacher_model_api_key}``.

For each student sequence (prompt + response) the teacher performs a forward pass — we feed the
student's token ids as the prompt with ``max_tokens=1`` and ``prompt_logprobs=K`` — and returns the
teacher's top-K logprob distribution at every position. We then keep only the **response** positions
(aligned 1:1 with the student's ``responses`` / ``response_mask`` / ``log_probs``).

Two products, selected by ``topk``:
  * ``topk == 0`` (estimator modes kl/k1/k3/...): teacher logprob of the *sampled* token only,
    shape ``[B, resp_len, 1]``.
  * ``topk > 0`` (forward_kl_topk): teacher top-K ``(ids, logprobs)``, shapes
    ``[B, resp_len, K]``.

Tokenizer note: we send the student's token *ids* directly (vLLM skips re-tokenization), so the
teacher and student MUST share a vocabulary (same model family, e.g. Qwen3 -> Qwen3). Cross-family
distillation is not supported by this client.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

try:
    import requests  # type: ignore
    _HAVE_REQUESTS = True
except Exception:  # pragma: no cover
    import urllib.request
    import json as _json
    _HAVE_REQUESTS = False


class RemoteVllmTeacherClient:
    """HTTP client to a remote vLLM server for OPD teacher logprob scoring."""

    def __init__(
        self,
        vllm_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        topk: int = 0,
        timeout: float = 120.0,
        max_concurrent: int = 32,
        chunk_size: int = 16,
        max_model_len: Optional[int] = None,
        log_prob_min_clamp: float = -10.0,
    ):
        self.url = vllm_url.rstrip("/")
        if self.url.endswith("/v1"):
            self.base = self.url
        else:
            self.base = self.url + "/v1" if "/v1" not in self.url else self.url
        # ensure base ends with /v1 for the OpenAI endpoints
        if not self.base.endswith("/v1"):
            self.base = self.base.rstrip("/") + "/v1"
        self.model = model_name
        self.api_key = api_key or "EMPTY"
        self.topk = int(topk)
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.max_model_len = max_model_len
        self.log_prob_min_clamp = log_prob_min_clamp
        self._headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        logger.info(
            "[OPD-teacher] remote vLLM client ready: base=%s model=%s topk=%d", self.base, self.model, self.topk
        )

    # ------------------------------------------------------------------
    # low-level: score a chunk of (already token-id) sequences
    # ------------------------------------------------------------------
    def _post_completions(self, prompts_ids: list[list[int]]) -> list[Optional[list]]:
        """Call ``/v1/completions`` with a batch of token-id prompts; return per-prompt
        ``prompt_logprobs`` lists (each is a list over positions of dict|None)."""
        prompt_logprobs_n = max(1, self.topk)
        payload = {
            "model": self.model,
            "prompt": prompts_ids,          # vLLM accepts token-id lists
            "max_tokens": 1,
            "temperature": 1.0,             # no effect on prompt_logprobs (forward pass)
            "logprobs": 0,
            "prompt_logprobs": prompt_logprobs_n,
            "stream": False,
        }
        url = self.base + "/completions"
        if _HAVE_REQUESTS:
            resp = requests.post(url, json=payload, headers=self._headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        else:  # urllib fallback
            req = urllib.request.Request(
                url, data=_json.dumps(payload).encode(), headers=self._headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = _json.loads(r.read().decode())
        choices = data.get("choices", [])
        out = []
        for i in range(len(prompts_ids)):
            if i < len(choices):
                out.append(choices[i].get("prompt_logprobs"))
            else:
                out.append(None)
        return out

    # ------------------------------------------------------------------
    # public API: score a full DataProto batch
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_teacher_logprobs(
        self,
        input_ids: torch.Tensor,        # [B, T] prompt|response (left-pad prompt, right-pad resp)
        attention_mask: torch.Tensor,   # [B, T]
        response_mask: torch.Tensor,    # [B, resp_len]
    ) -> dict[str, Optional[torch.Tensor]]:
        """Return teacher logprobs aligned to the response tokens.

        Returns a dict with:
          ``teacher_logprobs``: [B, resp_len, K'] where K'=1 for estimator, K'=topk for top-k.
          ``teacher_ids``:      [B, resp_len, topk] (None / omitted when topk==0).
        """
        input_ids = input_ids.cpu()
        attention_mask = attention_mask.cpu()
        response_mask = response_mask.cpu()
        B, resp_len = response_mask.shape
        K = max(1, self.topk)

        # reconstruct unpadded prompt|response sequences + per-sample valid response length
        seqs: list[list[int]] = []
        resp_lens: list[int] = []
        am = attention_mask.bool()
        for i in range(B):
            ids = input_ids[i][am[i]].tolist()  # [prompt_real ..., resp_real ...] in order
            r_len = int(response_mask[i].sum().item())
            # left-truncate the prompt if the teacher context is too small to hold prompt+response
            if self.max_model_len is not None and len(ids) > self.max_model_len:
                keep_resp = ids[-r_len:]
                prompt_part = ids[:-r_len]
                allowed_prompt = max(0, self.max_model_len - r_len)
                if allowed_prompt < len(prompt_part):
                    logger.warning(
                        "[OPD-teacher] sample %d len=%d > teacher max_model_len=%d; "
                        "left-truncating prompt to %d tokens (response kept intact).",
                        i, len(ids), self.max_model_len, allowed_prompt,
                    )
                    prompt_part = prompt_part[-allowed_prompt:] if allowed_prompt > 0 else []
                ids = prompt_part + keep_resp
            seqs.append(ids)
            resp_lens.append(r_len)

        # chunk + concurrent POST
        chunks = [seqs[i : i + self.chunk_size] for i in range(0, len(seqs), self.chunk_size)]
        chunk_results: list[Optional[list]] = [None] * len(seqs)

        def _run_chunk(start: int, chunk: list[list[int]]):
            return start, self._post_completions(chunk)

        workers = min(self.max_concurrent, max(1, len(chunks)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_chunk, j * self.chunk_size, c) for j, c in enumerate(chunks)]
            for fut in as_completed(futs):
                start, res = fut.result()
                for k, r in enumerate(res):
                    chunk_results[start + k] = r

        # parse into padded tensors aligned to response tokens
        teacher_lp = torch.full((B, resp_len, K), self.log_prob_min_clamp, dtype=torch.float32)
        teacher_ids = None
        if self.topk > 0:
            teacher_ids = torch.zeros((B, resp_len, self.topk), dtype=torch.int64)

        for i in range(B):
            pl = chunk_results[i]  # list over positions of dict|None
            r_len = resp_lens[i]
            if pl is None:
                continue
            seq = seqs[i]
            # response tokens occupy the last r_len positions of `seq`
            for j in range(r_len):
                pos = len(seq) - r_len + j
                if pos < 0 or pos >= len(pl):
                    continue
                entry = pl[pos]
                if entry is None:
                    continue  # position 0 or missing → keep clamp
                # vLLM OpenAI prompt_logprobs: each entry is a dict keyed by the candidate
                # TOKEN_ID (as a string) -> {"logprob": float, "rank": int, "decoded_token": str}.
                # (There is no "token_id" field in the value — the id IS the key.)
                items = []
                for tok_id_str, info in entry.items():
                    if not isinstance(info, dict):
                        continue
                    try:
                        tid = int(tok_id_str)
                    except (ValueError, TypeError):
                        continue
                    lp = float(info.get("logprob", self.log_prob_min_clamp))
                    items.append((tid, lp))
                if not items:
                    continue
                # sort by logprob desc to get top-K
                items.sort(key=lambda x: x[1], reverse=True)
                # estimator mode (topk==0 → K==1): prefer the teacher logprob of the ACTUAL sampled token
                if self.topk == 0:
                    actual_tok = seq[pos]
                    hit = next((lp for (tid, lp) in items if tid == actual_tok), None)
                    if hit is not None:
                        teacher_lp[i, j, 0] = hit
                    else:
                        # actual token not in returned set → clamp (rare; means teacher is very peaked)
                        teacher_lp[i, j, 0] = self.log_prob_min_clamp
                else:
                    for kk in range(min(self.topk, len(items))):
                        tid, lp = items[kk]
                        teacher_lp[i, j, kk] = lp
                        teacher_ids[i, j, kk] = tid
        return {"teacher_logprobs": teacher_lp, "teacher_ids": teacher_ids}


def build_teacher_client_from_ajet(ajet_cfg: Any) -> Optional[RemoteVllmTeacherClient]:
    """Construct a :class:`RemoteVllmTeacherClient` from the ``ajet.teacher_model`` block.

    Returns None if OPD is disabled or the URL is empty.
    """
    teacher = getattr(ajet_cfg, "teacher_model", None) if not hasattr(ajet_cfg, "get") else ajet_cfg.get("teacher_model")
    if teacher is None:
        return None
    if hasattr(teacher, "get"):
        enabled = bool(teacher.get("teacher_opd_enabled", False))
        url = teacher.get("teacher_model_vllm_url", "")
        name = teacher.get("teacher_model_name", "")
        api_key = teacher.get("teacher_model_api_key", "EMPTY") or "EMPTY"
        topk = int(teacher.get("teacher_topk", 0) or 0)
        timeout = float(teacher.get("teacher_request_timeout", 120) or 120)
        max_concurrent = int(teacher.get("teacher_max_concurrent", 32) or 32)
        chunk_size = int(teacher.get("teacher_chunk_size", 16) or 16)
        max_model_len = teacher.get("teacher_max_model_len", None)
        min_clamp = float(teacher.get("teacher_log_prob_min_clamp", -10.0))
    else:
        enabled = bool(getattr(teacher, "teacher_opd_enabled", False))
        url = getattr(teacher, "teacher_model_vllm_url", "")
        name = getattr(teacher, "teacher_model_name", "")
        api_key = getattr(teacher, "teacher_model_api_key", "EMPTY") or "EMPTY"
        topk = int(getattr(teacher, "teacher_topk", 0) or 0)
        timeout = float(getattr(teacher, "teacher_request_timeout", 120) or 120)
        max_concurrent = int(getattr(teacher, "teacher_max_concurrent", 32) or 32)
        chunk_size = int(getattr(teacher, "teacher_chunk_size", 16) or 16)
        max_model_len = getattr(teacher, "teacher_max_model_len", None)
        min_clamp = float(getattr(teacher, "teacher_log_prob_min_clamp", -10.0))
    if not (enabled and url and name):
        return None
    if max_model_len is not None:
        max_model_len = int(max_model_len)
    return RemoteVllmTeacherClient(
        vllm_url=url,
        model_name=name,
        api_key=api_key,
        topk=topk,
        timeout=timeout,
        max_concurrent=max_concurrent,
        chunk_size=chunk_size,
        max_model_len=max_model_len,
        log_prob_min_clamp=min_clamp,
    )
