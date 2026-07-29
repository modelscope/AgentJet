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


import json
import os

from verl import DataProto


IMAGE_PAD_ID = 151655  # Qwen2.5-VL <|image_pad|>


def capture_training_token_ids(batch: DataProto, tokenizer, out_path: str, global_step: int = 0) -> None:
    """One-shot debug dump of the *training-side* token ids for a batch.

    Writes each sample's real (unpadded) prompt token id array, tagging the
    image-pad (151655) tokens, plus per-sample image counts and grid_thw. Used
    by ``tutorial/claudecode_geo3k_swarm/test_token_ids.py`` to compare the
    training-side tokenization against the vLLM-side capture in
    ``token_id_io.md``. Inert during normal training (gated on an env var by
    the caller).
    """
    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]
    attention_mask = batch.batch["attention_mask"]
    prompt_len = prompts.shape[1]
    resp_len = responses.shape[1]
    mmi = batch.non_tensor_batch.get("multi_modal_inputs", None)
    task_ids = batch.non_tensor_batch.get("task_ids", None)

    def _real_prompt(i):
        mask = attention_mask[i, :prompt_len].bool()
        return prompts[i][mask].tolist()

    def _real_response(i):
        mask = attention_mask[i, prompt_len:prompt_len + resp_len].bool()
        return responses[i][mask].tolist()

    def _grid(i):
        if mmi is None:
            return None
        d = mmi[i]
        if not isinstance(d, dict):
            return None
        g = d.get("image_grid_thw", None)
        if g is None:
            return None
        try:
            return g.tolist()
        except AttributeError:
            return g

    n = len(prompts)

    def _sample_md(idx: int, header_note: str) -> str:
        """Full markdown dump (summary row + arrays + decoded) for one sample."""
        p = _real_prompt(idx)
        r = _real_response(idx)
        tid = task_ids[idx] if task_ids is not None else "?"
        lines = [
            "# Training-side token ids captured at trainer_verl.py:599",
            "",
            "Final tokenization consumed by training (image already expanded to "
            f"`<|image_pad|>` = {IMAGE_PAD_ID}). Compare against the vLLM-side "
            "ground-truth capture.",
            "",
            f"- global_step: {global_step}",
            f"- {header_note}",
            f"- task_id: {tid}",
            f"- image_tokens: {p.count(IMAGE_PAD_ID)}",
            f"- grid_thw: {_grid(idx)}",
            "",
            f"## Full arrays for sample idx={idx}",
            "",
            f"Input (prompt) token ids — length {len(p)}, image_tokens {p.count(IMAGE_PAD_ID)}:",
            "",
            "```json",
            json.dumps(p),
            "```",
            "",
            f"Output (response) token ids — length {len(r)}:",
            "",
            "```json",
            json.dumps(r),
            "```",
            "",
        ]
        if tokenizer is not None:
            try:
                lines += [
                    "Decoded prompt (special tokens kept):",
                    "",
                    "```",
                    tokenizer.decode(p, skip_special_tokens=False),
                    "```",
                    "",
                ]
            except Exception:
                pass
        return "\n".join(lines)

    # Directory mode: AJET_CAPTURE_TOKEN_IDS points at a directory -> write one
    # <task_id>.debug.md per distinct task_id (first sample of each), so the
    # multimodal test can compare each case independently. File mode keeps the
    # original single-file behavior (first image-bearing sample) for the
    # existing single-image test_token_ids.py.
    is_dir = os.path.isdir(out_path) or out_path.endswith(("/", os.sep))
    if is_dir:
        os.makedirs(out_path, exist_ok=True)
        seen = set()
        for i in range(n):
            tid = str(task_ids[i]) if task_ids is not None else str(i)
            if tid in seen:
                continue
            seen.add(tid)
            safe = tid.replace("/", "_").replace(os.sep, "_")
            with open(os.path.join(out_path, f"{safe}.debug.md"), "w", encoding="utf-8") as f:
                f.write(_sample_md(i, f"batch size: {n} (per-task capture)"))
        return

    chosen = None
    for i in range(n):
        if _real_prompt(i).count(IMAGE_PAD_ID) > 0:
            chosen = i
            break
    if chosen is None:
        chosen = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_sample_md(chosen, f"batch size: {n} (first image-bearing sample)"))
