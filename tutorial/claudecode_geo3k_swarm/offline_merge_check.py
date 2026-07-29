# -*- coding: utf-8 -*-
"""
Offline (no GPU / no swarm) sanity check for the multi-message multimodal fix.

It proves, using the real Qwen2.5-VL processor and the real
``ExtendedMessage`` / ``merge_multi_modal_inputs`` / ``get_rope_index`` code
paths, that when images arrive in *separate* messages the training-side merge:

  1. reconstructs the same ``pixel_values`` / ``image_grid_thw`` as a single
     whole-conversation processor call (the vLLM request path), and
  2. yields 4-channel position ids whose length equals ``input_ids``, and
  3. places every ``<|image_pad|>`` (151655) token where a loss mask would be 0
     (i.e. inside non-``llm`` messages — never trained on).

Run:
    source .venv/bin/activate
    python -m tutorial.claudecode_geo3k_swarm.offline_merge_check
"""

import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from ajet.schema.extended_msg import ExtendedMessage
from ajet.context_tracker.single_agent_tracking import merge_multi_modal_inputs

IMAGE_PAD_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653

MODEL_PATH = os.environ.get(
    "REMOTE_MODEL_PATH",
    "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-VL-7B-Instruct",
)


def _img(w, h):
    return Image.new("RGB", (w, h), "white")


def build_timeline(tokenizer, processor):
    """A multi-message conversation: system / user(imgA) / assistant / user(imgB)+text.

    Mirrors what MultiAgentContextTracker.step_spawn_timeline produces: the
    first message is the system message; image-bearing messages carry
    ``images=`` + ``processor=`` so ExtendedMessage runs the processor path.
    """
    imgA = _img(256, 168)   # -> 54 pads, grid [1,12,18]
    imgB = _img(336, 224)   # -> 96 pads, grid [1,16,24]

    system = ExtendedMessage(
        author="initialization", role="system", content="You are helpful.",
        tokenizer=tokenizer, token_generator="auto", first_message=True,
    )
    user1 = ExtendedMessage(
        author="env", role="user", content="Here is the first figure.",
        tokenizer=tokenizer, token_generator="auto",
        images=[imgA], processor=processor,
    )
    asst = ExtendedMessage(
        author="env", role="assistant", content="I see the first figure.",
        tokenizer=tokenizer, token_generator="auto",
    )
    user2 = ExtendedMessage(
        author="env", role="user", content="Now compare with the second. Find y.",
        tokenizer=tokenizer, token_generator="auto",
        images=[imgB], processor=processor,
    )
    return [system, user1, asst, user2], [imgA, imgB]


def reference_whole_conversation(processor, images):
    """One processor call over the whole conversation = the vLLM request path."""
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": "You are helpful."}]},
        {"role": "user", "content": [
            {"type": "image", "image": images[0]},
            {"type": "text", "text": "Here is the first figure."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "I see the first figure."}]},
        {"role": "user", "content": [
            {"type": "image", "image": images[1]},
            {"type": "text", "text": "Now compare with the second. Find y."}]},
    ]
    text = processor.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
    mi = processor(text=[text], images=images, return_tensors="pt")
    return dict(mi)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

    ext_steps, images = build_timeline(tokenizer, processor)
    merged, proc = merge_multi_modal_inputs(ext_steps)
    ref = reference_whole_conversation(processor, images)

    ok = True

    # ---- (1) merged tensors == whole-conversation processor tensors ----------
    grid_ok = torch.equal(merged["image_grid_thw"], ref["image_grid_thw"])
    px_ok = torch.equal(merged["pixel_values"], ref["pixel_values"])
    print(f"[1] merged image_grid_thw == ref : {grid_ok}  ({merged['image_grid_thw'].tolist()})")
    print(f"[1] merged pixel_values   == ref : {px_ok}  {tuple(merged['pixel_values'].shape)}")
    ok = ok and grid_ok and px_ok

    # total pad count across the two grids
    pads = int((merged["image_grid_thw"].prod(dim=1) // (processor.image_processor.merge_size ** 2)).sum())
    print(f"[1] expected total image pads    : {pads} (want 54+96=150)")
    ok = ok and (pads == 150)

    # ---- (2) 4-channel rope aligns with concatenated input_ids ---------------
    input_ids = []
    attention_mask = []
    loss_mask = []
    for m in ext_steps:
        input_ids += m.token_arr
        attention_mask += [1] * len(m.token_arr)
        # non-llm messages contribute loss_mask 0 (as tokenize_steps does)
        loss_mask += m.get_loss_mask(blackout_token_combo=tokenizer.encode("<|im_start|>assistant\n"))

    from verl.models.transformers.qwen2_vl import get_rope_index
    from verl.utils.model import compute_position_id_with_mask

    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    attn_t = torch.tensor(attention_mask, dtype=torch.long)
    mrope_3 = get_rope_index(
        processor, input_ids=input_ids_t,
        image_grid_thw=merged.get("image_grid_thw"),
        attention_mask=attn_t,
    )
    text_pos = compute_position_id_with_mask(attn_t)
    pos_ids_4 = torch.cat([text_pos.view(1, -1), mrope_3], dim=0)
    len_ok = pos_ids_4.shape == (4, len(input_ids))
    print(f"[2] 4-channel position_ids shape : {tuple(pos_ids_4.shape)}  (want (4, {len(input_ids)}))  {len_ok}")
    ok = ok and len_ok

    # ---- (3) image pads present and all at loss_mask == 0 --------------------
    n_pad_ids = input_ids.count(IMAGE_PAD_ID)
    pads_all_masked = all(
        loss_mask[i] == 0 for i, t in enumerate(input_ids) if t == IMAGE_PAD_ID
    )
    has_spans = input_ids.count(VISION_START_ID) == 2 and input_ids.count(VISION_END_ID) == 2
    print(f"[3] image_pad tokens in input_ids: {n_pad_ids} (want 150)")
    print(f"[3] all image pads loss_mask==0  : {pads_all_masked}")
    print(f"[3] two vision spans present     : {has_spans}")
    ok = ok and (n_pad_ids == 150) and pads_all_masked and has_spans

    print("=" * 60)
    print("OFFLINE MERGE CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
