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
SimCT cross-tokenizer alignment utilities (ported from the SimCT codebase,
https://github.com/sunjie279/SimCT- , file ``kdflow/algorithms/span_ctkd.py``).

These are the tokenizer-only core of SimCT — the parts that define the *common
supervision space* used by cross-tokenizer on-policy distillation:

  1. ``find_overlap_tokens`` — the 1:1 shared vocabulary (teacher ∩ student),
     normalizing the GPT-2 / Llama space markers (``Ġ`` ↔ ``▁``).
  2. ``align_sequences_with_spans`` — the *minimal aligned units*: a greedy
     cumulative-text walk over the two tokenizations of the SAME text. Consecutive
     1:1-aligned boundary points delimit segments; a segment whose teacher or
     student side has >1 token is a "span" (a multi-token unit both tokenizers
     realize, e.g. teacher "hap|py" vs student "ha|pp|y" → the unit "happy").

The scoring / loss / remote-vLLM teacher adaptation live elsewhere
(``distillation.py`` loss_mode="simct", ``teacher_client.py``); this module only
depends on the two HF tokenizers and is fully CPU-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

# a segment = (teacher_start, teacher_end, student_start, student_end) half-open
# index ranges into the two *label-id* (next-token) sequences. Both ranges encode
# the same text. (1,1) => 1:1 aligned token; otherwise => a span (minimal aligned unit).
Segment = Tuple[int, int, int, int]


@dataclass
class Alignment:
    """Result of aligning a teacher and a student tokenization of the same text."""
    segments: List[Segment]               # minimal aligned units (1:1 or span)
    tea_ids: List[int]                    # teacher label ids that were aligned
    stu_ids: List[int]                    # student label ids that were aligned
    tea_texts: List[str]                  # decoded teacher token texts
    stu_texts: List[str]                  # decoded student token texts

    @property
    def num_spans(self) -> int:
        return sum(1 for (ts, te, ss, se) in self.segments if (te - ts) > 1 or (se - ss) > 1)

    @property
    def num_1to1(self) -> int:
        return sum(1 for (ts, te, ss, se) in self.segments if (te - ts) == 1 and (se - ss) == 1)


def _normalize_space(tok_str: str) -> str:
    """Normalize the GPT-2 ``Ġ`` / Llama ``▁`` space marker so the same token text
    matches across tokenizers that use different markers."""
    return tok_str.replace("Ġ", "▁")


def find_overlap_tokens(tok_student, tok_teacher) -> Tuple[List[int], List[int]]:
    """Return (student_overlap_ids, teacher_overlap_ids) for the shared vocabulary.

    Matches token *strings* after space-marker normalization, so a token present
    in both vocabs (identical decoded text) is treated as 1:1-aligned regardless
    of its id in either vocab. The EOS token is force-included.
    """
    stu_vocab = {_normalize_space(k): v for k, v in tok_student.get_vocab().items()}
    tea_vocab = {_normalize_space(k): v for k, v in tok_teacher.get_vocab().items()}
    overlap = set(stu_vocab.keys()) & set(tea_vocab.keys())
    stu_ids = [stu_vocab[t] for t in overlap]
    tea_ids = [tea_vocab[t] for t in overlap]
    # guarantee EOS is a comparable unit
    stu_eos, tea_eos = tok_student.eos_token_id, tok_teacher.eos_token_id
    if stu_eos is not None and stu_eos not in stu_ids:
        stu_ids.append(stu_eos); tea_ids.append(tea_eos)
    return stu_ids, tea_ids


def align_sequences_with_spans(
    teacher_ids: List[int],
    student_ids: List[int],
    tok_teacher,
    tok_student,
) -> Alignment:
    """Greedy cumulative-text alignment between a teacher and a student tokenization
    of the SAME underlying text → minimal aligned units (segments).

    Ported faithfully from SimCT ``span_ctkd._align_sequences_with_spans``: decode
    every token to text, then walk both sequences accumulating running text
    (``history_tea`` / ``history_stu``); whenever the histories are equal AND the
    next tokens' decoded texts match, emit a 1:1 boundary. Regions between
    boundaries become segments (spans when multi-token on either side).

    Args:
        teacher_ids / student_ids: token id lists of the SAME text under each tokenizer.
    """
    if len(teacher_ids) == 0 or len(student_ids) == 0:
        return Alignment([], list(teacher_ids), list(student_ids), [], [])

    tea_texts = [tok_teacher.decode([tid]) for tid in teacher_ids]
    stu_texts = [tok_student.decode([tid]) for tid in student_ids]
    tea_eos = tok_teacher.eos_token
    stu_eos = tok_student.eos_token

    i = j = 0
    boundaries: List[Tuple[int, int]] = []   # (teacher_idx, student_idx) of 1:1 points
    history_tea = ""
    history_stu = ""
    while i < len(tea_texts) and j < len(stu_texts):
        is_eos_match = (tea_texts[i] == tea_eos and stu_texts[j] == stu_eos)
        if history_tea == history_stu and (tea_texts[i] == stu_texts[j] or is_eos_match):
            boundaries.append((i, j))
            history_tea += tea_texts[i]
            history_stu += stu_texts[j]
            i += 1
            j += 1
        elif len(history_tea) > len(history_stu):
            history_stu += stu_texts[j]
            j += 1
        elif len(history_tea) < len(history_stu):
            history_tea += tea_texts[i]
            i += 1
        else:
            history_tea += tea_texts[i]
            history_stu += stu_texts[j]
            i += 1
            j += 1

    # boundaries[k] ends segment k; the previous boundary (or -1) starts it
    segments: List[Segment] = []
    for idx, (ti, sj) in enumerate(boundaries):
        ts = 0 if idx == 0 else boundaries[idx - 1][0] + 1
        ss = 0 if idx == 0 else boundaries[idx - 1][1] + 1
        segments.append((ts, ti + 1, ss, sj + 1))
    return Alignment(segments, list(teacher_ids), list(student_ids), tea_texts, stu_texts)


def segments_summary(alignment: Alignment) -> str:
    """Human-readable view: each segment as tea_text|stu_text with token counts."""
    parts = []
    for (ts, te, ss, se) in alignment.segments:
        t_txt = "".join(alignment.tea_texts[ts:te])
        s_txt = "".join(alignment.stu_texts[ss:se])
        kind = "1:1" if (te - ts) == 1 and (se - ss) == 1 else "span"
        parts.append(f"[{kind}] tea({te-ts})={t_txt!r} stu({se-ss})={s_txt!r}")
    return " ".join(parts)
