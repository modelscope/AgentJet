# -*- coding: utf-8 -*-
"""
Download DAPO-Math-17k (training) and AIME-2024/2025/2026 (test) datasets.

Usage:
    proxychains python -m tutorial.opencode_build_aime.download_data
"""

import os
import subprocess

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "dapo-math-17k.parquet")

TRAIN_URL = "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
AIME_2024_URL = "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
AIME_2025_RAW_URL = "https://huggingface.co/datasets/yentinglin/aime_2025/resolve/main/data/train-00000-of-00001-243207c6c994e1bd.parquet?download=true"
AIME_2026_RAW_URL = "https://huggingface.co/datasets/MathArena/aime_2026/resolve/main/data/train-00000-of-00001.parquet?download=true"

# Same instruction wrapper that BytedTsinghua-SIA/AIME-2024 bakes into its `prompt` field.
INSTRUCTION_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)


def _download_file(url: str, output_path: str, use_proxychains: bool = True):
    """Download a file using wget, optionally with proxychains."""
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exists")
        return

    print(f"[DOWNLOAD] {url} -> {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = ["wget", "-O", output_path, url]
    if use_proxychains:
        cmd = ["proxychains"] + cmd

    try:
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] Downloaded {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


def _convert_to_dapo_schema(raw_path: str, output_path: str, data_source: str):
    """Convert {problem, answer, ...}-style parquet to the BytedTsinghua-SIA AIME schema."""
    import pandas as pd

    df = pd.read_parquet(raw_path)
    rows = []
    for idx, src in df.iterrows():
        problem = src["problem"]
        answer = str(src["answer"])
        rows.append({
            "data_source": data_source,
            "prompt": [{"content": INSTRUCTION_PREFIX + problem, "role": "user"}],
            "ability": "MATH",
            "reward_model": {"ground_truth": answer, "style": "rule-lighteval/MATH_v2"},
            "extra_info": {"index": int(idx), "raw_problem": problem, "split": None},
        })
    pd.DataFrame(rows).to_parquet(output_path)
    print(f"[CONVERT] {raw_path} -> {output_path} ({len(rows)} rows, schema=dapo)")


def _ensure_with_conversion(raw_url: str, output_path: str, data_source: str, use_proxychains: bool = True):
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exists")
        return output_path
    raw_path = output_path + ".raw"
    if os.path.exists(raw_path):
        os.remove(raw_path)
    _download_file(raw_url, raw_path, use_proxychains=use_proxychains)
    try:
        _convert_to_dapo_schema(raw_path, output_path, data_source=data_source)
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)
    return output_path


def ensure_train(use_proxychains: bool = True) -> str:
    _download_file(TRAIN_URL, TRAIN_FILE, use_proxychains=use_proxychains)
    return TRAIN_FILE


def _dedup_aime_2024_in_place(path: str):
    """Upstream BytedTsinghua-SIA/AIME-2024 replicates each of the 30 problems 32x.
    Collapse to 30 unique rows so it lines up with AIME-2025/2026."""
    import pandas as pd

    df = pd.read_parquet(path)
    if len(df) == 30:
        return
    keys = df["extra_info"].apply(lambda x: x["raw_problem"])
    df = df.loc[~keys.duplicated()].reset_index(drop=True)
    df.to_parquet(path)
    print(f"[DEDUP] {path} -> {len(df)} unique rows")


def ensure_aime_2024(use_proxychains: bool = True) -> str:
    out = os.path.join(DATA_DIR, "aime-2024.parquet")
    _download_file(AIME_2024_URL, out, use_proxychains=use_proxychains)
    _dedup_aime_2024_in_place(out)
    return out


def ensure_aime_2025(use_proxychains: bool = True) -> str:
    out = os.path.join(DATA_DIR, "aime-2025.parquet")
    return _ensure_with_conversion(AIME_2025_RAW_URL, out, data_source="aime-2025", use_proxychains=use_proxychains)


def ensure_aime_2026(use_proxychains: bool = True) -> str:
    out = os.path.join(DATA_DIR, "aime-2026.parquet")
    return _ensure_with_conversion(AIME_2026_RAW_URL, out, data_source="aime-2026", use_proxychains=use_proxychains)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("Downloading DAPO-Math-17k and AIME-2024/2025/2026 datasets")
    print("=" * 60)

    print("\n[1/4] Downloading training set: dapo-math-17k.parquet")
    ensure_train()

    print("\n[2/4] Downloading test set: aime-2024.parquet")
    ensure_aime_2024()

    print("\n[3/4] Downloading test set: aime-2025.parquet")
    ensure_aime_2025()

    print("\n[4/4] Downloading test set: aime-2026.parquet")
    ensure_aime_2026()

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"  Training set: {TRAIN_FILE}")
    print(f"  Test sets:    {DATA_DIR}/aime-2024|2025|2026.parquet")
    print("=" * 60)


if __name__ == "__main__":
    main()
