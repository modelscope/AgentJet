# -*- coding: utf-8 -*-
"""
Download DAPO-Math-17k (training) and AIME-2024 (test) datasets.

Usage:
    proxychains python -m tutorial.opencode_build_aime.download_data
"""

import os
import subprocess

# Default paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "dapo-math-17k.parquet")
TEST_FILE = os.path.join(DATA_DIR, "aime-2024.parquet")

# HuggingFace URLs
TRAIN_URL = "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
TEST_URL = "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"


def download_file(url: str, output_path: str, use_proxychains: bool = True):
    """Download a file using wget, optionally with proxychains."""
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exists")
        return

    print(f"[DOWNLOAD] {url} -> {output_path}")

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


def main():
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("Downloading DAPO-Math-17k and AIME-2024 datasets")
    print("=" * 60)

    # Download training data
    print("\n[1/2] Downloading training set: dapo-math-17k.parquet")
    download_file(TRAIN_URL, TRAIN_FILE)

    # Download test data
    print("\n[2/2] Downloading test set: aime-2024.parquet")
    download_file(TEST_URL, TEST_FILE)

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"  Training set: {TRAIN_FILE}")
    print(f"  Test set:     {TEST_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
