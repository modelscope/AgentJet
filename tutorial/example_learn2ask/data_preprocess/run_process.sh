#!/bin/bash

# export DASHSCOPE_API_KEY=your_api_key

# python tutorial/example_learn2ask/data_preprocess/step1.py --input_file data/realmedconv/train_original.jsonl --output_file data/realmedconv/train_processed.jsonl
# python tutorial/example_learn2ask/data_preprocess/step2.py --input_file data/realmedconv/train_processed.jsonl --output_file data/realmedconv/train.jsonl

# python tutorial/example_learn2ask/data_preprocess/step1.py --input_file data/realmedconv/test_original.jsonl --output_file data/realmedconv/test_processed.jsonl
# python tutorial/example_learn2ask/data_preprocess/step2.py --input_file data/realmedconv/test_processed.jsonl --output_file data/realmedconv/test.jsonl


set -euo pipefail

DATA_DIR="${1:-}"
if [[ -z "${DATA_DIR}" ]]; then
  echo "Usage: $0 <data_dir>" >&2
  exit 2
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Error: data_dir is not a directory: ${DATA_DIR}" >&2
  exit 2
fi

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "Error: DASHSCOPE_API_KEY is not set. Please run: export DASHSCOPE_API_KEY=your_api_key" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TRAIN_ORIG="${DATA_DIR%/}/train_origin.jsonl"
TRAIN_PROC="${DATA_DIR%/}/train_processed.jsonl"
TRAIN_OUT="${DATA_DIR%/}/train.jsonl"

TEST_ORIG="${DATA_DIR%/}/test_origin.jsonl"
TEST_PROC="${DATA_DIR%/}/test_processed.jsonl"
TEST_OUT="${DATA_DIR%/}/test.jsonl"

if [[ ! -f "${TRAIN_ORIG}" ]]; then
  echo "Error: missing file: ${TRAIN_ORIG}" >&2
  exit 2
fi

if [[ ! -f "${TEST_ORIG}" ]]; then
  echo "Error: missing file: ${TEST_ORIG}" >&2
  exit 2
fi

python "${REPO_ROOT}/tutorial/example_learn2ask/data_preprocess/step1.py" --input_file "${TRAIN_ORIG}" --output_file "${TRAIN_PROC}"
python "${REPO_ROOT}/tutorial/example_learn2ask/data_preprocess/step2.py" --input_file "${TRAIN_PROC}" --output_file "${TRAIN_OUT}"

python "${REPO_ROOT}/tutorial/example_learn2ask/data_preprocess/step1.py" --input_file "${TEST_ORIG}" --output_file "${TEST_PROC}"
python "${REPO_ROOT}/tutorial/example_learn2ask/data_preprocess/step2.py" --input_file "${TEST_PROC}" --output_file "${TEST_OUT}"

echo "Done. Generated:"
echo "- ${TRAIN_OUT}"
echo "- ${TEST_OUT}"
