#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
URL="https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/astuner_archive/RealMedConv_processed.zip"
TMP_DIR="/tmp/learn2ask"

TARGET_DIR="${1:-${REPO_ROOT}/data/realmedconv}"
if [[ "${TARGET_DIR}" != /* ]]; then
  TARGET_DIR="${REPO_ROOT}/${TARGET_DIR}"
fi

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}" "${TARGET_DIR}"

wget -O "${TMP_DIR}/RealMedConv_processed.zip" "${URL}"
unzip -o "${TMP_DIR}/RealMedConv_processed.zip" -d "${TMP_DIR}"

mv "${TMP_DIR}/test.jsonl" "${TARGET_DIR}/test.jsonl"
mv "${TMP_DIR}/train.jsonl" "${TARGET_DIR}/train.jsonl"

echo "Saved processed dataset to ${TARGET_DIR}"
