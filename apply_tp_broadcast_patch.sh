#!/usr/bin/env bash
#
# Applies the "TP-replicated KV cache HCCS broadcast" patch to a
# vllm-ascend checkout pinned at commit 50e0ce60849a7b6fe6f94 (tag
# v0.25.1rc1).
#
# Usage:
#   ./apply_tp_broadcast_patch.sh /path/to/vllm-ascend
#
# What it does:
#   1. Verifies the target looks like the right vllm-ascend checkout.
#   2. `git apply --check`'s both patches first (aborts cleanly, with no
#      files touched, if either would not apply -- e.g. because your
#      checkout has local modifications in these two files).
#   3. Applies envs.py.patch and kv_transfer.py.patch.
#   4. Copies in the new tp_kv_broadcast.py file.
#   5. Byte-compiles the three touched files with `python3 -m py_compile`
#      as a cheap smoke test.
#
# This script does NOT modify vllm, Mooncake, or anything outside
# vllm-ascend, and does not require network access on the target
# machine (the patches are plain-text diffs shipped alongside this
# script).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-}"

if [[ -z "${TARGET}" ]]; then
  echo "Usage: $0 /path/to/vllm-ascend" >&2
  exit 1
fi
if [[ ! -d "${TARGET}" ]]; then
  echo "ERROR: ${TARGET} is not a directory" >&2
  exit 1
fi

cd "${TARGET}"

KV_TRANSFER_REL="vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py"
ENVS_REL="vllm_ascend/envs.py"
NEW_FILE_REL="vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/tp_kv_broadcast.py"

if [[ ! -f "${KV_TRANSFER_REL}" || ! -f "${ENVS_REL}" ]]; then
  echo "ERROR: ${TARGET} does not look like a vllm-ascend checkout" \
       "(missing ${KV_TRANSFER_REL} or ${ENVS_REL})." >&2
  exit 1
fi

if command -v git >/dev/null 2>&1 && git -C "${TARGET}" rev-parse HEAD >/dev/null 2>&1; then
  CUR_COMMIT="$(git -C "${TARGET}" rev-parse HEAD)"
  EXPECTED_PREFIX="50e0ce60"
  if [[ "${CUR_COMMIT}" != ${EXPECTED_PREFIX}* ]]; then
    echo "WARNING: current HEAD is ${CUR_COMMIT}, not ${EXPECTED_PREFIX}*." >&2
    echo "This patch was handwritten against vllm-ascend commit" \
         "50e0ce60849a7b6fe6f94 (tag v0.25.1rc1). It may still apply on a" \
         "nearby commit, but review the result carefully if your checkout" \
         "has diverged." >&2
  fi
else
  echo "WARNING: could not determine current git commit (not a git repo" \
       "or git not installed); skipping the commit-match check." >&2
fi

echo "==> Checking that both patches apply cleanly (no files modified yet)..."
git apply --check "${SCRIPT_DIR}/kv_transfer.py.patch"
git apply --check "${SCRIPT_DIR}/envs.py.patch"
echo "    OK."

echo "==> Applying envs.py.patch ..."
git apply "${SCRIPT_DIR}/envs.py.patch"

echo "==> Applying kv_transfer.py.patch ..."
git apply "${SCRIPT_DIR}/kv_transfer.py.patch"

echo "==> Installing new file: ${NEW_FILE_REL}"
cp "${SCRIPT_DIR}/tp_kv_broadcast.py" "${NEW_FILE_REL}"

echo "==> Byte-compiling touched files as a smoke test ..."
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -n "${PYTHON_BIN}" ]]; then
  "${PYTHON_BIN}" -m py_compile "${KV_TRANSFER_REL}" "${ENVS_REL}" "${NEW_FILE_REL}"
  echo "    OK."
else
  echo "    WARNING: no python interpreter found on PATH; skipped." >&2
fi

cat <<'EOF'

==> Done. The patch is applied but DISABLED by default (no behavior
    change until you opt in).

To enable TP-replicated KV cache broadcast for an MLA/DSA-style model
(e.g. DeepSeek V3/V4, where the whole KV cache is one byte-identical
group across TP ranks), set before launching every TP worker:

    export VLLM_ASCEND_KV_TP_BROADCAST_GROUPS=all
    # optional, defaults to 8 MiB:
    export VLLM_ASCEND_KV_TP_BROADCAST_SHARD_BYTES=8388608

See README_tp_broadcast.md (shipped next to this script) for full
details, correctness caveats, and how to verify it on your hardware
before relying on it in production.
EOF
