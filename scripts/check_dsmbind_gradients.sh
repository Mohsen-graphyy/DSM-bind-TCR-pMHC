#!/usr/bin/env bash
set -euo pipefail
DIAG_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${DIAG_REPO_ROOT}"
python -u scripts/check_dsmbind_gradients.py \
  --dataset "${DATASET:-outputs/atlas_compare/dataset.sqlite}" \
  --embeddings "${ESM_CACHE:-data/esm_cache/esm2_t36_3B.sqlite}" \
  --pretrained "${PRETRAINED_CHECKPOINT:-DSMBind/ckpts/model.skempi.allatom}" \
  --finetuned "${FINETUNED_CHECKPOINT:-artifacts/checkpoints/finetuned_best.pt}" \
  --output-dir "${DIAG_OUTPUT:-outputs/gradient_check_01}" \
  --pair-index "${PAIR_INDEX:-0}" \
  --max-residues "${MAX_RESIDUES:-4}" \
  --device "${DEVICE:-cuda}"
