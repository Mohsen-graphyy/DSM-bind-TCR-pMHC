#!/usr/bin/env bash
set -euo pipefail
BENCH_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BENCH_REPO_ROOT}"
python scripts/dsmbind_benchmark.py run \
  --bundle "${BUNDLE_DIR:-data/atlas_benchmark/bundle}" \
  --work-dir "${WORK_DIR:-outputs/atlas_compare}" \
  --pretrained "${PRETRAINED_CHECKPOINT:-DSMBind/ckpts/model.skempi.allatom}" \
  --finetuned "${FINETUNED_CHECKPOINT:-artifacts/checkpoints/finetuned_best.pt}" \
  --embeddings "${ESM_CACHE:-data/esm_cache/esm2_t36_3B.sqlite}" \
  --device "${DEVICE:-cuda}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --esm-token-budget "${ESM_TOKEN_BUDGET:-512}" \
  --bootstrap "${BOOTSTRAP_SAMPLES:-2000}"
