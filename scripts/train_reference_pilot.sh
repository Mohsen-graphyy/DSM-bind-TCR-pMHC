#!/usr/bin/env bash
set -euo pipefail
PILOT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PILOT_REPO_ROOT}"
export PYTHONPATH="${PILOT_REPO_ROOT}/DSMBind${PYTHONPATH:+:${PYTHONPATH}}"
PILOT_DATA="${PILOT_DATA:-data/reference_pilot/bundle}"
PILOT_OUTPUT="${PILOT_OUTPUT:-outputs/reference_pilot_01}"
PILOT_ESM_CACHE="${ESM_CACHE:-data/esm_cache/esm2_t36_3B.sqlite}"
# Reuse the ESM model download/cache from Atlas; compute only the missing sequences.
python -u -m bindenergy.apps.tcr_pmhc.embeddings \
  --datasets "${PILOT_DATA}/train.sqlite" "${PILOT_DATA}/validation.sqlite" \
  --output "${PILOT_ESM_CACHE}" --device cuda --token-budget "${ESM_TOKEN_BUDGET:-512}"
PILOT_ARGS=(
  --train-db "${PILOT_DATA}/train.sqlite"
  --validation-db "${PILOT_DATA}/validation.sqlite"
  --embeddings "${PILOT_ESM_CACHE}"
  --init-checkpoint DSMBind/ckpts/model.skempi.allatom
  --output-dir "${PILOT_OUTPUT}"
  --recurrence-backend reference --device cuda
  --batch-size 1 --learning-rate "${LEARNING_RATE:-1e-5}"
  --epochs "${EPOCHS:-2}" --patience 2 --anneal-rate 1.0
  --validation-batches 0 --checkpoint-every-batches 16 --seed 7
)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  PILOT_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi
mkdir -p "$(dirname "${PILOT_OUTPUT}")"
python -u -m bindenergy.apps.tcr_pmhc.train "${PILOT_ARGS[@]}" 2>&1 | tee -a "${PILOT_OUTPUT}.log"
echo "Pilot finished. Review ${PILOT_OUTPUT}/final_metrics.json before Atlas evaluation."
