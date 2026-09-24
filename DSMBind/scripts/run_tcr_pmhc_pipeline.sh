#!/usr/bin/env bash
set -euo pipefail

# Paths can be overridden without editing this file.
DATA_ROOT="${DATA_ROOT:-../data}"
CSV_PATH="${CSV_PATH:-${DATA_ROOT}/linked_data/final_dataset_datasail_like_C1f_4labels_fixed.csv}"
PDB_DIR="${PDB_DIR:-${DATA_ROOT}/new_structure}"
PREPARED_DIR="${PREPARED_DIR:-${DATA_ROOT}/dsmbind_tcr_pmhc}"
RUN_DIR="${RUN_DIR:-outputs/tcr_pmhc_allatom}"
EMBEDDINGS_DB="${EMBEDDINGS_DB:-${PREPARED_DIR}/esm2_t36_3B.sqlite}"

python -m bindenergy.apps.tcr_pmhc.prepare \
  --csv "${CSV_PATH}" \
  --pdb-dir "${PDB_DIR}" \
  --output-dir "${PREPARED_DIR}" \
  --patch-size 50 \
  --min-residue-contacts 1 \
  --resume

python -m bindenergy.apps.tcr_pmhc.embeddings \
  --datasets \
    "${PREPARED_DIR}/train.sqlite" \
    "${PREPARED_DIR}/validation.sqlite" \
    "${PREPARED_DIR}/test.sqlite" \
    "${PREPARED_DIR}/final_unseen_data.sqlite" \
  --output "${EMBEDDINGS_DB}" \
  --device cuda \
  --token-budget "${ESM_TOKEN_BUDGET:-1024}"

mkdir -p "${RUN_DIR}/pretrained_test"
python -m bindenergy.apps.tcr_pmhc.score \
  --dataset "${PREPARED_DIR}/test.sqlite" \
  --embeddings "${EMBEDDINGS_DB}" \
  --checkpoint ckpts/model.skempi.allatom \
  --output-dir "${RUN_DIR}/pretrained_test" \
  --device cuda

python -m bindenergy.apps.tcr_pmhc.train \
  --train-db "${PREPARED_DIR}/train.sqlite" \
  --validation-db "${PREPARED_DIR}/validation.sqlite" \
  --test-db "${PREPARED_DIR}/test.sqlite" \
  --embeddings "${EMBEDDINGS_DB}" \
  --init-checkpoint ckpts/model.skempi.allatom \
  --output-dir "${RUN_DIR}/finetune" \
  --device cuda \
  --batch-size "${BATCH_SIZE:-4}" \
  --learning-rate 1e-4 \
  --epochs "${EPOCHS:-5}" \
  --patience "${PATIENCE:-2}" \
  --checkpoint-every-batches "${CHECKPOINT_EVERY_BATCHES:-1000}"

for split in test final_unseen_data; do
  python -m bindenergy.apps.tcr_pmhc.score \
    --dataset "${PREPARED_DIR}/${split}.sqlite" \
    --embeddings "${EMBEDDINGS_DB}" \
    --checkpoint "${RUN_DIR}/finetune/best.pt" \
    --output-dir "${RUN_DIR}/finetuned_${split}" \
    --device cuda
done
