#!/usr/bin/env bash
set -euo pipefail
FULL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FULL_REPO_ROOT}"
export PYTHONPATH="${FULL_REPO_ROOT}/DSMBind${PYTHONPATH:+:${PYTHONPATH}}"
FULL_DATA="${TRAINING_DATA:-data/dsmbind_tcr_pmhc}"
FULL_OUTPUT="${TRAINING_OUTPUT:-outputs/reference_full_01}"
FULL_ESM="${ESM_CACHE:-data/esm_cache/esm2_t36_3B.sqlite}"
python - "${FULL_DATA}" "${FULL_OUTPUT}" "${RESUME_CHECKPOINT:-}" <<'PY'
import pathlib, sqlite3, sys
data, output = map(pathlib.Path, sys.argv[1:3])
if not sys.argv[3] and output.exists() and any(output.iterdir()):
    raise SystemExit('Choose a new TRAINING_OUTPUT or resume last.pt in the same output directory.')
for name in ('train', 'validation'):
    path = data / (name + '.sqlite')
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        splits = {r[0] for r in connection.execute('SELECT DISTINCT split FROM pairs')}
        if splits != {name}:
            raise SystemExit(f'Expected only {name} pairs in {path}, got {splits}')
        print(name, 'structures=', connection.execute('SELECT COUNT(*) FROM structures').fetchone()[0],
              'pairs=', connection.execute('SELECT COUNT(*) FROM pairs').fetchone()[0], flush=True)
    finally:
        connection.close()
PY
mkdir -p "$(dirname "${FULL_OUTPUT}")"
python -u -m bindenergy.apps.tcr_pmhc.embeddings \
  --datasets "${FULL_DATA}/train.sqlite" "${FULL_DATA}/validation.sqlite" \
  --output "${FULL_ESM}" --device cuda --token-budget "${ESM_TOKEN_BUDGET:-512}" \
  2>&1 | tee -a "${FULL_OUTPUT}.embeddings.log"
FULL_ARGS=(
  --train-db "${FULL_DATA}/train.sqlite"
  --validation-db "${FULL_DATA}/validation.sqlite"
  --embeddings "${FULL_ESM}"
  --init-checkpoint DSMBind/ckpts/model.skempi.allatom
  --output-dir "${FULL_OUTPUT}" --recurrence-backend reference --device cuda
  --batch-size 1 --learning-rate "${LEARNING_RATE:-1e-6}"
  --epochs "${EPOCHS:-3}" --patience 0 --anneal-rate 1.0
  --validation-batches "${VALIDATION_BATCHES:-512}"
  --validate-every-batches "${VALIDATE_EVERY_BATCHES:-2000}"
  --checkpoint-every-batches 500 --seed 7
)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  FULL_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi
python -u -m bindenergy.apps.tcr_pmhc.train "${FULL_ARGS[@]}" 2>&1 | tee -a "${FULL_OUTPUT}.log"
echo "Training finished. Selected checkpoint: ${FULL_OUTPUT}/best.pt; inspect final_metrics.json."
