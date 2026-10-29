#!/usr/bin/env bash
set -euo pipefail
PILOT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PILOT_REPO_ROOT}"
PILOT_OUTPUT="${PILOT_OUTPUT:-outputs/reference_pilot_01}"
python - "${PILOT_OUTPUT}/final_metrics.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    m = json.load(stream)
print(json.dumps(m, indent=2))
if m.get('recurrence_backend') != 'reference':
    raise SystemExit('Expected a reference-backend pilot.')
if m.get('best_checkpoint_is_initial', True):
    raise SystemExit('No checkpoint improved fixed validation over pretrained. Stop here and review the pilot; do not tune on Atlas.')
updates = m.get('best_head_max_abs_updates', {})
if not updates or any(value <= 0 for value in updates.values()):
    raise SystemExit('Not all energy heads changed; review gradient checks before Atlas.')
PY
FINETUNED_CHECKPOINT="${PILOT_OUTPUT}/best.pt" \
PRETRAINED_CHECKPOINT=DSMBind/ckpts/model.skempi.allatom \
WORK_DIR="${ATLAS_WORK_DIR:-outputs/atlas_reference_pilot_01}" \
  bash scripts/run_atlas_benchmark.sh
