#!/usr/bin/env bash
set -euo pipefail

# Run from the DSMBind repository root inside a CUDA-enabled Vast.ai instance.
python -m pip install --upgrade pip
python -m pip install -r requirements-tcr-pmhc.txt

SRU_DIR="${SRU_DIR:-/tmp/dsmbind-sru}"
SRU_REF="${SRU_REF:-c2d44e62b90115db59ab92ccea4b2e5b77077cf9}"
if [[ ! -d "${SRU_DIR}/.git" ]]; then
  git clone https://github.com/asappresearch/sru.git "${SRU_DIR}"
fi
git -C "${SRU_DIR}" fetch origin "${SRU_REF}"
git -C "${SRU_DIR}" checkout --detach "${SRU_REF}"
python -m pip install "${SRU_DIR}"
python -m pip install -e . --no-deps

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("A CUDA-enabled PyTorch runtime is required for embedding/training")
print("GPU:", torch.cuda.get_device_name(0))
PY
