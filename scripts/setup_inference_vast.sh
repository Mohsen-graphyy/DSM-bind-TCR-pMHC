#!/usr/bin/env bash
# Run from the Git checkout root. Uses the CUDA PyTorch already installed by Vast.
set -euo pipefail
python - <<'PY'
import torch
import sys
if sys.version_info >= (3, 13):
    raise SystemExit('Use Python 3.10-3.12 on Vast for the historical NumPy/SRU stack; local packing may use newer Python.')
print('Python/PyTorch CUDA preflight:', torch.__version__, torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('No usable CUDA device; run this setup on Vast, not the local CPU computer.')
print('GPU:', torch.cuda.get_device_name(0))
PY
command -v git >/dev/null
command -v nvcc >/dev/null || { echo 'Use a CUDA devel image with nvcc (SRU compiles CUDA kernels).'; exit 1; }
python -m pip install 'numpy<2' 'scipy>=1.10' tqdm 'fair-esm==2.0.0' ninja 'setuptools>=68' wheel
if ! python -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("sru") else 1)'; then
  python -m pip install --no-deps 'git+https://github.com/asappresearch/sru.git@c2d44e62b90115db59ab92ccea4b2e5b77077cf9'
fi
# Compilation compatibility only. This does NOT fix/certify higher-order training gradients.
python - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec('sru')
kernel = Path(spec.origin).parent / 'csrc' / 'sru_cuda_kernel.cu'
source = kernel.read_text()
if '.type()' in source:
    backup = kernel.with_suffix('.cu.dsmbind-inference.bak')
    if not backup.exists():
        backup.write_text(source)
    kernel.write_text(source.replace('.type()', '.scalar_type()'))
    print('Applied SRU ScalarType compilation compatibility; backup:', backup)
PY
# The historical package metadata pins Python <3.11; modern inference is tested by
# running real model scoring. Do not install a second PyTorch or change the checkpoint.
python -m pip install --ignore-requires-python --no-deps -e DSMBind
python - <<'PY'
import torch
from sru import SRUpp
from bindenergy.models.energy import AllAtomEnergyModel
print('Real SRU/DSMBind import OK; CUDA:', torch.cuda.is_available())
PY
