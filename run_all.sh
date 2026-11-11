#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export USE_FLAX=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
BASE_PYTHON="${PYTHON:-python3}"
# An overlay inherits the working GPU stack; pip is constrained from replacing it.
if [[ ! -x .venv-molab/bin/python ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages .venv-molab
fi
PY=.venv-molab/bin/python
"$PY" scripts/check_install.py --install
"$PY" -m pytest -q -m 'not cuda'
# This internal command includes CUDA integration smoke, cache, all six cells,
# round/step resume, evaluation, reports, archive, and explicit opt-in backup.
"$PY" -m fdit.cli run --preset "${EXPERIMENT_PRESET:-fast}" --resume
