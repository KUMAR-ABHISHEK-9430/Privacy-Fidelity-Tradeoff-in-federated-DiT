#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
MODE="${1:-diagnose}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$MODE" != diagnose && "$MODE" != run ]]; then
  echo "Usage: bash run_all.sh [diagnose|run] [CLI options]" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$PWD/.cache/pip"
mkdir -p "$PIP_CACHE_DIR" .cache/logs
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export USE_FLAX=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
BASE_PYTHON="${PYTHON:-python3}"
# Inherit the actual launcher venv (including Molab's /tmp/uv-venv), not just
# its underlying Python install. Preserve old overlays if the interpreter changes.
PY="$("$BASE_PYTHON" scripts/prepare_env.py)"
"$PY" scripts/check_install.py --install
"$PY" -u -m pytest -q -m 'not cuda'
if [[ "$MODE" == diagnose ]]; then
  echo "Running diagnostics only; the six-condition federated suite is disabled in this mode."
  "$PY" -u -m fdit.cli diagnose --steps "${DIAGNOSTIC_STEPS:-1000}" --resume "$@" 2>&1 | tee -a .cache/logs/diagnostic.log
else
  "$PY" -u -m fdit.cli run --preset "${EXPERIMENT_PRESET:-fast}" --resume "$@" 2>&1 | tee -a .cache/logs/federated.log
fi
