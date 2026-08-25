#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./datasets/DroneVehicle}"
EXPERIMENT="${2:-shasp_DroneVehicle}"
EXTRA_ARGS=("${@:3}")
PYTHON_BIN="${SHASP_PYTHON:-python}"
PYTHON_CMD=("$PYTHON_BIN")

if ! "${PYTHON_CMD[@]}" -c "import torch" >/dev/null 2>&1; then
  CONDA_ENV="${SHASP_CONDA_ENV:-pytorch1.13-py39}"
  if command -v conda >/dev/null 2>&1 \
      && conda run -n "$CONDA_ENV" python -c "import torch" >/dev/null 2>&1; then
    PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV" python)
  elif python3 -c "import torch" >/dev/null 2>&1; then
    PYTHON_CMD=(python3)
  else
    echo "Error: no Python interpreter with PyTorch was found." >&2
    echo "Set SHASP_PYTHON=/path/to/python or SHASP_CONDA_ENV=env_name." >&2
    exit 1
  fi
fi
export PYTHONUNBUFFERED=1

"${PYTHON_CMD[@]}" -u test.py \
  --dataroot "$DATA_ROOT" \
  --name "$EXPERIMENT" \
  --model shasp \
  --dataset_mode cross_spectral \
  --pairing paired \
  --direction AtoB \
  --input_nc 3 \
  --output_nc 1 \
  --norm instance \
  --specific_dim 256 \
  --mapper_blocks 3 \
  --tone_hidden_dim 64 \
  --thermal_base_gain 0.5 \
  --preprocess none \
  --epoch latest \
  --num_test 500 \
  --eval \
  --no_dropout \
  --separate_visual_dirs \
  "${EXTRA_ARGS[@]}"
