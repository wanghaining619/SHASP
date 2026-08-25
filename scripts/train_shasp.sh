#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./datasets/DroneVehicle}"
EXPERIMENT="${2:-shasp_DroneVehicle}"
WARMSTART_CHECKPOINT="${3:-}"
EXTRA_ARGS=("${@:4}")
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

WARMSTART_ARGS=()
if [ -n "$WARMSTART_CHECKPOINT" ]; then
  WARMSTART_ARGS=(--warmstart_checkpoint "$WARMSTART_CHECKPOINT")
fi

"${PYTHON_CMD[@]}" -u train.py \
  --dataroot "$DATA_ROOT" \
  --name "$EXPERIMENT" \
  --model shasp \
  --dataset_mode cross_spectral \
  --pairing paired \
  --direction AtoB \
  --input_nc 3 \
  --output_nc 1 \
  --norm instance \
  --preprocess crop \
  --crop_size 512 \
  --batch_size 1 \
  --pool_size 0 \
  --gan_mode lsgan \
  --specific_dim 256 \
  --mapper_blocks 3 \
  --tone_hidden_dim 64 \
  --thermal_base_gain 0.5 \
  --global_warmup_epochs 5 \
  --lambda_specific 2 \
  --lambda_semantic 0.5 \
  --lambda_cycle 5 \
  --lambda_adversarial 1 \
  --lambda_paired 20 \
  --lambda_gradient 5 \
  --lambda_intensity 1 \
  --lambda_bright 5 \
  --bright_threshold 0.8 \
  --paired_warmup_epochs 5 \
  --gan_ramp_epochs 5 \
  --lr 0.0001 \
  --lr_policy linear \
  --n_epochs 100 \
  --n_epochs_decay 30 \
  --print_freq 20 \
  --display_freq 200 \
  --update_html_freq 200 \
  --save_latest_freq 200 \
  --save_epoch_freq 5 \
  --display_id -1 \
  "${WARMSTART_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
