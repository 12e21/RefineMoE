#!/usr/bin/env bash
set -euo pipefail

# Train SM gating ablations (branch=2) on 4 GPUs.
#
# NOTE: This script forces a short run (2 epochs) to speed up ablations.
#
# Run from mmdetection3d/:
#   bash scripts/train_sm_gating_ablation_4g.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

CFG_DIR="configs/RefineMoE"

cfgs=(
  "${CFG_DIR}/SM_branch2.py"
  "${CFG_DIR}/SM_branch2_tempsoftmax.py"
  "${CFG_DIR}/SM_branch2_learnedrouter.py"
)

for cfg in "${cfgs[@]}"; do
  echo "[train] ${cfg}"
  bash tools/dist_train.sh "${cfg}" 4 \
    --cfg-options train_cfg.max_epochs=2
done
