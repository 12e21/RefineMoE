#!/usr/bin/env bash
set -euo pipefail

# Evaluate KITTI val AP for SM gating ablations (epoch_2 by default).
#
# Run from mmdetection3d/:
#   bash scripts/eval_sm_gating_ablation_ap.sh

EPOCH="${EPOCH:-2}"
declare -a names=(
  "baseline"
  "tempsoftmax_tau0p5"
  "learnedrouter"
)

declare -a cfgs=(
  "configs/RefineMoE/SM_branch2.py"
  "configs/RefineMoE/SM_branch2_tempsoftmax.py"
  "configs/RefineMoE/SM_branch2_learnedrouter.py"
)

declare -a ckpts=(
  "work_dirs/RefineMoE/SM_branch2/epoch_${EPOCH}.pth"
  "work_dirs/RefineMoE/SM_branch2_tempsoftmax_tau0p5/epoch_${EPOCH}.pth"
  "work_dirs/RefineMoE/SM_branch2_learnedrouter/epoch_${EPOCH}.pth"
)

declare -a work_dirs=(
  "work_dirs/RefineMoE/SM_branch2/eval_epoch${EPOCH}"
  "work_dirs/RefineMoE/SM_branch2_tempsoftmax_tau0p5/eval_epoch${EPOCH}"
  "work_dirs/RefineMoE/SM_branch2_learnedrouter/eval_epoch${EPOCH}"
)

for i in "${!names[@]}"; do
  name="${names[$i]}"
  cfg="${cfgs[$i]}"
  ckpt="${ckpts[$i]}"
  work_dir="${work_dirs[$i]}"
  mkdir -p "${work_dir}"

  echo "[eval] ${name} cfg=${cfg} ckpt=${ckpt}"
  python tools/test.py "${cfg}" "${ckpt}" --work-dir "${work_dir}" \
    2>&1 | tee "${work_dir}/test.log"
done
