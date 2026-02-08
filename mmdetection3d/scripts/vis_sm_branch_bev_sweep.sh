#!/usr/bin/env bash
set -euo pipefail

# Generate KITTI val BEV visualizations for SM_branch{N} and create 1x4 mosaics.
#
# Prereqs:
# - Activate the mm3d conda env.
# - Run from mmdetection3d/.
# - Checkpoints exist at: work_dirs/RefineMoE/SM_branchN/epoch_40.pth
#
# Usage:
#   bash scripts/vis_sm_branch_bev_sweep.sh
#
# Optional env vars:
#   EPOCH=40 MAX_SAMPLES=50 SCORE_THR=0.3

EPOCH="${EPOCH:-40}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"   # -1 means all
SCORE_THR="${SCORE_THR:-0.3}"

BRANCHES=(1 2 3 4)

OUT_ROOT="../results/vis_bev/kitti_val"
MOSAIC_OUT="${OUT_ROOT}/mosaic_1x4_epoch${EPOCH}"

mkdir -p "${OUT_ROOT}"

for N in "${BRANCHES[@]}"; do
  cfg="configs/RefineMoE/SM_branch${N}.py"
  ckpt="work_dirs/RefineMoE/SM_branch${N}/epoch_${EPOCH}.pth"
  out_dir="${OUT_ROOT}/sm_branch${N}_epoch${EPOCH}"

  echo "[SM_branch${N}] ${cfg} ${ckpt} -> ${out_dir}"

  python tools/analysis_tools/vis_sm_branch_bev.py \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --out-dir "${out_dir}" \
    --max-samples "${MAX_SAMPLES}" \
    --score-thr "${SCORE_THR}" \
    --draw-points
done

python tools/analysis_tools/mosaic_sm_branch_bev.py \
  --dirs \
    "${OUT_ROOT}/sm_branch1_epoch${EPOCH}" \
    "${OUT_ROOT}/sm_branch2_epoch${EPOCH}" \
    "${OUT_ROOT}/sm_branch3_epoch${EPOCH}" \
    "${OUT_ROOT}/sm_branch4_epoch${EPOCH}" \
  --labels branch1 branch2 branch3 branch4 \
  --out-dir "${MOSAIC_OUT}" \
  --max-samples "${MAX_SAMPLES}"

echo "Done. Mosaic dir: ${MOSAIC_OUT}"
