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
#   EPOCH=40            # checkpoint epoch number (uses epoch_${EPOCH}.pth)
#   MAX_SAMPLES=50      # max frames to render per branch (-1 means all KITTI val)
#   SCORE_THR=0.3       # prediction score threshold for drawing boxes
#
# Rendering knobs:
#   FIG_W=10            # figure width (inches)
#   FIG_H=8             # figure height (inches)
#   DPI=400             # output image DPI (higher => higher resolution)
#   LW_GT=0.7           # GT box line width
#   LW_PRED=0.6         # pred box line width
#
#   MAX_POINTS=50000     # max points to draw per frame (randomly subsampled)
#   POINT_SIZE=0.15      # point marker size
#   POINT_ALPHA=0.15     # point marker alpha (transparency)

EPOCH="${EPOCH:-40}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"   # -1 means all
SCORE_THR="${SCORE_THR:-0.3}"

FIG_W="${FIG_W:-10}"
FIG_H="${FIG_H:-8}"
DPI="${DPI:-400}"
LW_GT="${LW_GT:-0.7}"
LW_PRED="${LW_PRED:-0.6}"

MAX_POINTS="${MAX_POINTS:-50000}"
POINT_SIZE="${POINT_SIZE:-0.15}"
POINT_ALPHA="${POINT_ALPHA:-0.5}"

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
    --draw-points \
    --max-points "${MAX_POINTS}" \
    --point-size "${POINT_SIZE}" \
    --point-alpha "${POINT_ALPHA}" \
    --figsize "${FIG_W}" "${FIG_H}" \
    --dpi "${DPI}" \
    --lw-gt "${LW_GT}" \
    --lw-pred "${LW_PRED}"
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
