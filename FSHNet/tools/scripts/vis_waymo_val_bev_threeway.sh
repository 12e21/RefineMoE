#!/usr/bin/env bash
set -euo pipefail

# Render Waymo val BEV visualizations (baseline vs SM vs AM) and create 1x3 mosaics.
#
# Run from `FSHNet/`:
#   CKPT_BASELINE=/path/to/baseline.pth \
#   CKPT_SM=/path/to/sm.pth \
#   CKPT_AM=/path/to/am.pth \
#   bash tools/scripts/vis_waymo_val_bev_threeway.sh

MAX_SAMPLES="${MAX_SAMPLES:-50}"
SCORE_THR="${SCORE_THR:-0.3}"

# Stricter post-processing to reduce duplicate / overlapping boxes.
# These override cfg keys used in Detector3DTemplate.post_processing and ROI head NMS.
FINAL_SCORE_THR="${FINAL_SCORE_THR:-0.25}"
FINAL_NMS_THR="${FINAL_NMS_THR:-0.5}"
ROI_NMS_THR="${ROI_NMS_THR:-0.5}"

FIG_W="${FIG_W:-10}"
FIG_H="${FIG_H:-10}"
DPI="${DPI:-350}"
LW_GT="${LW_GT:-0.7}"
LW_PRED="${LW_PRED:-0.6}"

MAX_POINTS="${MAX_POINTS:-50000}"
POINT_SIZE="${POINT_SIZE:-0.15}"
POINT_ALPHA="${POINT_ALPHA:-0.15}"

OUT_ROOT="${OUT_ROOT:-results/vis_bev/waymo_val}"
MOSAIC_OUT="${MOSAIC_OUT:-${OUT_ROOT}/mosaic_1x3}"

# Default checkpoints live under FSHNet/ckpt.
CKPT_DIR="${CKPT_DIR:-ckpt}"

CFG_BASELINE="${CFG_BASELINE:-tools/cfgs/fshnet_rcnn_car_only_models/fshnet_two_stage_car_only.yaml}"
CFG_SM="${CFG_SM:-tools/cfgs/fshnet_rcnn_car_only_models/sm_car_only.yaml}"
CFG_AM="${CFG_AM:-tools/cfgs/fshnet_rcnn_car_only_models/am_car_only.yaml}"

# If not provided, use repo defaults.
CKPT_BASELINE="${CKPT_BASELINE:-${CKPT_DIR}/FSHNet-two-stage.pth}"
CKPT_SM="${CKPT_SM:-${CKPT_DIR}/FSHNet-SM.pth}"
CKPT_AM="${CKPT_AM:-${CKPT_DIR}/FSHNet-AM.pth}"

# Fail fast with a clear message if any ckpt is missing.
[[ -f "${CKPT_BASELINE}" ]] || { echo "Missing CKPT_BASELINE: ${CKPT_BASELINE}" 1>&2; exit 2; }
[[ -f "${CKPT_SM}" ]] || { echo "Missing CKPT_SM: ${CKPT_SM}" 1>&2; exit 2; }
[[ -f "${CKPT_AM}" ]] || { echo "Missing CKPT_AM: ${CKPT_AM}" 1>&2; exit 2; }

mkdir -p "${OUT_ROOT}"

echo "[baseline] ${CFG_BASELINE} ${CKPT_BASELINE}"
python tools/analysis_tools/vis_waymo_bev.py \
  --cfg-file "${CFG_BASELINE}" \
  --ckpt "${CKPT_BASELINE}" \
  --tag baseline \
  --out-dir "${OUT_ROOT}" \
  --max-samples "${MAX_SAMPLES}" \
  --score-thr "${SCORE_THR}" \
  --draw-points \
  --max-points "${MAX_POINTS}" \
  --point-size "${POINT_SIZE}" \
  --point-alpha "${POINT_ALPHA}" \
  --figsize "${FIG_W}" "${FIG_H}" \
  --dpi "${DPI}" \
  --lw-gt "${LW_GT}" \
  --lw-pred "${LW_PRED}" \
  --set MODEL.POST_PROCESSING.SCORE_THRESH "${FINAL_SCORE_THR}" \
       MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH "${FINAL_NMS_THR}" \
       MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_THRESH "${ROI_NMS_THR}"

echo "[sm] ${CFG_SM} ${CKPT_SM}"
python tools/analysis_tools/vis_waymo_bev.py \
  --cfg-file "${CFG_SM}" \
  --ckpt "${CKPT_SM}" \
  --tag sm \
  --out-dir "${OUT_ROOT}" \
  --max-samples "${MAX_SAMPLES}" \
  --score-thr "${SCORE_THR}" \
  --draw-points \
  --max-points "${MAX_POINTS}" \
  --point-size "${POINT_SIZE}" \
  --point-alpha "${POINT_ALPHA}" \
  --figsize "${FIG_W}" "${FIG_H}" \
  --dpi "${DPI}" \
  --lw-gt "${LW_GT}" \
  --lw-pred "${LW_PRED}" \
  --set MODEL.POST_PROCESSING.SCORE_THRESH "${FINAL_SCORE_THR}" \
       MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH "${FINAL_NMS_THR}" \
       MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_THRESH "${ROI_NMS_THR}"

echo "[am] ${CFG_AM} ${CKPT_AM}"
python tools/analysis_tools/vis_waymo_bev.py \
  --cfg-file "${CFG_AM}" \
  --ckpt "${CKPT_AM}" \
  --tag am \
  --out-dir "${OUT_ROOT}" \
  --max-samples "${MAX_SAMPLES}" \
  --score-thr "${SCORE_THR}" \
  --draw-points \
  --max-points "${MAX_POINTS}" \
  --point-size "${POINT_SIZE}" \
  --point-alpha "${POINT_ALPHA}" \
  --figsize "${FIG_W}" "${FIG_H}" \
  --dpi "${DPI}" \
  --lw-gt "${LW_GT}" \
  --lw-pred "${LW_PRED}" \
  --set MODEL.POST_PROCESSING.SCORE_THRESH "${FINAL_SCORE_THR}" \
       MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH "${FINAL_NMS_THR}" \
       MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_THRESH "${ROI_NMS_THR}"

python tools/analysis_tools/mosaic_waymo_bev_1x3.py \
  --dirs "${OUT_ROOT}/baseline" "${OUT_ROOT}/sm" "${OUT_ROOT}/am" \
  --labels baseline sm am \
  --out-dir "${MOSAIC_OUT}" \
  --max-samples "${MAX_SAMPLES}"

echo "Done. Output: ${OUT_ROOT}"
echo "Mosaic: ${MOSAIC_OUT}"
