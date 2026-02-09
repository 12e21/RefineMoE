#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper kept under FSHNet/scripts/.
# Actual implementation lives in tools/scripts/.
#
# Run from `FSHNet/`:
#   bash scripts/vis_waymo_val_bev_threeway.sh
#
# Common env overrides:
#   (This wrapper sets defaults directly below; you can still override by
#   exporting variables before running this script.)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,40p' "${BASH_SOURCE[0]}"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSHNET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default experiment knobs (edit here for your preferred settings).
export MAX_SAMPLES="${MAX_SAMPLES:-50}"
export SCORE_THR="${SCORE_THR:-0.35}"
export FINAL_SCORE_THR="${FINAL_SCORE_THR:-0.35}"
export FINAL_NMS_THR="${FINAL_NMS_THR:-0.4}"
export ROI_NMS_THR="${ROI_NMS_THR:-0.4}"
export CKPT_DIR="${CKPT_DIR:-ckpt}"

exec bash "${FSHNET_ROOT}/tools/scripts/vis_waymo_val_bev_threeway.sh" "$@"
