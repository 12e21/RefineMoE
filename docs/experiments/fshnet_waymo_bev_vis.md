# FSHNet Waymo Val BEV Visualization (Baseline vs SM vs AM)

This experiment renders Bird's-Eye-View (BEV) images on the Waymo validation
split for three FSHNet variants:

- two-stage baseline
- FSHNet-SM (Sparsity-MoE)
- FSHNet-AM (Attribute-MoE)

Each frame is rendered with:

- background point cloud scatter (XY)
- GT boxes overlay
- predicted boxes overlay (filtered by a score threshold)

The implementation follows the OpenPCDet/PCDet data structures (from `FSHNet/`),
not the MMDet3D runner/hook system.

## Configs

All three configs are Waymo-based (`_BASE_CONFIG_` points to
`tools/cfgs/dataset_configs/waymo_dataset.yaml`).

- Baseline (two-stage): `FSHNet/tools/cfgs/fshnet_rcnn_car_only_models/fshnet_two_stage_car_only.yaml`
- SM: `FSHNet/tools/cfgs/fshnet_rcnn_car_only_models/sm_car_only.yaml`
- AM: `FSHNet/tools/cfgs/fshnet_rcnn_car_only_models/am_car_only.yaml`

## Tools

- Renderer: `FSHNet/tools/analysis_tools/vis_waymo_bev.py`
- Mosaic (1x3): `FSHNet/tools/analysis_tools/mosaic_waymo_bev_1x3.py`
- Three-way runner: `FSHNet/tools/scripts/vis_waymo_val_bev_threeway.sh`

Dependencies:

- `matplotlib` (renderer)
- `Pillow` (mosaic)

## Run (Three-Way)

Run from the `FSHNet/` directory:

```bash
cd FSHNet

bash tools/scripts/vis_waymo_val_bev_threeway.sh
```

By default, the script expects the following checkpoints under `FSHNet/ckpt/`:

- `ckpt/FSHNet-two-stage.pth`
- `ckpt/FSHNet-SM.pth`
- `ckpt/FSHNet-AM.pth`

You can override them:

```bash
cd FSHNet

CKPT_BASELINE=/path/to/baseline.pth \
CKPT_SM=/path/to/sm.pth \
CKPT_AM=/path/to/am.pth \
bash tools/scripts/vis_waymo_val_bev_threeway.sh
```

Defaults:

- `MAX_SAMPLES=50`
- `SCORE_THR=0.3`
- `OUT_ROOT=results/vis_bev/waymo_val`

To reduce overlapping boxes in visualization, the runner also overrides the
model-side post-processing by default:

- `FINAL_SCORE_THR=0.25` (maps to `MODEL.POST_PROCESSING.SCORE_THRESH`)
- `FINAL_NMS_THR=0.5` (maps to `MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH`)
- `ROI_NMS_THR=0.5` (maps to `MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_THRESH`)

Outputs:

- `results/vis_bev/waymo_val/baseline/*.png`
- `results/vis_bev/waymo_val/sm/*.png`
- `results/vis_bev/waymo_val/am/*.png`
- `results/vis_bev/waymo_val/mosaic_1x3/*.png`

You can override common knobs via environment variables, for example:

```bash
cd FSHNet

MAX_SAMPLES=50 SCORE_THR=0.25 MAX_POINTS=80000 POINT_ALPHA=0.12 \
CKPT_BASELINE=/path/to/baseline.pth \
CKPT_SM=/path/to/sm.pth \
CKPT_AM=/path/to/am.pth \
bash tools/scripts/vis_waymo_val_bev_threeway.sh
```

## Run (Single Variant)

Render only one variant (example: SM):

```bash
cd FSHNet

python tools/analysis_tools/vis_waymo_bev.py \
  --cfg-file tools/cfgs/fshnet_rcnn_car_only_models/sm_car_only.yaml \
  --ckpt /path/to/sm.pth \
  --tag sm \
  --out-dir results/vis_bev/waymo_val \
  --max-samples 50 \
  --score-thr 0.3 \
  --draw-points
```

Notes:

- BEV limits default to `DATA_CONFIG.POINT_CLOUD_RANGE` in the config.
- For speed and simpler file writing, the renderer uses `batch_size=1` and
  non-distributed dataloader by default.
