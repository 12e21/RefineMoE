# KITTI Val Efficiency (RTX 4090, FP32, batch=1)

This document records efficiency metrics for baseline vs RefineMoE variants.

Benchmark contract:

- GPU: RTX 4090 (single GPU)
- Precision: FP32 (AMP/FP16 disabled)
- Batch size: 1 (forced)
- Inference: end-to-end includes dataloader + H2D + forward + post-processing
- Training: end-to-end includes dataloader + H2D + forward + backward + step
- Dataset: KITTI val split

How to run:

```bash
# PV-RCNN (mm3d)
cd mmdetection3d
python tools/bench_efficiency.py \
  --configs \
    configs/RefineMoE/pv_rcnn.py \
    configs/RefineMoE/AM.py \
    configs/RefineMoE/SM.py \
  --batch-size 1 --warmup 50 --iters 200 \
  --out ../results/bench/kitti_val_mm3d.json

# VoxelRCNN (FSHNet)
cd ../FSHNet
python tools/bench_efficiency.py \
  --cfgs \
    tools/cfgs/voxelrcnn_kitti_models/voxelrcnn_kitti.yaml \
    tools/cfgs/voxelrcnn_kitti_models/am_kitti.yaml \
    tools/cfgs/voxelrcnn_kitti_models/sm_kitti.yaml \
  --batch_size 1 --warmup 50 --iters 200 \
  --out ../results/bench/kitti_val_fshnet.json
```

## PV-RCNN (MMDet3D)

| Variant | Params (M) | Infer FPS | Infer peak VRAM (MiB) | Train sec/iter | Train peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|
| Baseline (`pv_rcnn.py`) |  |  |  |  |  |
| AM (`AM.py`) |  |  |  |  |  |
| SM (`SM.py`) |  |  |  |  |  |

## VoxelRCNN (FSHNet)

| Variant | Params (M) | Infer FPS | Infer peak VRAM (MiB) | Train sec/iter | Train peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|
| Baseline (`voxelrcnn_kitti.yaml`) |  |  |  |  |  |
| AM (`am_kitti.yaml`) |  |  |  |  |  |
| SM (`sm_kitti.yaml`) |  |  |  |  |  |
