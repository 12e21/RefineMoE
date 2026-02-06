# SM Branch Count Sweep (Training Only)

This experiment varies the number of **Sparsity-MoE experts/branches** in the
SM pipeline (PV-RCNN refinement) and trains a separate model for each setting.

Scope:

- Codebase: `mmdetection3d/` (MMDet3D)
- Baseline pipeline: SM (`SoftSparsityBranchRoiHead` + `SoftSparsityBBoxHead`)
- Variable: `num_branch == sparsity_branch_num == N`
- Training and visualization are intentionally separated. This document covers
  training only.

## Configs

Configs live in `mmdetection3d/configs/RefineMoE/`:

- `SM_branch1.py`
- `SM_branch2.py`
- `SM_branch3.py`
- `SM_branch4.py`

Each config:

- sets `model.roi_head.num_branch = N`
- sets `model.roi_head.bbox_head.sparsity_branch_num = N`
- disables visualization during training (`default_hooks.visualization.draw=False`)
- uses an isolated `work_dir`: `./work_dirs/RefineMoE/SM_branchN`

## Train

Run from `mmdetection3d/`:

```bash
python tools/train.py configs/RefineMoE/SM_branch1.py
python tools/train.py configs/RefineMoE/SM_branch2.py
python tools/train.py configs/RefineMoE/SM_branch3.py
python tools/train.py configs/RefineMoE/SM_branch4.py
```

### 8-GPU (single node)

Run from `mmdetection3d/`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash tools/dist_train.sh configs/RefineMoE/SM_branch1.py 8
bash tools/dist_train.sh configs/RefineMoE/SM_branch2.py 8
bash tools/dist_train.sh configs/RefineMoE/SM_branch3.py 8
bash tools/dist_train.sh configs/RefineMoE/SM_branch4.py 8
```

Notes:

- Changing branch count changes the model architecture; you must train each `N`
  separately (do not reuse a checkpoint trained with a different `N`).
- Distributed training changes the effective total batch size; for clean
  ablations, keep everything else constant across `N`.

## Visualization

Visualization instructions will be added after the corresponding test-side
workflow is implemented/standardized for this experiment.
