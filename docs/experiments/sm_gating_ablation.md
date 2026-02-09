# SM Gating Ablation (Branch=2, KITTI)

This document introduces two alternative gating mechanisms for the SM pipeline
to address reviewer feedback (learned routing and temperature-softmax).

Scope:

- Codebase: `mmdetection3d/`
- Pipeline: SM (`SoftSparsityBranchRoiHead` + `SoftSparsityBBoxHead`)
- Branch count: fixed to 2 (no branch-count ablation)

## Baseline (Current)

Baseline SM uses sparsity-based gating derived from the number of points inside
each RoI (proposal box):

1) Count points per RoI.
2) Convert the scalar count to per-branch scores using fixed-interval Gaussian
   scoring.
3) Use these scores:
   - during training: as per-RoI loss weights (see `SoftSparsityBBoxHead.loss`)
   - during inference: normalize scores across branches and use them to fuse the
     per-branch bbox head outputs.

## Alternative 1: Temperature-softmax gating

We keep the same Gaussian scoring prior, but convert per-branch scores to a
temperature-controlled distribution:

  w = softmax(log(score + eps) / tau)

Smaller `tau` makes routing sharper (more confident branch selection); larger
`tau` makes routing smoother.

Config:

- `configs/RefineMoE/SM_branch2_tempsoftmax.py` (tau=0.5)

## Alternative 2: Learned router gating (learns sparsity)

We replace the hand-crafted mapping from point-count to branch weights with a
small MLP router. To ensure the router learns sparsity, its input is:

  z = log(points_count + 1)

The router outputs logits over branches and uses a softmax (optionally with
temperature). The resulting per-branch weights are used both for training loss
weighting and inference fusion.

Config:

- `configs/RefineMoE/SM_branch2_learnedrouter.py`

## Implementation Notes

New gating implementations are kept in a standalone module:

- `mmdet3d/models/roi_heads/gating/sm_gating.py`

`SoftSparsityBranchRoiHead` accepts an optional `gating_cfg` to switch gating
without touching other model components.

## Run (4 GPUs)

Run from `mmdetection3d/`.

Train (epoch=40 schedule):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# baseline
bash tools/dist_train.sh configs/RefineMoE/SM_branch2.py 4

# temperature-softmax
bash tools/dist_train.sh configs/RefineMoE/SM_branch2_tempsoftmax.py 4

# learned router
bash tools/dist_train.sh configs/RefineMoE/SM_branch2_learnedrouter.py 4
```

Evaluate AP on KITTI val (uses epoch_40 checkpoints):

```bash
python tools/test.py configs/RefineMoE/SM_branch2.py \
  work_dirs/RefineMoE/SM_branch2/epoch_40.pth

python tools/test.py configs/RefineMoE/SM_branch2_tempsoftmax.py \
  work_dirs/RefineMoE/SM_branch2_tempsoftmax_tau0p5/epoch_40.pth

python tools/test.py configs/RefineMoE/SM_branch2_learnedrouter.py \
  work_dirs/RefineMoE/SM_branch2_learnedrouter/epoch_40.pth
```
