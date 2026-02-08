# Copyright (c) OpenMMLab. All rights reserved.
"""Render KITTI val BEV images with pred+GT (for SM branch sweep).

This script is intentionally independent from training:
- Loads model from config + checkpoint.
- Builds a *vis* dataloader that includes GT annotations.
- Runs inference (includes model post-processing) and renders BEV overlays.

Run from `mmdetection3d/`.
"""

import argparse
import os
import os.path as osp
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize KITTI val BEV pred+GT for a checkpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument(
        "--draw-points",
        action="store_true",
        help="Draw BEV point cloud as background scatter",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Max points to draw per frame (randomly subsampled)",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.15,
        help="Scatter marker size for points",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.15,
        help="Scatter alpha for points",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=[0.0, 70.4],
        help="BEV x-axis limits (meters)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=[-40.0, 40.0],
        help="BEV y-axis limits (meters)",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default=None,
        help="Override config options (key=value)",
    )
    return parser.parse_args()


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _bev_polys_from_boxes(boxes_3d) -> np.ndarray:
    """Return (N, 4, 2) array from BaseInstance3DBoxes.

    Use BEV (x, y, dx, dy, yaw) to build an ordered rectangle.
    This avoids relying on the corner indexing/order of `boxes_3d.corners`.
    """
    if boxes_3d is None or len(boxes_3d) == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)

    bev = boxes_3d.bev  # (N, 5): x, y, dx, dy, yaw
    num = bev.shape[0]
    centers = bev[:, 0:2]
    dx = bev[:, 2]
    dy = bev[:, 3]
    yaw = bev[:, 4]

    # (4, 2) corners in box local frame (x forward, y left).
    # Order is consistent for drawing a closed polygon.
    base = torch.tensor(
        [[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]],
        device=bev.device,
        dtype=bev.dtype,
    )
    # (N, 4, 2)
    dims = torch.stack([dx, dy], dim=-1)
    corners = base[None, :, :] * dims[:, None, :]

    c = torch.cos(yaw)
    s = torch.sin(yaw)
    rot = torch.stack(
        [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)],
        dim=-2,
    )  # (N, 2, 2)
    corners = torch.matmul(corners, rot.transpose(-1, -2))
    corners = corners + centers[:, None, :]
    return corners.detach().cpu().numpy().astype(np.float32)


def _draw_polys(ax, polys: np.ndarray, *, color: str, lw: float) -> None:
    for poly in polys:
        xs = np.concatenate([poly[:, 0], poly[:1, 0]])
        ys = np.concatenate([poly[:, 1], poly[:1, 1]])
        ax.plot(xs, ys, color=color, linewidth=lw)


def _points_xy(points) -> np.ndarray:
    """Extract (N, 2) XY from a BasePoints or Tensor."""
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)
    if hasattr(points, "tensor"):
        pts = points.tensor
    else:
        pts = points
    xy = pts[:, 0:2].detach().cpu().numpy().astype(np.float32)
    return xy


def _get_sample_idx(data_sample) -> str:
    # Kitti metrics and datasets use `sample_idx`.
    sample_idx = None
    try:
        sample_idx = data_sample["sample_idx"]
    except Exception:
        pass
    if sample_idx is None:
        sample_idx = data_sample.metainfo.get("sample_idx", None)
    if isinstance(sample_idx, (int, np.integer)):
        return f"{int(sample_idx):06d}"
    return str(sample_idx) if sample_idx is not None else "unknown"


def _build_vis_pipeline(cfg: Config) -> List[dict]:
    """Build a deterministic pipeline that includes GT for visualization."""
    # Start from test pipeline defined in config (already deterministic).
    pipeline = list(cfg.test_dataloader.dataset.pipeline)

    # Insert GT loading right after loading points.
    # test pipeline begins with LoadPointsFromFile.
    insert_idx = 1
    pipeline.insert(
        insert_idx,
        dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    )

    # Ensure Pack3DDetInputs includes GT keys.
    for t in pipeline:
        if t.get("type") == "Pack3DDetInputs":
            keys = t.get("keys", [])
            for k in ["points", "gt_bboxes_3d", "gt_labels_3d"]:
                if k not in keys:
                    keys.append(k)
            t["keys"] = keys
    return pipeline


@torch.no_grad()
def main() -> None:
    args = parse_args()
    init_default_scope("mmdet3d")

    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # Build a visualization dataloader (needs ann_info, so disable test_mode).
    cfg.test_dataloader.dataset.test_mode = False
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.sampler = dict(type="DefaultSampler", shuffle=False)
    cfg.test_dataloader.dataset.pipeline = _build_vis_pipeline(cfg)

    # Work dir only for runner internal artifacts.
    if cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", "_vis_tmp")
    cfg.load_from = args.checkpoint

    runner = Runner.from_cfg(cfg)
    # `Runner.from_cfg` does not automatically load weights; `tools/test.py`
    # loads via `runner.test()`. Since we call `model.test_step()` directly,
    # load checkpoint explicitly here.
    runner.load_checkpoint(args.checkpoint)
    model = runner.model
    model.eval()

    out_dir = args.out_dir
    _mkdir(out_dir)

    dataloader = runner.test_dataloader

    max_samples = args.max_samples
    seen = 0

    for data_batch in dataloader:
        pred_samples = model.test_step(data_batch)
        # `data_batch` includes `data_samples` for GT.
        gt_samples = data_batch.get("data_samples", None)
        if gt_samples is None:
            # Fallback: pred samples may still carry GT if packed.
            gt_samples = pred_samples

        for pred, gt in zip(pred_samples, gt_samples):
            sample_id = _get_sample_idx(pred)

            pred_inst = pred.pred_instances_3d
            gt_inst = gt.gt_instances_3d

            # Filter predictions by score.
            keep = None
            if hasattr(pred_inst, "scores_3d") and pred_inst.scores_3d is not None:
                keep = pred_inst.scores_3d > args.score_thr

            pred_boxes = (
                pred_inst.bboxes_3d[keep] if keep is not None else pred_inst.bboxes_3d
            )
            gt_boxes = gt_inst.bboxes_3d

            pred_polys = _bev_polys_from_boxes(pred_boxes)
            gt_polys = _bev_polys_from_boxes(gt_boxes)

            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
            ax.set_xlim(args.xlim[0], args.xlim[1])
            ax.set_ylim(args.ylim[0], args.ylim[1])
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

            if args.draw_points:
                pts_xy = _points_xy(data_batch["inputs"]["points"][0])
                if args.max_points > 0 and pts_xy.shape[0] > args.max_points:
                    idx = np.random.choice(
                        pts_xy.shape[0], args.max_points, replace=False
                    )
                    pts_xy = pts_xy[idx]
                ax.scatter(
                    pts_xy[:, 0],
                    pts_xy[:, 1],
                    s=args.point_size,
                    c="#808080",
                    alpha=args.point_alpha,
                    linewidths=0,
                )

            # Draw GT then pred on top.
            _draw_polys(ax, gt_polys, color="#00aa00", lw=1.2)
            _draw_polys(ax, pred_polys, color="#cc0000", lw=1.0)

            out_path = osp.join(out_dir, f"{sample_id}.png")
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)

            seen += 1
            if max_samples > 0 and seen >= max_samples:
                return


if __name__ == "__main__":
    main()
