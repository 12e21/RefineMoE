"""Render Waymo val BEV images with pred + GT overlays.

This is a standalone analysis tool (does not modify training/eval).

Run from `FSHNet/`:
  python tools/analysis_tools/vis_waymo_bev.py \
    --cfg-file tools/cfgs/fshnet_rcnn_car_only_models/fshnet_two_stage_car_only.yaml \
    --ckpt /path/to/checkpoint.pth \
    --tag baseline \
    --out-dir results/vis_bev/waymo_val \
    --max-samples 50 \
    --draw-points
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

# Make `pcdet` importable when running as `python tools/analysis_tools/*.py`.
import sys

_FSHNET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_FSHNET_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Waymo val BEV (points + GT + preds)"
    )
    parser.add_argument("--cfg-file", required=True, help="Path to a YAML config")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--tag", default="default", help="Subdir name under out-dir")
    parser.add_argument("--out-dir", required=True, help="Output root directory")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for point subsampling"
    )

    # Allow overriding cfg keys (OpenPCDet-style), e.g.:
    #   --set MODEL.POST_PROCESSING.SCORE_THRESH 0.3 MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH 0.5
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="Set extra config keys if needed",
    )

    # Rendering knobs
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[8.0, 8.0],
        help="Figure size in inches, e.g. --figsize 8 8",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--lw-gt", type=float, default=0.8)
    parser.add_argument("--lw-pred", type=float, default=0.7)
    parser.add_argument(
        "--draw-points",
        action="store_true",
        help="Draw BEV point cloud as background scatter",
    )
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--point-size", type=float, default=0.15)
    parser.add_argument("--point-alpha", type=float, default=0.15)
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        help="Override BEV x-axis limits (meters)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        help="Override BEV y-axis limits (meters)",
    )
    return parser.parse_args()


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sanitize_frame_id(frame_id: str) -> str:
    # Keep filenames portable.
    s = str(frame_id)
    s = s.replace(os.sep, "_").replace(" ", "_")
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return s


def _bev_polys_from_boxes_lidar(boxes_lidar: torch.Tensor) -> np.ndarray:
    """Convert lidar boxes (N, 7) -> polygons (N, 4, 2) in XY."""
    if boxes_lidar is None or boxes_lidar.numel() == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)

    # (N,): x, y, dx, dy, yaw
    ctr = boxes_lidar[:, 0:2]
    dx = boxes_lidar[:, 3]
    dy = boxes_lidar[:, 4]
    yaw = boxes_lidar[:, 6]

    base = torch.tensor(
        [[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]],
        device=boxes_lidar.device,
        dtype=boxes_lidar.dtype,
    )
    dims = torch.stack([dx, dy], dim=-1)  # (N, 2)
    corners = base[None, :, :] * dims[:, None, :]  # (N, 4, 2)

    c = torch.cos(yaw)
    s = torch.sin(yaw)
    rot = torch.stack(
        [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2
    )  # (N, 2, 2)
    corners = torch.matmul(corners, rot.transpose(-1, -2))
    corners = corners + ctr[:, None, :]
    return corners.detach().cpu().numpy().astype(np.float32)


def _draw_polys(ax, polys: np.ndarray, *, color: str, lw: float) -> None:
    for poly in polys:
        xs = np.concatenate([poly[:, 0], poly[:1, 0]])
        ys = np.concatenate([poly[:, 1], poly[:1, 1]])
        ax.plot(xs, ys, color=color, linewidth=lw)


def _extract_points_xy(batch_points: torch.Tensor, batch_idx: int) -> np.ndarray:
    # batch_points: (N, >=4) [batch_idx, x, y, z, ...]
    mask = batch_points[:, 0].to(dtype=torch.long) == int(batch_idx)
    pts = batch_points[mask, 1:3]
    return pts.detach().cpu().numpy().astype(np.float32)


def _extract_gt_boxes_lidar(
    batch_gt_boxes: torch.Tensor, batch_idx: int
) -> torch.Tensor:
    # gt_boxes: (B, M, 8) [x,y,z,dx,dy,dz,heading,class]
    gt = batch_gt_boxes[batch_idx]
    # Filter padding boxes.
    if gt.shape[-1] >= 8:
        valid = gt[:, 7] > 0
    else:
        valid = gt[:, 3:6].sum(dim=-1) > 0
    gt = gt[valid]
    return gt[:, :7]


def _default_xy_limits() -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # cfg.DATA_CONFIG.POINT_CLOUD_RANGE: [xmin, ymin, zmin, xmax, ymax, zmax]
    data_cfg = cfg.get("DATA_CONFIG", {})
    pcr = data_cfg.get("POINT_CLOUD_RANGE", None)
    if pcr is None:
        return ((-80.0, 80.0), (-80.0, 80.0))
    return ((float(pcr[0]), float(pcr[3])), (float(pcr[1]), float(pcr[4])))


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

    cfg_from_yaml_file(args.cfg_file, cfg)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[-2:-1])
    cfg.LOCAL_RANK = 0

    out_root = osp.join(args.out_dir, args.tag)
    _mkdir(out_root)
    log_file = osp.join(out_root, "vis_waymo_bev.log")
    logger = common_utils.create_logger(log_file, rank=0)

    class_names_any = cfg.get("CLASS_NAMES")
    assert class_names_any is not None, "CLASS_NAMES missing from cfg"
    class_names: Sequence[str] = list(class_names_any)

    # Force a simple, deterministic val loader for visualization.
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.get("DATA_CONFIG"),
        class_names=class_names,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    model = build_network(
        model_cfg=cfg.get("MODEL"),
        num_class=len(class_names),
        dataset=test_set,
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    (xlim_def, ylim_def) = _default_xy_limits()
    xlim = tuple(args.xlim) if args.xlim is not None else xlim_def
    ylim = tuple(args.ylim) if args.ylim is not None else ylim_def

    max_samples = int(args.max_samples)
    max_points = int(args.max_points)

    logger.info("cfg=%s", args.cfg_file)
    logger.info("ckpt=%s", args.ckpt)
    logger.info("out=%s", out_root)
    logger.info("xlim=%s ylim=%s", xlim, ylim)
    logger.info("max_samples=%s score_thr=%s", max_samples, args.score_thr)

    seen = 0
    for batch_dict in test_loader:
        load_data_to_gpu(batch_dict)
        pred_dicts, _ = model(batch_dict)

        # batch_size forced to 1
        frame_id = batch_dict.get("frame_id", ["unknown"])
        if isinstance(frame_id, (list, tuple)):
            frame_id = frame_id[0]

        pts_xy = _extract_points_xy(batch_dict["points"], 0)
        if args.draw_points and max_points > 0 and pts_xy.shape[0] > max_points:
            idx = np.random.choice(pts_xy.shape[0], max_points, replace=False)
            pts_xy = pts_xy[idx]

        gt_boxes = None
        if "gt_boxes" in batch_dict:
            gt_boxes = _extract_gt_boxes_lidar(batch_dict["gt_boxes"], 0)

        pred_boxes = pred_dicts[0].get("pred_boxes", None)
        pred_scores = pred_dicts[0].get("pred_scores", None)
        if pred_boxes is None or pred_boxes.numel() == 0:
            pred_keep = None
        elif pred_scores is not None:
            pred_keep = pred_scores > float(args.score_thr)
        else:
            pred_keep = None

        if pred_keep is not None:
            pred_boxes = pred_boxes[pred_keep]

        pred_polys = _bev_polys_from_boxes_lidar(pred_boxes)
        gt_polys = (
            _bev_polys_from_boxes_lidar(gt_boxes)
            if gt_boxes is not None
            else np.zeros((0, 4, 2), dtype=np.float32)
        )

        fig: Figure = plt.figure(
            figsize=(args.figsize[0], args.figsize[1]), dpi=args.dpi
        )
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        if args.draw_points:
            ax.scatter(
                pts_xy[:, 0],
                pts_xy[:, 1],
                s=float(args.point_size),
                c="#808080",
                alpha=float(args.point_alpha),
                linewidths=0,
            )

        # Draw GT first then preds.
        _draw_polys(ax, gt_polys, color="#00aa00", lw=float(args.lw_gt))
        _draw_polys(ax, pred_polys, color="#ff7a00", lw=float(args.lw_pred))

        name = _sanitize_frame_id(frame_id)
        out_path = osp.join(out_root, f"{name}.png")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        seen += 1
        if max_samples > 0 and seen >= max_samples:
            break

    logger.info("Done. Rendered %d samples", seen)


if __name__ == "__main__":
    main()
