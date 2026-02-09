"""Create 1x3 mosaics for Waymo BEV comparisons.

Given three directories containing per-frame PNGs named by frame_id, this tool
finds the intersection and stitches 1x3 grids.

Run from `FSHNet/`:
  python tools/analysis_tools/mosaic_waymo_bev_1x3.py \
    --dirs results/vis_bev/waymo_val/baseline results/vis_bev/waymo_val/sm results/vis_bev/waymo_val/am \
    --labels baseline sm am \
    --out-dir results/vis_bev/waymo_val/mosaic_1x3
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mosaic 1x3 BEV images")
    parser.add_argument(
        "--dirs", nargs=3, required=True, help="Three image dirs (baseline/sm/am)"
    )
    parser.add_argument(
        "--labels",
        nargs=3,
        default=["baseline", "sm", "am"],
        help="Three titles (one per dir)",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--tile-height", type=int, default=0)
    return parser.parse_args()


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_pngs(d: str) -> List[str]:
    if not osp.isdir(d):
        raise NotADirectoryError(d)
    return [f for f in os.listdir(d) if f.lower().endswith(".png")]


def _sort_key(name: str) -> Tuple[int, str]:
    # Waymo frame_id is string; fallback to lexicographic.
    return (0, name)


def _add_title(im: Image.Image, title: str, *, bar_h: int = 28) -> Image.Image:
    w, h = im.size
    out = Image.new("RGB", (w, h + bar_h), color=(255, 255, 255))
    out.paste(im, (0, bar_h))

    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        ((w - tw) // 2, (bar_h - th) // 2),
        title,
        fill=(0, 0, 0),
        font=font,
    )
    return out


def _resize_to_height(im: Image.Image, height: int) -> Image.Image:
    if height <= 0:
        return im
    w, h = im.size
    if h == height:
        return im
    new_w = int(round(w * (height / h)))
    return im.resize((new_w, height), resample=Image.BILINEAR)


def main() -> None:
    args = _parse_args()
    _mkdir(args.out_dir)

    file_sets = [set(_list_pngs(d)) for d in args.dirs]
    common = set.intersection(*file_sets)
    names = sorted(common, key=_sort_key)
    if args.max_samples > 0:
        names = names[: args.max_samples]

    for name in names:
        tiles: List[Image.Image] = []
        for d, label in zip(args.dirs, args.labels):
            im = Image.open(osp.join(d, name)).convert("RGB")
            im = _resize_to_height(im, args.tile_height)
            im = _add_title(im, label)
            tiles.append(im)

        widths = [t.size[0] for t in tiles]
        heights = [t.size[1] for t in tiles]
        out_w = sum(widths)
        out_h = max(heights)

        canvas = Image.new("RGB", (out_w, out_h), color=(255, 255, 255))
        x = 0
        for t in tiles:
            canvas.paste(t, (x, 0))
            x += t.size[0]

        canvas.save(osp.join(args.out_dir, name))


if __name__ == "__main__":
    main()
