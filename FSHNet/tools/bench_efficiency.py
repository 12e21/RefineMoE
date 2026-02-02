import _init_path  # noqa: F401

"""End-to-end efficiency benchmark for FSHNet (OpenPCDet-style).

Contract (for this repo):
- FP32 only (no AMP).
- batch_size forced by CLI (recommended: 1).
- Inference includes post_processing (NMS) via model(batch_dict) in eval mode.
- End-to-end timing includes dataloader + H2D + forward + postprocess.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from pathlib import Path

import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end efficiency for FSHNet"
    )
    parser.add_argument(
        "--cfgs", nargs="+", required=True, help="One or more YAML config paths"
    )
    parser.add_argument(
        "--ckpt", default=None, help="Optional checkpoint to load for all configs"
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    parser.add_argument(
        "--out", default="results/bench/kitti_val_fshnet.json", help="Output JSON path"
    )
    return parser.parse_args()


def _mkdir_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def _bench_loop(
    *,
    step_fn,
    dataloader,
    warmup: int,
    iters: int,
    use_cuda: bool,
) -> Dict[str, float]:
    data_iter = iter(dataloader)
    for _ in range(warmup):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        if use_cuda:
            torch.cuda.synchronize()
        step_fn(batch)
        if use_cuda:
            torch.cuda.synchronize()

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    total = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        if use_cuda:
            torch.cuda.synchronize()
        step_fn(batch)
        if use_cuda:
            torch.cuda.synchronize()
        total += time.perf_counter() - t0

    sec_per_iter = total / iters
    fps = iters / total
    peak_alloc = (
        float(torch.cuda.max_memory_allocated() / (1024**2)) if use_cuda else 0.0
    )
    peak_reserved = (
        float(torch.cuda.max_memory_reserved() / (1024**2)) if use_cuda else 0.0
    )
    return {
        "sec_per_iter": sec_per_iter,
        "fps": fps,
        "peak_alloc_mib": peak_alloc,
        "peak_reserved_mib": peak_reserved,
    }


def _load_cfg(yaml_path: str, set_cfgs: Optional[List[str]]) -> Any:
    # Note: `cfg` is a global object in this codebase.
    # FSHNet yaml configs commonly reference base configs via paths like
    # `tools/cfgs/...`, which are relative to the FSHNet root. To make this
    # script robust when invoked from repo root (or elsewhere), we temporarily
    # chdir to the FSHNet root while parsing.
    fshnet_root = Path(__file__).resolve().parents[1]
    yaml_abs = str(Path(yaml_path).resolve())
    old_cwd = os.getcwd()
    try:
        os.chdir(str(fshnet_root))
        cfg_from_yaml_file(yaml_abs, cfg)
    finally:
        os.chdir(old_cwd)
    if set_cfgs is not None:
        cfg_from_list(set_cfgs, cfg)
    return cfg


def benchmark_one(
    yaml_path: str,
    *,
    ckpt: Optional[str],
    warmup: int,
    iters: int,
    batch_size: int,
    workers: int,
    set_cfgs: Optional[List[str]],
) -> Dict[str, Any]:
    cfg_local = _load_cfg(yaml_path, set_cfgs)
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    # Build val (test) dataloader.
    val_set, val_loader, _ = build_dataloader(
        dataset_cfg=cfg_local.DATA_CONFIG,
        class_names=cfg_local.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=None,
        training=False,
    )

    # Build train dataloader.
    train_set, train_loader, _ = build_dataloader(
        dataset_cfg=cfg_local.DATA_CONFIG,
        class_names=cfg_local.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=None,
        training=True,
    )

    model = build_network(
        model_cfg=cfg_local.MODEL,
        num_class=len(cfg_local.CLASS_NAMES),
        dataset=train_set,
    )

    if ckpt:
        model.load_params_from_file(filename=ckpt, logger=None, to_cpu=False)

    if use_cuda:
        model.cuda()

    # Inference includes post_processing when model.training is False.
    model.eval()

    def _infer_step(batch_dict):
        common_utils.load_data_to_gpu(batch_dict)
        with torch.no_grad():
            model(batch_dict)

    infer_stats = _bench_loop(
        step_fn=_infer_step,
        dataloader=val_loader,
        warmup=warmup,
        iters=iters,
        use_cuda=use_cuda,
    )

    # Train step (FP32): use OpenPCDet-style model_fn_decorator.
    model.train()
    optimizer = build_optimizer(model, cfg_local.OPTIMIZATION)
    model_func = model_fn_decorator()

    def _train_step(batch_dict):
        common_utils.load_data_to_gpu(batch_dict)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model_func(model, batch_dict)
        loss.backward()
        optimizer.step()

    train_stats = _bench_loop(
        step_fn=_train_step,
        dataloader=train_loader,
        warmup=warmup,
        iters=iters,
        use_cuda=use_cuda,
    )

    return {
        "config": yaml_path,
        "checkpoint": ckpt,
        "device": device,
        "batch_size": batch_size,
        "params_m": _params_m(model),
        "infer": infer_stats,
        "train": train_stats,
    }


def main() -> None:
    args = parse_args()
    results: List[Dict[str, Any]] = []
    for yaml_path in args.cfgs:
        results.append(
            benchmark_one(
                yaml_path,
                ckpt=args.ckpt,
                warmup=args.warmup,
                iters=args.iters,
                batch_size=args.batch_size,
                workers=args.workers,
                set_cfgs=args.set_cfgs,
            )
        )

    payload = {
        "meta": {
            "warmup": args.warmup,
            "iters": args.iters,
            "batch_size": args.batch_size,
            "amp": False,
            "includes_postprocess": True,
            "includes_dataloader": True,
        },
        "results": results,
    }
    _mkdir_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
