# Copyright (c) OpenMMLab. All rights reserved.
"""End-to-end efficiency benchmark for MMDet3D models.

This script is intentionally lightweight and follows existing repo patterns:
- Build from config via MMEngine Runner.
- Measure end-to-end time including dataloader + H2D + forward + postprocess.

Benchmark contract (for this repo):
- FP32 only (AMP disabled).
- batch_size forced by cfg-options (recommended: 1).
- Use val_dataloader for inference, train_dataloader for train step timing.
"""

import argparse
import json
import os
import os.path as osp
import time
from typing import Any, Dict, List, Optional

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, autocast
from mmengine.optim import OptimWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end efficiency for MMDet3D configs"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="One or more config paths (e.g. configs/RefineMoE/AM.py)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path to load for all configs",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Force train/val/test dataloader batch_size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional override for dataloader num_workers",
    )
    parser.add_argument(
        "--out", default="results/bench/kitti_val_mm3d.json", help="Output JSON path"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default=None,
        help="Extra overrides merged into config (key=value)",
    )
    return parser.parse_args()


def _force_fp32(cfg: Config) -> None:
    # Disable AMP for training.
    if "optim_wrapper" in cfg and isinstance(cfg.optim_wrapper, dict):
        if cfg.optim_wrapper.get("type", None) == "AmpOptimWrapper":
            cfg.optim_wrapper["type"] = "OptimWrapper"
            cfg.optim_wrapper.pop("loss_scale", None)


def _force_dataloader_bs_workers(
    dataloader_cfg: Any, batch_size: int, workers: Optional[int]
) -> None:
    if dataloader_cfg is None:
        return
    if isinstance(dataloader_cfg, dict):
        dataloader_cfg["batch_size"] = batch_size
        if workers is not None:
            dataloader_cfg["num_workers"] = workers


def _bench_loop(
    *,
    step_fn,
    dataloader,
    warmup: int,
    iters: int,
    use_cuda: bool,
) -> Dict[str, float]:
    """Measure end-to-end time including dataloader iteration."""
    assert warmup >= 0 and iters > 0

    data_iter = iter(dataloader)
    # Warmup
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


def _mkdir_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def benchmark_one(
    config_path: str,
    *,
    checkpoint: Optional[str],
    warmup: int,
    iters: int,
    batch_size: int,
    workers: Optional[int],
    extra_cfg_options: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    init_default_scope("mmdet3d")
    cfg = Config.fromfile(config_path)
    cfg.launcher = "none"
    if extra_cfg_options:
        cfg.merge_from_dict(extra_cfg_options)

    # Runner.from_cfg requires `work_dir` to exist in cfg; mirror tools/train.py.
    if cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs",
            "bench_efficiency",
            osp.splitext(osp.basename(config_path))[0],
        )

    # Force FP32.
    _force_fp32(cfg)

    # Force batch_size/num_workers across known dataloaders.
    _force_dataloader_bs_workers(cfg.get("train_dataloader", None), batch_size, workers)
    _force_dataloader_bs_workers(cfg.get("val_dataloader", None), batch_size, workers)
    _force_dataloader_bs_workers(cfg.get("test_dataloader", None), batch_size, workers)

    runner = Runner.from_cfg(cfg)
    model = runner.model

    # Optional checkpoint.
    if checkpoint:
        runner.load_checkpoint(checkpoint)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    # Dataloaders: prefer val_dataloader for inference.
    val_loader = getattr(runner, "val_dataloader", None)
    if val_loader is None:
        val_loader = getattr(runner, "test_dataloader", None)
    train_loader = getattr(runner, "train_dataloader", None)
    optim_wrapper = getattr(runner, "optim_wrapper", None)
    assert val_loader is not None, "val_dataloader/test_dataloader not found in config."
    assert train_loader is not None, "train_dataloader not found in config."

    # In MMEngine, `Runner.from_cfg` stores `optim_wrapper` as a config dict.
    # The real OptimWrapper is built during `runner.train()`. Since we are
    # running a manual train_step loop (benchmarking only), build it here.
    if optim_wrapper is not None and not isinstance(optim_wrapper, OptimWrapper):
        optim_wrapper = runner.build_optim_wrapper(optim_wrapper)
        runner.optim_wrapper = optim_wrapper
    assert isinstance(optim_wrapper, OptimWrapper), "optim_wrapper not built."

    # Inference (includes postprocess via test_step).
    model.eval()
    with torch.no_grad():

        def _infer_step(batch):
            with autocast(enabled=False):
                model.test_step(batch)

        infer_stats = _bench_loop(
            step_fn=_infer_step,
            dataloader=val_loader,
            warmup=warmup,
            iters=iters,
            use_cuda=use_cuda,
        )

    # Train step.
    model.train()

    def _train_step(batch):
        with autocast(enabled=False):
            model.train_step(batch, optim_wrapper)

    train_stats = _bench_loop(
        step_fn=_train_step,
        dataloader=train_loader,
        warmup=warmup,
        iters=iters,
        use_cuda=use_cuda,
    )

    return {
        "config": config_path,
        "checkpoint": checkpoint,
        "device": device,
        "batch_size": batch_size,
        "params_m": _params_m(model),
        "infer": infer_stats,
        "train": train_stats,
    }


def main() -> None:
    args = parse_args()
    extra_cfg_options = args.cfg_options or {}
    results: List[Dict[str, Any]] = []
    for cfg_path in args.configs:
        results.append(
            benchmark_one(
                cfg_path,
                checkpoint=args.checkpoint,
                warmup=args.warmup,
                iters=args.iters,
                batch_size=args.batch_size,
                workers=args.workers,
                extra_cfg_options=extra_cfg_options,
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
