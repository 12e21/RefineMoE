from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


def _gaussian_fixed_interval_scoring(
    x: torch.Tensor,
    k: int,
    *,
    global_min: float,
    global_max: float,
    sigma_scale: float,
) -> torch.Tensor:
    """Fixed-interval Gaussian scoring.

    Returns scores of shape (k, n) for input x of shape (n,).
    This is copied (with minor cleanup) from the existing SM implementation to
    keep gating mechanisms self-contained.
    """
    x = x.view(1, -1)
    n = x.shape[1]
    device = x.device

    if k == 1:
        return torch.ones((1, n), device=device)
    if k < 1:
        raise ValueError(f"k must be >= 1, but got {k}.")

    width = (global_max - global_min) / (k - 1)
    sigma = width * sigma_scale

    centers = torch.linspace(
        global_min + width / 2,
        global_max - width / 2,
        k - 1,
        device=device,
        dtype=x.dtype,
    )
    centers = torch.cat(
        [
            centers,
            torch.tensor([global_max + width / 2], device=device, dtype=x.dtype),
        ]
    )

    a = torch.linspace(
        global_min,
        global_max - width,
        k - 1,
        device=device,
        dtype=x.dtype,
    )
    b = a + width
    a = torch.cat([a, torch.tensor([global_max], device=device, dtype=x.dtype)])
    b = torch.cat([b, torch.tensor([float("inf")], device=device, dtype=x.dtype)])

    x_expanded = x.expand(k, -1)
    centers_expanded = centers.view(-1, 1)

    scores = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * sigma**2))
    mask_in_range = (x_expanded >= a.view(-1, 1)) & (x_expanded <= b.view(-1, 1))
    scores[mask_in_range] = 1.0
    return scores


@MODELS.register_module()
class TemperatureSoftmaxGating(BaseModule):
    """Temperature-based softmax gating on sparsity (points_count).

    The gating is computed from fixed-interval Gaussian scores, then converted
    to a temperature-controlled distribution:

      w = softmax(log(score + eps) / tau)

    We return unnormalized positive scores (k, n) that reproduce the desired
    weights under downstream normalization (score / sum(score)).
    """

    def __init__(
        self,
        *,
        num_branch: int,
        global_min: float = 0.0,
        global_max: float = 600.0,
        sigma_scale: float = 0.5,
        temperature: float = 0.5,
        eps: float = 1e-6,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        self.num_branch = num_branch
        self.global_min = global_min
        self.global_max = global_max
        self.sigma_scale = sigma_scale
        self.temperature = temperature
        self.eps = eps

    def forward(self, points_count_list: List[torch.Tensor]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        k = self.num_branch
        for pc in points_count_list:
            raw = _gaussian_fixed_interval_scoring(
                pc.to(dtype=torch.float32),
                k,
                global_min=self.global_min,
                global_max=self.global_max,
                sigma_scale=self.sigma_scale,
            ).to(dtype=torch.float32)

            # score' = exp(log(raw+eps)/tau) = (raw+eps)^(1/tau)
            scores = torch.exp(torch.log(raw + self.eps) / self.temperature)
            out.append(scores)
        return out


def _build_mlp(
    in_dim: int, hidden: int, out_dim: int, num_layers: int
) -> torch.nn.Module:
    if num_layers <= 1:
        return torch.nn.Linear(in_dim, out_dim)
    layers: List[torch.nn.Module] = [
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(inplace=True),
    ]
    for _ in range(num_layers - 2):
        layers.extend([torch.nn.Linear(hidden, hidden), torch.nn.ReLU(inplace=True)])
    layers.append(torch.nn.Linear(hidden, out_dim))
    return torch.nn.Sequential(*layers)


@MODELS.register_module()
class LearnedRouterGating(BaseModule):
    """Learned routing from sparsity (points_count) to branch weights.

    Minimal router: MLP over log(points_count + 1).
    """

    def __init__(
        self,
        *,
        num_branch: int,
        hidden_channels: int = 16,
        num_layers: int = 2,
        temperature: float = 1.0,
        scale: Optional[float] = None,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        self.num_branch = num_branch
        self.temperature = temperature
        # Scale scores to keep average weight ~1 when used as loss weights.
        self.scale = float(scale) if scale is not None else float(num_branch)

        self.router = _build_mlp(
            in_dim=1,
            hidden=hidden_channels,
            out_dim=num_branch,
            num_layers=num_layers,
        )

    def forward(self, points_count_list: List[torch.Tensor]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        k = self.num_branch
        for pc in points_count_list:
            s = torch.log(pc.to(dtype=torch.float32) + 1.0).view(-1, 1)
            logits = self.router(s) / self.temperature  # (n, k)
            w = torch.softmax(logits, dim=-1).transpose(0, 1)  # (k, n)
            # Keep routing weights floating to preserve gradients.
            out.append(w * self.scale)
        return out
