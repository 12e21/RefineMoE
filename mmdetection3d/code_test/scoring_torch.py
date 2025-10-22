import torch
import numpy as np

def gaussian_fixed_interval_scoring(x: torch.Tensor, 
                                     K: int, 
                                     global_min: float = 0.0, 
                                     global_max: float = 600.0, 
                                     sigma_scale: float = 0.5) -> torch.Tensor:
    """
    Generate a K-dimensional score vector for each scalar input using fixed interval Gaussian scoring.

    The value range [global_min, global_max] is divided evenly into (K - 1) intervals.
    Each of the first K - 1 groups focuses on a specific subrange, with samples in the interval scored as 1,
    and others decaying based on a Gaussian kernel. The K-th group focuses on values greater than global_max.

    Args:
        x (torch.Tensor): A 1D tensor of shape (N,) containing input scalar values.
        K (int): The number of scoring groups. The last group targets (global_max, +inf).
        global_min (float): The lower bound of the fixed scoring range.
        global_max (float): The upper bound of the fixed scoring range.
        sigma_scale (float): Scaling factor for Gaussian kernel width (sigma = sigma_scale * interval width).

    Returns:
        torch.Tensor: A (K, N) tensor where each row contains the scores for one group.
    """
    x = x.view(1, -1)  # shape: (1, N)
    N = x.shape[1]
    device = x.device

    width = (global_max - global_min) / (K - 1)
    sigma = width * sigma_scale

    centers = torch.linspace(global_min + width / 2, global_max - width / 2, K - 1, device=device)
    centers = torch.cat([centers, torch.tensor([global_max + width / 2], device=device)])  # shape: (K,)

    a = torch.linspace(global_min, global_max - width, K - 1, device=device)
    b = a + width
    a = torch.cat([a, torch.tensor([global_max], device=device)])
    b = torch.cat([b, torch.tensor([float('inf')], device=device)])  # shape: (K,)

    x_expanded = x.expand(K, -1)  # shape: (K, N)
    centers_expanded = centers.view(-1, 1)  # shape: (K, 1)

    # Gaussian kernel scoring
    scores = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * sigma ** 2))

    # Assign score 1.0 to samples within the focused interval
    mask_in_range = (x_expanded >= a.view(-1, 1)) & (x_expanded <= b.view(-1, 1))
    scores[mask_in_range] = 1.0

    return scores  # shape: (K, N)

x = torch.tensor([0, 0, 75, 150, 240, 310, 420, 500, 580, 640, 2000], dtype=torch.float32)
scores = gaussian_fixed_interval_scoring(x, K=2, global_min=0, global_max=100, sigma_scale=4)
print(np.round(scores.numpy(), decimals=2))
