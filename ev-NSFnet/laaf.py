"""
Layer-wise Locally Adaptive Activation Function (L-LAAF)

This module provides a simple layer-wise adaptive scaling for tanh:
    y = tanh(a * x)
where 'a' is a learnable positive scalar per activation layer.

Notes
- We keep implementation lightweight and PyTorch-eager friendly (no Triton).
- We expose a small regularization helper via attribute access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class LAAFScalar(nn.Module):
    """Layer-wise LAAF with a single learnable scale per activation layer.

    y = tanh(a * x), where a = abs(a_raw) + eps to ensure positivity.

    Args:
        init_scale: Initial value for a (>0). Default 1.0.
        max_scale: Optional clamp upper bound for a (e.g., 20.0).
        eps: Small epsilon added to abs to keep strictly positive.

    Shape:
        - Input: (N, D)
        - Output: (N, D)
    """

    def __init__(self, init_scale: float = 1.0, max_scale: Optional[float] = 20.0, eps: float = 1e-6):
        super().__init__()
        init_scale = float(init_scale)
        if init_scale <= 0:
            # 防止非法初始化
            init_scale = 1.0
        # 直接使用 init_scale 初始化
        self.a_raw = nn.Parameter(torch.tensor(init_scale).view(1))
        self.max_scale = None if max_scale is None else float(max_scale)
        self.eps = float(eps)

    @property
    def a(self) -> torch.Tensor:
        """Positive scale a = abs(a_raw) + eps, optionally clamped."""
        a_pos = torch.abs(self.a_raw) + self.eps
        if self.max_scale is not None:
            return torch.clamp(a_pos, max=self.max_scale)
        return a_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # layer-wise: broadcast scalar a to input shape
        return torch.tanh(self.a * x)

    def regularization(self, target: float = 1.0) -> torch.Tensor:
        """Simple L2 regularization on (a - target).

        Args:
            target: Desired scale center, default 1.0
        Returns:
            Scalar tensor regularization value.
        """
        return (self.a - float(target)).pow(2).mean()

    def get_stats(self) -> dict:
        """Return current scale value for monitoring."""
        a_val = self.a.detach().item()
        return {"a": a_val}


@dataclass
class LAAFConfig:
    """Configuration holder for LAAF usage in this project."""
    enabled: bool = False
    init_scale: float = 1.0
    max_scale: float = 20.0
    reg_lambda: float = 0.0


def compute_laaf_regularization(model: nn.Module, target: float = 1.0) -> torch.Tensor:
    """Compute summed regularization over all LAAFScalar layers in a model.

    Returns zero if no LAAF layer exists.
    """
    reg = None
    for m in model.modules():
        if isinstance(m, LAAFScalar):
            term = m.regularization(target)
            reg = term if reg is None else (reg + term)
    if reg is None:
        return torch.tensor(0.0, device=next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else 'cpu')
    return reg

