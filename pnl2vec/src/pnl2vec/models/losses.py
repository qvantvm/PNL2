"""Numerically stable negative-sampling losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def neg_sampling_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
) -> torch.Tensor:
    """pos_score: (B,), neg_score: (B, K)."""
    pos = F.logsigmoid(pos_score)
    neg = F.logsigmoid(-neg_score).sum(dim=-1)
    return -(pos + neg).mean()
