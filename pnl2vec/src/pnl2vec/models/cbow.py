"""CBOW with negative sampling."""

from __future__ import annotations

import torch
import torch.nn as nn

from .losses import neg_sampling_loss


class CBOWNS(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, *, sparse: bool = False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        nn.init.uniform_(self.input_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(
        self,
        context_ids: torch.Tensor,
        centers: torch.Tensor,
        negatives: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """context_ids: (B, C), centers: (B,), negatives: (B, K)."""
        ctx = self.input_embeddings(context_ids)  # (B, C, D)
        if context_mask is not None:
            mask = context_mask.unsqueeze(-1).float()
            ctx = ctx * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            v = ctx.sum(dim=1) / denom
        else:
            v = ctx.mean(dim=1)
        u_pos = self.output_embeddings(centers)
        u_neg = self.output_embeddings(negatives)
        pos_score = torch.sum(v * u_pos, dim=-1)
        neg_score = torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)
        return neg_sampling_loss(pos_score, neg_score)
