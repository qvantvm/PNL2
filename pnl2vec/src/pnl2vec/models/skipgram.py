"""Skip-gram with negative sampling."""

from __future__ import annotations

import torch
import torch.nn as nn

from .losses import neg_sampling_loss


class SkipGramNS(nn.Module):
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
        centers: torch.Tensor,
        contexts: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """centers/contexts: (B,), negatives: (B, K)."""
        v = self.input_embeddings(centers)  # (B, D)
        u_pos = self.output_embeddings(contexts)  # (B, D)
        u_neg = self.output_embeddings(negatives)  # (B, K, D)
        pos_score = torch.sum(v * u_pos, dim=-1)
        neg_score = torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)
        return neg_sampling_loss(pos_score, neg_score)

    def scores(self, centers: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        v = self.input_embeddings(centers)
        u = self.output_embeddings(contexts)
        return torch.sum(v * u, dim=-1)
