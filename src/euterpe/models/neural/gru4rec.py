"""Gated Recurrent Unit for Session-based Recommendation (GRU4Rec) Architecture for EUTERPE.

Implements Hidasi et al. (ICLR 2016):
- Gated recurrent units processing chronological listening sequences.
- Recurrent hidden state update capturing temporal transition dynamics.
- Bounded to chronological sequence order without future peeking.
- InfoNCE / Sampled Softmax loss over consecutive track transitions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU4Rec(nn.Module):
    """Gated Recurrent Unit Sequential Recommender."""

    def __init__(
        self,
        n_items: int,
        max_seq_len: int = 50,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers

        # Item Embedding (0 is padding)
        self.item_embedding = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        # Multi-layer GRU
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """Encode item sequence with GRU.
        
        Args:
            seqs: Integer tensor of shape (B, L) with track IDs (1-indexed).
        Returns:
            hidden: Tensor of shape (B, L, d_model).
        """
        embs = self.item_embedding(seqs)  # (B, L, d)
        embs = self.emb_dropout(embs)
        
        out, _ = self.gru(embs)  # (B, L, d)
        out = self.layernorm(out)
        return out

    def encode_user_sequence(self, seqs: torch.Tensor) -> torch.Tensor:
        """Extract the final non-padded representation as the dynamic user query vector."""
        hidden = self.forward(seqs)  # (B, L, d)
        lengths = (seqs != 0).sum(dim=1)  # (B,)
        last_indices = torch.clamp(lengths - 1, min=0)
        batch_indices = torch.arange(seqs.size(0), device=seqs.device)
        user_repr = hidden[batch_indices, last_indices]
        return F.normalize(user_repr, p=2, dim=-1)

    def get_all_item_embeddings(self) -> torch.Tensor:
        """Return normalized item embeddings for all eligible tracks (1..n_items)."""
        raw = self.item_embedding.weight[1 : self.n_items + 1]
        return F.normalize(raw, p=2, dim=-1)

    def compute_infonce_loss(
        self,
        seqs: torch.Tensor,
        pos_targets: torch.Tensor,
        neg_targets: torch.Tensor,
        temperature: float = 0.15,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss over consecutive track transitions."""
        hidden = self.forward(seqs)  # (B, L, d)
        hidden = F.normalize(hidden, p=2, dim=-1)

        pos_embs = self.item_embedding(pos_targets)  # (B, L, d)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)

        neg_embs = self.item_embedding(neg_targets)  # (B, L, K_neg, d)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)

        pos_logits = (hidden * pos_embs).sum(dim=-1, keepdim=True) / temperature
        neg_logits = torch.matmul(hidden.unsqueeze(2), neg_embs.transpose(-2, -1)).squeeze(2) / temperature

        all_logits = torch.cat([pos_logits, neg_logits], dim=-1)
        labels = torch.zeros(all_logits.size(0) * all_logits.size(1), dtype=torch.long, device=seqs.device)
        valid_mask = (pos_targets != 0).view(-1)

        flat_logits = all_logits.view(-1, all_logits.size(-1))
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=seqs.device, requires_grad=True)

        return F.cross_entropy(flat_logits[valid_mask], labels[valid_mask])

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "n_items": self.n_items,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "GRU4Rec":
        ckpt = torch.load(path, map_location=device)
        model = cls(
            n_items=ckpt["n_items"],
            max_seq_len=ckpt["max_seq_len"],
            d_model=ckpt["d_model"],
            n_layers=ckpt["n_layers"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
