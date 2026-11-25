"""Self-Attentive Sequential Recommendation (SASRec) Architecture for EUTERPE.

Implements Kang & McAuley (ICDM 2018):
- Multi-Head Self-Attention over chronological item sequences.
- Strict causal masking (position t can only attend to positions <= t; zero future leakage).
- Learned positional embeddings and padding tokens.
- Pointwise Feed-Forward Networks with GELU non-linearities and residual LayerNorms.
- InfoNCE / Sampled Softmax sequential transition loss.
- Fast candidate generation via dynamic sequence representations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Linear(d_model, d_ff)
        self.conv2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d)
        out = self.conv2(self.dropout(self.activation(self.conv1(x))))
        out = self.dropout(out)
        return out


class CausalSelfAttention(nn.Module):
    """Multi-Head Self-Attention with strict causal masking."""

    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, L, d)
        B, L, d = x.size()

        # Project and reshape to (B, n_heads, L, d_head)
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, L, L)

        if mask is not None:
            # mask is (B, 1, L, L) or (1, 1, L, L)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (B, H, L, d_head)
        context = context.transpose(1, 2).contiguous().view(B, L, d)
        out = self.out_proj(context)
        out = self.dropout(out)
        return out


class SASRecBlock(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm and Residuals."""

    def __init__(self, d_model: int, n_heads: int = 2, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = CausalSelfAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = PointwiseFeedForward(d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN architecture
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommender."""

    def __init__(
        self,
        n_items: int,
        max_seq_len: int = 50,
        d_model: int = 64,
        n_heads: int = 2,
        n_blocks: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        # Item Embedding: 0 is reserved for padding; 1..n_items are tracks
        self.item_embedding = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        # Learned Positional Embedding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.emb_dropout = nn.Dropout(dropout)
        self.emb_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.final_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings and linear weights."""
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)

    def _get_causal_mask(self, seqs: torch.Tensor) -> torch.Tensor:
        """Construct causal lower-triangular + key-padding attention mask.
        
        Returns tensor of shape (B, 1, L, L) where 1 indicates valid attention and 0 masked.
        """
        B, L = seqs.size()
        # Causal mask: lower triangular
        causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=seqs.device))  # (L, L)
        causal = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        # Padding mask: positions where item != 0
        pad_mask = (seqs != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)

        mask = causal & pad_mask  # (B, 1, L, L)
        return mask

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """Encode item sequence into contextual hidden states.
        
        Args:
            seqs: Integer tensor of shape (B, L) with track IDs (1-indexed, 0 is pad).
        Returns:
            hidden_states: Tensor of shape (B, L, d_model).
        """
        B, L = seqs.size()
        device = seqs.device

        # Item embeddings
        item_embs = self.item_embedding(seqs)  # (B, L, d)

        # Positional embeddings
        positions = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, L)
        pos_embs = self.pos_embedding(positions)  # (B, L, d)

        x = item_embs + pos_embs
        x = self.emb_layernorm(x)
        x = self.emb_dropout(x)

        # Attention mask
        mask = self._get_causal_mask(seqs)

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        out = self.final_layernorm(x)
        return out

    def encode_user_sequence(self, seqs: torch.Tensor) -> torch.Tensor:
        """Extract the final non-padded representation as the dynamic user query vector.
        
        Args:
            seqs: Integer tensor of shape (B, L) with track IDs (1-indexed).
        Returns:
            user_vector: Normalized representation of shape (B, d_model).
        """
        hidden = self.forward(seqs)  # (B, L, d)
        
        # Take the vector at the last valid position
        # If all 0, defaults to index 0
        lengths = (seqs != 0).sum(dim=1)  # (B,)
        last_indices = torch.clamp(lengths - 1, min=0)  # (B,)
        
        batch_indices = torch.arange(seqs.size(0), device=seqs.device)
        user_repr = hidden[batch_indices, last_indices]  # (B, d)
        
        # Unit-normalize representation for stable inner product retrieval
        user_repr = F.normalize(user_repr, p=2, dim=-1)
        return user_repr

    def get_all_item_embeddings(self) -> torch.Tensor:
        """Return normalized embeddings for all eligible items (1..n_items).
        
        Returns tensor of shape (n_items, d_model).
        """
        # Exclude padding token at index 0
        raw_items = self.item_embedding.weight[1 : self.n_items + 1]  # (n_items, d)
        normed_items = F.normalize(raw_items, p=2, dim=-1)
        return normed_items

    def compute_infonce_loss(
        self,
        seqs: torch.Tensor,
        pos_targets: torch.Tensor,
        neg_targets: torch.Tensor,
        temperature: float = 0.15,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss over all non-padded positions.
        
        Args:
            seqs: (B, L) input track IDs.
            pos_targets: (B, L) actual next track IDs.
            neg_targets: (B, L, K_neg) sampled negative track IDs.
            temperature: Softmax scaling temperature.
        """
        hidden = self.forward(seqs)  # (B, L, d)
        hidden = F.normalize(hidden, p=2, dim=-1)

        pos_embs = self.item_embedding(pos_targets)  # (B, L, d)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)

        neg_embs = self.item_embedding(neg_targets)  # (B, L, K_neg, d)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)

        # Positive logits: (B, L)
        pos_logits = (hidden * pos_embs).sum(dim=-1, keepdim=True) / temperature  # (B, L, 1)

        # Negative logits: (B, L, K_neg)
        neg_logits = torch.matmul(hidden.unsqueeze(2), neg_embs.transpose(-2, -1)).squeeze(2) / temperature

        # Concatenate: (B, L, 1 + K_neg)
        all_logits = torch.cat([pos_logits, neg_logits], dim=-1)

        # Target label is 0 (positive item)
        labels = torch.zeros(all_logits.size(0) * all_logits.size(1), dtype=torch.long, device=seqs.device)

        # Valid mask: position is valid if target != 0
        valid_mask = (pos_targets != 0).view(-1)  # (B*L,)

        flat_logits = all_logits.view(-1, all_logits.size(-1))
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=seqs.device, requires_grad=True)

        loss = F.cross_entropy(flat_logits[valid_mask], labels[valid_mask])
        return loss

    def save(self, path: Union[str, Path]) -> None:
        """Save model weights and architectural hyperparameters."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "n_items": self.n_items,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_blocks": self.n_blocks,
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "SASRec":
        """Load trained SASRec model checkpoint."""
        ckpt = torch.load(path, map_location=device)
        model = cls(
            n_items=ckpt["n_items"],
            max_seq_len=ckpt["max_seq_len"],
            d_model=ckpt["d_model"],
            n_heads=ckpt["n_heads"],
            n_blocks=ckpt["n_blocks"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
