"""Negative sampling and in-batch sampled softmax loss for EUTERPE Two-Tower model.

Implements:
1. In-batch sampled softmax loss with popularity log-Q correction (Yi et al., 2019).
2. False-negative masking: identifies and masks instances where a candidate in the batch
   was previously consumed by the user in the training set (preventing unfair negative penalties).
3. Mixed negative sampling: combines in-batch negatives with uniform catalog negatives.
"""

from typing import Optional, Set
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_popularity_log_q(
    item_frequencies: np.ndarray,
    power: float = 0.75,
) -> np.ndarray:
    """Compute log-Q sampling probabilities for popularity correction (Yi et al., 2019).
    
    Args:
        item_frequencies: Array of shape (n_items,) with track scrobble/play counts.
        power: Exponent scaling factor (default 0.75 following Word2Vec/Mikolov).
    Returns:
        log_q: Array of shape (n_items,) with log sampling probabilities.
    """
    counts = np.maximum(item_frequencies, 1.0) ** power
    total = np.sum(counts)
    q = counts / total
    return np.log(np.maximum(q, 1e-12))


class InBatchSampledSoftmaxLoss(nn.Module):
    """In-batch sampled softmax loss with log-Q correction and false-negative masking."""

    def __init__(
        self,
        temperature: float = 0.05,
        use_popularity_correction: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.use_popularity_correction = use_popularity_correction
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        user_repr: torch.Tensor,
        pos_item_repr: torch.Tensor,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        log_q_table: Optional[torch.Tensor] = None,
        extra_neg_repr: Optional[torch.Tensor] = None,
        extra_neg_ids: Optional[torch.Tensor] = None,
        train_csr_matrix: Optional[sparse.csr_matrix] = None,
        user_consumed_sets: Optional[dict] = None,
    ) -> torch.Tensor:
        """Compute sampled softmax loss.
        
        Args:
            user_repr: Tensor of shape (B, d), normalized user embeddings.
            pos_item_repr: Tensor of shape (B, d), normalized positive item embeddings.
            user_ids: Tensor of shape (B,), user integer indices.
            pos_item_ids: Tensor of shape (B,), positive track integer indices.
            log_q_table: Optional 1D tensor of precomputed log_q values for all items.
            extra_neg_repr: Optional tensor of shape (M, d) of negative candidate items.
            extra_neg_ids: Optional tensor of shape (M,) of extra negative item IDs.
            train_csr_matrix: Optional CSR matrix of shape (n_users, n_items) for fast vectorized false-negative masking.
            user_consumed_sets: Optional dict mapping user_idx -> set of consumed track IDs (for testing/small batches).
        Returns:
            loss: Scalar tensor representing the batch cross-entropy loss.
        """
        batch_size = user_repr.size(0)

        # Build candidate item pool: [pos_items (B), extra_neg_items (M)]
        if extra_neg_repr is not None and extra_neg_ids is not None:
            cand_repr = torch.cat([pos_item_repr, extra_neg_repr], dim=0)  # (B + M, d)
            cand_ids = torch.cat([pos_item_ids, extra_neg_ids], dim=0)    # (B + M,)
        else:
            cand_repr = pos_item_repr  # (B, d)
            cand_ids = pos_item_ids    # (B,)

        # Compute dot product similarity matrix: (B, num_candidates)
        # Scaled by temperature: S_{u, i} = (z_u^T z_i) / tau
        logits = torch.matmul(user_repr, cand_repr.t()) / self.temperature

        # Apply Log-Q Popularity Correction (Yi et al., 2019)
        # S_{u, i} <- S_{u, i} - log Q(i)
        if self.use_popularity_correction and log_q_table is not None:
            log_q_cands = log_q_table[cand_ids].unsqueeze(0)  # (1, num_candidates)
            logits = logits - log_q_cands

        # False-negative masking:
        # If user u has consumed candidate item j in training history and it is NOT the target (u != j),
        # mask it out with -1e9 so the model is not penalized for scoring known positive tracks highly.
        if train_csr_matrix is not None:
            user_ids_cpu = user_ids.detach().cpu().numpy()
            cand_ids_cpu = cand_ids.detach().cpu().numpy()
            submat = train_csr_matrix[user_ids_cpu, :][:, cand_ids_cpu].toarray() > 0
            np.fill_diagonal(submat[:, :batch_size], False)
            if np.any(submat):
                mask = torch.from_numpy(submat).to(logits.device)
                logits = logits.masked_fill(mask, -1e9)
        elif user_consumed_sets is not None:
            user_ids_cpu = user_ids.detach().cpu().numpy()
            cand_ids_cpu = cand_ids.detach().cpu().numpy()
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for b in range(batch_size):
                u_id = user_ids_cpu[b]
                if u_id in user_consumed_sets:
                    u_set = user_consumed_sets[u_id]
                    for j in range(cand_ids.size(0)):
                        if b != j and cand_ids_cpu[j] in u_set:
                            mask[b, j] = True
            logits = logits.masked_fill(mask, -1e9)

        # Ground truth targets: the diagonal index (b == b)
        targets = torch.arange(batch_size, device=user_repr.device)
        loss = self.ce_loss(logits, targets)
        return loss
