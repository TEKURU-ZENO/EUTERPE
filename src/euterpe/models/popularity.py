"""Popularity-based recommendation models for EUTERPE.

Model 2.1: Global Popularity (frequency-based non-personalized baseline).
Model 2.2: Recency-Decayed Popularity (exponential half-life time decay).
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
from scipy import sparse


class GlobalPopularityRecommender:
    """Non-personalized baseline recommending the most popular tracks in training data."""

    def __init__(self, mode: str = "play_count"):
        """Args:
            mode: 'play_count' (sum of scrobbles) or 'binary' (unique listener count).
        """
        self.mode = mode
        self.global_scores: np.ndarray = None
        self.ranked_item_indices: np.ndarray = None
        self.n_items: int = 0

    def fit(self, train_matrix: sparse.csr_matrix) -> "GlobalPopularityRecommender":
        """Compute aggregate item popularity scores across all users.
        
        Args:
            train_matrix: Shape (n_users, n_items). If mode='play_count', contains
                          play counts. If mode='binary', treats non-zero as 1.
        """
        self.n_items = train_matrix.shape[1]
        if self.mode == "binary":
            binary_mat = train_matrix.copy()
            binary_mat.data = np.ones_like(binary_mat.data)
            # Sum columns
            self.global_scores = np.asarray(binary_mat.sum(axis=0)).flatten()
        else:
            self.global_scores = np.asarray(train_matrix.sum(axis=0)).flatten()

        # Pre-sort items in descending order of popularity
        self.ranked_item_indices = np.argsort(-self.global_scores)
        return self

    def recommend(
        self,
        user_idx: int,
        user_consumed_items: Set[int],
        k: int = 20
    ) -> List[Tuple[int, float]]:
        """Generate top-K recommendations for a single user with consumed item masking."""
        recs = []
        for item_idx in self.ranked_item_indices:
            if item_idx not in user_consumed_items:
                recs.append((int(item_idx), float(self.global_scores[item_idx])))
                if len(recs) == k:
                    break
        return recs

    def recommend_all(
        self,
        train_matrix: sparse.csr_matrix,
        k: int = 20
    ) -> Dict[int, List[int]]:
        """Generate top-K recommendations for all users in the training matrix."""
        n_users = train_matrix.shape[0]
        results = {}
        for u in range(n_users):
            consumed = set(train_matrix.indices[train_matrix.indptr[u]:train_matrix.indptr[u+1]])
            user_recs = self.recommend(u, consumed, k=k)
            results[u] = [item_id for item_id, _ in user_recs]
        return results


class RecencyDecayedPopularityRecommender:
    """Time-sensitive popularity baseline applying exponential half-life decay.
    
    Score(i) = sum_{e in E_i} 2^{-(T_train - t_e) / half_life}
    """

    def __init__(self, half_life_days: float = 90.0):
        self.half_life_days = half_life_days
        self.half_life_seconds = half_life_days * 86400.0
        self.global_scores: np.ndarray = None
        self.ranked_item_indices: np.ndarray = None
        self.n_items: int = 0

    def fit_from_dataframe(
        self,
        df_train: pl.DataFrame,
        item_to_idx: Dict[str, int],
        t_cutoff_epoch: int
    ) -> "RecencyDecayedPopularityRecommender":
        """Compute recency-decayed scores from the raw training event log.
        
        Args:
            df_train: Training interactions dataframe with 'recording_key' and 'epoch_seconds'.
            item_to_idx: Dictionary mapping recording_key to contiguous integer index.
            t_cutoff_epoch: Latest epoch timestamp in training set (T_train).
        """
        self.n_items = len(item_to_idx)
        self.global_scores = np.zeros(self.n_items, dtype=np.float64)

        # Compute decay weight per event: 2 ** ( - (T - t) / tau )
        # Using polars vectorization
        df_decay = df_train.with_columns([
            (2.0 ** (- (t_cutoff_epoch - pl.col("epoch_seconds")) / self.half_life_seconds)).alias("decay_weight")
        ]).group_by("recording_key").agg([
            pl.col("decay_weight").sum().alias("total_recency_score")
        ])

        for row in df_decay.iter_rows(named=True):
            rk = row["recording_key"]
            if rk in item_to_idx:
                idx = item_to_idx[rk]
                self.global_scores[idx] = row["total_recency_score"]

        self.ranked_item_indices = np.argsort(-self.global_scores)
        return self

    def recommend(
        self,
        user_idx: int,
        user_consumed_items: Set[int],
        k: int = 20
    ) -> List[Tuple[int, float]]:
        """Generate top-K recommendations for a single user with consumed item masking."""
        recs = []
        for item_idx in self.ranked_item_indices:
            if item_idx not in user_consumed_items:
                recs.append((int(item_idx), float(self.global_scores[item_idx])))
                if len(recs) == k:
                    break
        return recs

    def recommend_all(
        self,
        train_matrix: sparse.csr_matrix,
        k: int = 20
    ) -> Dict[int, List[int]]:
        """Generate top-K recommendations for all users."""
        n_users = train_matrix.shape[0]
        results = {}
        for u in range(n_users):
            consumed = set(train_matrix.indices[train_matrix.indptr[u]:train_matrix.indptr[u+1]])
            user_recs = self.recommend(u, consumed, k=k)
            results[u] = [item_id for item_id, _ in user_recs]
        return results
