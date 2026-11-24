"""Implicit Alternating Least Squares (ALS) recommendation model for EUTERPE.

Collaborative Model 1:
Factorizes confidence-weighted implicit feedback interaction matrices following
Hu, Koren & Volinsky (2008) and Takács et al. (Conjugate Gradient optimization).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse
import implicit.cpu.als as implicit_als


class ImplicitALSRecommender:
    """Implicit Alternating Least Squares (ALS) collaborative filtering model."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 40.0,
        iterations: int = 25,
        random_state: int = 42,
        use_cg: bool = True
    ):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state
        self.use_cg = use_cg
        
        self.model: Optional[implicit_als.AlternatingLeastSquares] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(
        self,
        user_items: sparse.csr_matrix,
        show_progress: bool = True
    ) -> "ImplicitALSRecommender":
        """Fit ALS model on the user_items CSR matrix.
        
        Args:
            user_items: CSR matrix of shape (n_users, n_items).
        """
        # Initialize model
        self.model = implicit_als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=self.random_state,
            use_cg=self.use_cg
        )
        
        # Fit model: user_items should be CSR float32
        self.model.fit(user_items.astype(np.float32), show_progress=show_progress)
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors
        return self

    def recommend_all(
        self,
        train_matrix: sparse.csr_matrix,
        k: int = 20,
        batch_size: int = 128
    ) -> Dict[int, List[int]]:
        """Generate top-K recommendations for all users with consumed item masking."""
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model must be fit before calling recommend_all.")
            
        n_users = train_matrix.shape[0]
        results = {}
        
        # Vectorized batch scoring for speed and exact masking
        for start_idx in range(0, n_users, batch_size):
            end_idx = min(start_idx + batch_size, n_users)
            batch_users = self.user_factors[start_idx:end_idx]  # (B, d)
            
            # Scores = batch_users @ item_factors.T -> (B, n_items)
            batch_scores = np.dot(batch_users, self.item_factors.T)
            
            # Mask consumed items from train_matrix
            for i, u in enumerate(range(start_idx, end_idx)):
                consumed_items = train_matrix.indices[train_matrix.indptr[u]:train_matrix.indptr[u+1]]
                batch_scores[i, consumed_items] = -np.inf
                
                # Top K items via argpartition
                if k < batch_scores.shape[1]:
                    top_k_unsorted = np.argpartition(-batch_scores[i], kth=k)[:k]
                    top_k_sorted = top_k_unsorted[np.argsort(-batch_scores[i, top_k_unsorted])]
                else:
                    top_k_sorted = np.argsort(-batch_scores[i])[:k]
                    
                results[u] = [int(item_id) for item_id in top_k_sorted]
                
        return results

    def save(self, filepath: Path) -> None:
        """Save model factors and hyperparameters to npz file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            filepath,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations
        )

    def load(self, filepath: Path) -> "ImplicitALSRecommender":
        """Load model factors from npz file."""
        data = np.load(filepath)
        self.user_factors = data["user_factors"]
        self.item_factors = data["item_factors"]
        self.factors = int(data["factors"])
        self.regularization = float(data["regularization"])
        self.alpha = float(data["alpha"])
        self.iterations = int(data["iterations"])
        return self
