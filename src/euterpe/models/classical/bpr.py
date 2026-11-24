"""Bayesian Personalized Ranking (BPR) recommendation model for EUTERPE.

Collaborative Model 2:
Pairwise preference learning from implicit feedback optimizing the AUC metric
following Rendle, Freudenthaler, Gantner & Schmidt-Thieme (2009).
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from scipy import sparse
import implicit.cpu.bpr as implicit_bpr


class BPRRecommender:
    """Bayesian Personalized Ranking (BPR) pairwise matrix factorization model."""

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.005,
        iterations: int = 50,
        random_state: int = 42,
        verify_negative_samples: bool = True
    ):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.verify_negative_samples = verify_negative_samples
        
        self.model: Optional[implicit_bpr.BayesianPersonalizedRanking] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(
        self,
        user_items: sparse.csr_matrix,
        show_progress: bool = True
    ) -> "BPRRecommender":
        """Fit BPR model on user_items CSR matrix.
        
        Args:
            user_items: CSR matrix of shape (n_users, n_items), non-zero entries denote
                        positive consumed interactions.
        """
        self.model = implicit_bpr.BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            verify_negative_samples=self.verify_negative_samples
        )
        
        # BPR expects binary or non-zero interaction matrix
        binary_mat = user_items.copy().astype(np.float32)
        binary_mat.data = np.ones_like(binary_mat.data)
        
        self.model.fit(binary_mat, show_progress=show_progress)
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
        
        for start_idx in range(0, n_users, batch_size):
            end_idx = min(start_idx + batch_size, n_users)
            batch_users = self.user_factors[start_idx:end_idx]
            
            # Scores = batch_users @ item_factors.T
            batch_scores = np.dot(batch_users, self.item_factors.T)
            
            for i, u in enumerate(range(start_idx, end_idx)):
                consumed_items = train_matrix.indices[train_matrix.indptr[u]:train_matrix.indptr[u+1]]
                batch_scores[i, consumed_items] = -np.inf
                
                if k < batch_scores.shape[1]:
                    top_k_unsorted = np.argpartition(-batch_scores[i], kth=k)[:k]
                    top_k_sorted = top_k_unsorted[np.argsort(-batch_scores[i, top_k_unsorted])]
                else:
                    top_k_sorted = np.argsort(-batch_scores[i])[:k]
                    
                results[u] = [int(item_id) for item_id in top_k_sorted]
                
        return results

    def save(self, filepath: Path) -> None:
        """Save BPR model factors and hyperparameters to npz file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            filepath,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations
        )

    def load(self, filepath: Path) -> "BPRRecommender":
        """Load BPR model factors from npz file."""
        data = np.load(filepath)
        self.user_factors = data["user_factors"]
        self.item_factors = data["item_factors"]
        self.factors = int(data["factors"])
        self.learning_rate = float(data["learning_rate"])
        self.regularization = float(data["regularization"])
        self.iterations = int(data["iterations"])
        return self
