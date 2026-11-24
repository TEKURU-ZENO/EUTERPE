"""Discovery and Recommendation Evaluation Metrics for EUTERPE.

Mathematically validated reference implementations of predictive relevance
and discovery science metrics with zero data leakage.
"""

import math
from typing import Dict, List, Set, Tuple, Union
import numpy as np


def recall_at_k(actual: Set[int], predicted: List[int], k: int = 20) -> float:
    """Calculate Conventional Recall@K.
    
    Formula:
        |predicted[:k] ∩ actual| / |actual|
    This follows standard recommender systems literature (Herlocker et al., Rendle et al.).
    """
    if not actual or k <= 0:
        return 0.0
    
    pred_k = predicted[:k]
    hits = len(set(pred_k) & actual)
    return hits / float(len(actual))


def capped_recall_at_k(actual: Set[int], predicted: List[int], k: int = 20) -> float:
    """Calculate Capped / Normalized Recall@K.
    
    Formula:
        |predicted[:k] ∩ actual| / min(k, |actual|)
    Useful as an auxiliary diagnostic metric when |actual| > k.
    """
    if not actual or k <= 0:
        return 0.0
    
    pred_k = predicted[:k]
    hits = len(set(pred_k) & actual)
    denominator = min(k, len(actual))
    return hits / float(denominator) if denominator > 0 else 0.0


def precision_at_k(actual: Set[int], predicted: List[int], k: int = 20) -> float:
    """Calculate Precision@K.
    
    Formula:
        |predicted[:k] ∩ actual| / k
    """
    if k <= 0:
        return 0.0
    pred_k = predicted[:k]
    hits = len(set(pred_k) & actual)
    return hits / float(k)


def ndcg_at_k(actual: Set[int], predicted: List[int], k: int = 20) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    
    Formula:
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
        IDCG@K = sum_{i=1}^{min(K, |actual|)} 1 / log2(i + 1)
        NDCG@K = DCG@K / IDCG@K
    """
    if not actual or k <= 0:
        return 0.0
    
    pred_k = predicted[:k]
    dcg = 0.0
    for idx, item in enumerate(pred_k):
        if item in actual:
            dcg += 1.0 / math.log2(idx + 2)  # idx + 2 corresponds to rank 1 -> log2(2) = 1
            
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(actual))))
    return (dcg / idcg) if idcg > 0.0 else 0.0


def mrr_at_k(actual: Set[int], predicted: List[int], k: int = 20) -> float:
    """Calculate Mean Reciprocal Rank (MRR@K).
    
    Formula:
        1 / rank_first_hit if any hit in predicted[:k] else 0.0
    """
    if not actual or k <= 0:
        return 0.0
    
    for rank, item in enumerate(predicted[:k], start=1):
        if item in actual:
            return 1.0 / rank
    return 0.0


def novelty_at_k(
    predicted: List[int],
    item_probabilities: Dict[int, float],
    k: int = 20
) -> float:
    """Calculate Novelty@K (Average Self-Information / Surprisal).
    
    Formula:
        - (1 / k) * sum_{i in predicted[:k]} log2(P(i))
    where P(i) is the global marginal probability of track i in training.
    """
    if not predicted or k <= 0:
        return 0.0
    
    pred_k = predicted[:k]
    total_self_info = 0.0
    count = 0
    
    for item in pred_k:
        prob = item_probabilities.get(item, 1e-7)
        prob = max(prob, 1e-9)  # guard against log(0)
        total_self_info += -math.log2(prob)
        count += 1
        
    return total_self_info / float(count) if count > 0 else 0.0


def intra_list_diversity(
    predicted: List[int],
    item_embeddings: Union[Dict[int, np.ndarray], np.ndarray],
    k: int = 20
) -> float:
    """Calculate Intra-List Diversity (ILD) using pairwise cosine distance.
    
    Formula:
        2 / (K * (K - 1)) * sum_{i < j} (1 - cos_sim(v_i, v_j))
    """
    if not predicted or k <= 1:
        return 0.0
    
    if isinstance(item_embeddings, np.ndarray):
        pred_k = [it for it in predicted[:k] if 0 <= it < item_embeddings.shape[0]]
        if len(pred_k) <= 1:
            return 0.0
        vectors = item_embeddings[pred_k].astype(np.float32)
    else:
        pred_k = [it for it in predicted[:k] if it in item_embeddings]
        if len(pred_k) <= 1:
            return 0.0
        vectors = np.array([item_embeddings[it] for it in pred_k], dtype=np.float32)
    
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed_vectors = vectors / norms
    
    # Pairwise cosine similarity matrix
    sim_matrix = np.dot(normed_vectors, normed_vectors.T)
    dist_matrix = 1.0 - sim_matrix
    
    n = len(pred_k)
    upper_tri_indices = np.triu_indices(n, k=1)
    total_dist = np.sum(dist_matrix[upper_tri_indices])
    pair_count = n * (n - 1) / 2.0
    
    return float(total_dist / pair_count) if pair_count > 0 else 0.0


def catalog_coverage(all_predicted: List[List[int]], total_catalog_items: int) -> float:
    """Calculate Catalog Coverage.
    
    Formula:
        |U_{u} predicted_u| / |Catalog|
    """
    if total_catalog_items <= 0:
        return 0.0
    
    unique_rec_items = set()
    for recs in all_predicted:
        unique_rec_items.update(recs)
        
    return len(unique_rec_items) / float(total_catalog_items)


def artist_coverage(
    all_predicted: List[List[int]],
    item_to_artist: Dict[int, int],
    total_catalog_artists: int
) -> float:
    """Calculate Artist Coverage.
    
    Formula:
        |U_{u} Artists(predicted_u)| / |Catalog Artists|
    """
    if total_catalog_artists <= 0:
        return 0.0
    
    unique_artists = set()
    for recs in all_predicted:
        for item in recs:
            if item in item_to_artist:
                unique_artists.add(item_to_artist[item])
                
    return len(unique_artists) / float(total_catalog_artists)


def gini_coefficient(recommendation_counts: np.ndarray) -> float:
    """Calculate the Gini Coefficient of recommendation distribution.
    
    0.0 = completely equal distribution (every track recommended equally)
    1.0 = maximal inequality (one track receives all recommendations)
    
    Formula:
        G = sum_{i=1}^n (2i - n - 1) * x_i / (n * sum x_i)
    where x is sorted in ascending order.
    """
    counts = np.asarray(recommendation_counts, dtype=np.float64)
    if counts.size == 0 or np.sum(counts) == 0:
        return 0.0
    
    # Values must be non-negative
    counts = np.sort(counts)
    n = counts.shape[0]
    index = np.arange(1, n + 1)
    return float(((np.sum((2 * index - n - 1) * counts)) / (n * np.sum(counts))))


def serendipity_at_k(
    actual: Set[int],
    predicted: List[int],
    user_history: Set[int],
    item_embeddings: Dict[int, np.ndarray],
    k: int = 20
) -> float:
    """Calculate Serendipity@K.
    
    Surfaces recommendations that are both relevant (in actual test set) and
    unexpected (minimum cosine distance from items in user's historical profile).
    
    Formula:
        (1 / k) * sum_{i in predicted[:k]} I(i in actual) * min_{j in history} (1 - cos(v_i, v_j))
    """
    if not actual or not user_history or k <= 0:
        return 0.0
    
    pred_k = predicted[:k]
    history_items = [it for it in user_history if it in item_embeddings]
    if not history_items:
        return 0.0
        
    hist_vecs = np.array([item_embeddings[it] for it in history_items], dtype=np.float32)
    hist_norms = np.linalg.norm(hist_vecs, axis=1, keepdims=True)
    hist_norms[hist_norms == 0] = 1.0
    normed_hist = hist_vecs / hist_norms
    
    total_serendipity = 0.0
    for item in pred_k:
        if item in actual and item in item_embeddings:
            v_i = item_embeddings[item].astype(np.float32)
            norm_v = np.linalg.norm(v_i)
            if norm_v == 0:
                continue
            normed_v = v_i / norm_v
            
            # Cosine similarity to all history items
            sims = np.dot(normed_hist, normed_v)
            # Minimum distance = 1 - max similarity
            unexpectedness = 1.0 - float(np.max(sims))
            total_serendipity += max(0.0, unexpectedness)
            
    return total_serendipity / float(k)


def paired_bootstrap_ci(
    metric_values: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Compute mean and 95% Bootstrap Confidence Interval.
    
    Returns:
        (mean, ci_lower, ci_upper)
    """
    if len(metric_values) == 0:
        return 0.0, 0.0, 0.0
        
    rng = np.random.default_rng(seed)
    n = len(metric_values)
    boot_means = np.empty(n_bootstraps)
    
    for b in range(n_bootstraps):
        sample = rng.choice(metric_values, size=n, replace=True)
        boot_means[b] = np.mean(sample)
        
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    mean = float(np.mean(metric_values))
    return mean, lower, upper
