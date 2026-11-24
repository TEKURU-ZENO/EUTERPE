"""Discovery Reranking, Multi-Source Candidate Fusion, and Pareto Optimization for EUTERPE.

Implements Phase 4 Discovery Science:
1. Score calibration (Min-Max, Z-score, Reciprocal Rank Fusion, Linear Fusion).
2. Maximal Marginal Relevance (MMR) with embedding cosine similarity.
3. Novelty-aware reranking with training-period information-theoretic novelty.
4. Artist-diversity constraints (bounded repetition per artist).
5. Pareto frontier identification across multi-objective trade-offs.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np


def min_max_normalize(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize score array to [0, 1]."""
    s_min = np.min(scores)
    s_max = np.max(scores)
    denom = s_max - s_min
    if denom < eps:
        return np.ones_like(scores, dtype=np.float64) * 0.5
    return (scores - s_min) / denom


def z_score_normalize(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardize score array to zero mean and unit variance."""
    mean = np.mean(scores)
    std = np.std(scores)
    if std < eps:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - mean) / std


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, List[int]],
    k_rrf: int = 60,
    top_k: int = 100,
) -> List[Tuple[int, float]]:
    """Fuse multiple ranked candidate lists using Reciprocal Rank Fusion (RRF).
    
    RRF(i) = sum_{m in Models} 1 / (k_rrf + rank_m(i))
    
    Args:
        ranked_lists: Dict mapping model_name -> ordered list of item IDs.
        k_rrf: Smoothing constant (standard literature default = 60).
        top_k: Number of fused items to return.
    Returns:
        List of (item_id, rrf_score) sorted descending by RRF score.
    """
    rrf_scores: Dict[int, float] = {}

    for _, items in ranked_lists.items():
        for rank, item_id in enumerate(items, 1):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (k_rrf + rank))

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_k]


def linear_score_fusion(
    candidate_scores: Dict[str, Dict[int, float]],
    weights: Dict[str, float],
    top_k: int = 100,
) -> List[Tuple[int, float]]:
    """Fuse normalized candidate scores linearly across models.
    
    S_fused(i) = sum_{m} w_m * s_m(i)
    
    Args:
        candidate_scores: Dict mapping model_name -> {item_id: normalized_score}.
        weights: Dict mapping model_name -> weight w_m.
    Returns:
        List of (item_id, fused_score) sorted descending.
    """
    all_items: Set[int] = set()
    for m in candidate_scores:
        all_items.update(candidate_scores[m].keys())

    fused_scores: Dict[int, float] = {}
    for item in all_items:
        score = 0.0
        for m, w in weights.items():
            score += w * candidate_scores.get(m, {}).get(item, 0.0)
        fused_scores[item] = score

    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_k]


def mmr_rerank(
    candidate_ids: List[int],
    relevance_scores: np.ndarray,
    item_embeddings: np.ndarray,
    top_k: int = 20,
    lambda_param: float = 0.5,
) -> List[int]:
    """Maximal Marginal Relevance (MMR) greedy slate selection (Carbonell & Goldstein, 1998).
    
    Objective:
        MMR(i) = lambda * Rel(u, i) - (1 - lambda) * max_{j in S} CosSim(v_i, v_j)
        
    Args:
        candidate_ids: Candidate track indices of length C.
        relevance_scores: Array of shape (C,) normalized relevance scores in [0, 1].
        item_embeddings: Matrix of shape (N_total, d) with L2-normalized item vectors.
        top_k: Size of final recommendation slate.
        lambda_param: Trade-off parameter (1.0 = pure relevance, 0.0 = maximal diversity).
    Returns:
        Ordered list of selected track IDs of length min(top_k, C).
    """
    n_candidates = len(candidate_ids)
    if n_candidates == 0 or top_k <= 0:
        return []

    # Normalized item vectors for candidate set: (C, d)
    cand_vectors = item_embeddings[candidate_ids]  # (C, d)
    # Ensure vectors are unit normalized
    norms = np.linalg.norm(cand_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cand_vectors = cand_vectors / norms

    # Precompute candidate-candidate pairwise cosine similarity matrix: (C, C)
    sim_matrix = np.dot(cand_vectors, cand_vectors.T)

    selected_indices: List[int] = []
    unselected_indices: Set[int] = set(range(n_candidates))

    # Normalize relevance scores to [0, 1] if not already
    rel_norm = min_max_normalize(relevance_scores)

    # First item is purely the most relevant candidate
    first_idx = int(np.argmax(rel_norm))
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)

    # Greedy selection for remaining k - 1 items
    target_k = min(top_k, n_candidates)
    while len(selected_indices) < target_k and unselected_indices:
        unselected_list = list(unselected_indices)
        rel_sub = rel_norm[unselected_list]

        # Max similarity to any already selected item
        sim_sub = np.max(sim_matrix[unselected_list][:, selected_indices], axis=1)

        # MMR equation
        mmr_scores = lambda_param * rel_sub - (1.0 - lambda_param) * sim_sub
        best_pos = int(np.argmax(mmr_scores))
        best_candidate = unselected_list[best_pos]

        selected_indices.append(best_candidate)
        unselected_indices.remove(best_candidate)

    return [candidate_ids[idx] for idx in selected_indices]


def novelty_rerank(
    candidate_ids: List[int],
    relevance_scores: np.ndarray,
    item_novelty_scores: np.ndarray,
    top_k: int = 20,
    beta: float = 0.3,
) -> List[int]:
    """Novelty-aware reranking balancing relevance and self-information.
    
    Score(u, i) = (1 - beta) * Rel_norm(u, i) + beta * Novelty_norm(i)
    
    Args:
        candidate_ids: List of candidate track indices.
        relevance_scores: Relevance scores array of shape (C,).
        item_novelty_scores: Precomputed self-information bits for each candidate track.
        top_k: Size of slate.
        beta: Weight on novelty in [0, 1].
    """
    rel_norm = min_max_normalize(relevance_scores)
    nov_norm = min_max_normalize(item_novelty_scores)

    combined_scores = (1.0 - beta) * rel_norm + beta * nov_norm
    ranked_order = np.argsort(-combined_scores)[:top_k]
    return [candidate_ids[idx] for idx in ranked_order]


def apply_artist_constraints(
    ranked_item_ids: List[int],
    track_to_artist: np.ndarray,
    max_per_artist: int = 2,
    top_k: int = 20,
) -> List[int]:
    """Apply bounded repetition constraint: at most max_per_artist tracks per artist.
    
    Greedily filters candidate stream, skipping tracks from artists whose quota is exhausted.
    """
    slate: List[int] = []
    artist_counts: Dict[int, int] = {}

    for item_id in ranked_item_ids:
        artist_id = int(track_to_artist[item_id])
        if artist_counts.get(artist_id, 0) < max_per_artist:
            slate.append(item_id)
            artist_counts[artist_id] = artist_counts.get(artist_id, 0) + 1
            if len(slate) == top_k:
                break

    # If constraint eliminated too many candidates to reach top_k, fill from remaining
    if len(slate) < top_k:
        slate_set = set(slate)
        for item_id in ranked_item_ids:
            if item_id not in slate_set:
                slate.append(item_id)
                if len(slate) == top_k:
                    break

    return slate


def identify_pareto_frontier(
    configs: List[Dict[str, float]],
    objectives: Dict[str, str],
) -> List[Dict[str, float]]:
    """Identify Pareto-optimal non-dominated configurations across multiple objectives.
    
    Args:
        configs: List of configuration dictionaries with metric scores.
        objectives: Dict mapping metric_name -> 'max' or 'min'.
    Returns:
        List of non-dominated configuration dictionaries with 'is_pareto_optimal': True.
    """
    n = len(configs)
    is_dominated = [False] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Check if j dominates i
            # j dominates i if j is at least as good on all objectives, and strictly better on >= 1
            all_at_least_as_good = True
            strictly_better_on_one = False

            for obj, direction in objectives.items():
                val_i = configs[i][obj]
                val_j = configs[j][obj]

                if direction == "max":
                    if val_j < val_i:
                        all_at_least_as_good = False
                        break
                    elif val_j > val_i:
                        strictly_better_on_one = True
                elif direction == "min":
                    if val_j > val_i:
                        all_at_least_as_good = False
                        break
                    elif val_j < val_i:
                        strictly_better_on_one = True

            if all_at_least_as_good and strictly_better_on_one:
                is_dominated[i] = True
                break

    pareto_configs = []
    for i in range(n):
        cfg = dict(configs[i])
        cfg["is_pareto_optimal"] = not is_dominated[i]
        pareto_configs.append(cfg)

    return pareto_configs
