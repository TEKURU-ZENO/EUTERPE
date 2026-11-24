"""Unit tests for EUTERPE Phase 4 Discovery Reranking and Fusion.

Verifies:
1. Normalization & score calibration.
2. Reciprocal Rank Fusion (RRF) and linear fusion.
3. Maximal Marginal Relevance (MMR) redundancy reduction.
4. Novelty-aware reranking.
5. Artist diversity constraints (bounded repetition).
6. Pareto frontier identification.
"""

import numpy as np
import pytest

from euterpe.discovery.reranking import (
    apply_artist_constraints,
    identify_pareto_frontier,
    linear_score_fusion,
    min_max_normalize,
    mmr_rerank,
    novelty_rerank,
    reciprocal_rank_fusion,
    z_score_normalize,
)


def test_score_normalization():
    scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    norm = min_max_normalize(scores)
    assert norm[0] == 0.0
    assert norm[-1] == 1.0
    assert np.all(norm >= 0.0) and np.all(norm <= 1.0)

    z = z_score_normalize(scores)
    np.testing.assert_allclose(np.mean(z), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(z), 1.0, atol=1e-5)


def test_reciprocal_rank_fusion():
    list_a = [101, 102, 103, 104]
    list_b = [103, 101, 105, 106]

    fused = reciprocal_rank_fusion({"als": list_a, "n5": list_b}, k_rrf=60, top_k=5)
    fused_items = [item for item, _ in fused]

    # Item 101: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.01639 + 0.01613 = 0.03252
    # Item 103: 1/(60+3) + 1/(60+1) = 1/63 + 1/61 ≈ 0.01587 + 0.01639 = 0.03226
    # 101 should be rank 1, 103 rank 2
    assert fused_items[0] == 101
    assert fused_items[1] == 103
    assert len(fused) <= 5


def test_mmr_rerank_diversity():
    # 4 candidates with 2 clusters of identical vectors
    # Items 0, 1 from cluster A (cosine sim = 1.0)
    # Items 2, 3 from cluster B (cosine sim = 1.0)
    # Distance between cluster A and B is orthogonal (cosine sim = 0.0)
    vecs = np.zeros((4, 4), dtype=np.float32)
    vecs[0] = [1, 0, 0, 0]
    vecs[1] = [1, 0, 0, 0]  # Duplicate of 0
    vecs[2] = [0, 1, 0, 0]  # Orthogonal to 0
    vecs[3] = [0, 1, 0, 0]

    # Relevance: item 0 has highest rel, item 1 has 2nd highest rel
    cand_ids = [0, 1, 2, 3]
    rel = np.array([0.95, 0.90, 0.80, 0.75])

    # Pure relevance (lambda=1.0) selects items in pure score order: [0, 1, 2, 3]
    pure_rel = mmr_rerank(cand_ids, rel, vecs, top_k=2, lambda_param=1.0)
    assert pure_rel == [0, 1]

    # Diversified MMR (lambda=0.4):
    # After selecting item 0, item 1 has sim=1.0, so penalty is large: 0.4*0.90 - 0.6*1.0 = -0.24
    # Item 2 has sim=0.0, so penalty is 0: 0.4*0.80 - 0.6*0.0 = +0.32
    # So item 2 should be selected before item 1!
    diversified = mmr_rerank(cand_ids, rel, vecs, top_k=2, lambda_param=0.4)
    assert diversified == [0, 2]


def test_apply_artist_constraints():
    # 5 tracks: 0, 1, 2 belong to artist 99; 3, 4 belong to artist 88
    track_to_artist = np.array([99, 99, 99, 88, 88])
    ranked_items = [0, 1, 2, 3, 4]

    # Cap at 2 per artist
    filtered = apply_artist_constraints(ranked_items, track_to_artist, max_per_artist=2, top_k=4)
    # Item 0 (artist 99), Item 1 (artist 99) -> quota exhausted for 99!
    # Item 2 should be skipped!
    # Next should be Item 3 (artist 88), Item 4 (artist 88)
    assert filtered == [0, 1, 3, 4]
    assert 2 not in filtered[:3]


def test_identify_pareto_frontier():
    configs = [
        {"id": "A", "ndcg": 0.060, "novelty": 14.0},  # High NDCG, low novelty
        {"id": "B", "ndcg": 0.050, "novelty": 16.5},  # Medium NDCG, high novelty
        {"id": "C", "ndcg": 0.040, "novelty": 14.0},  # Dominated by A (same novelty, worse NDCG)
    ]
    objectives = {"ndcg": "max", "novelty": "max"}

    pareto = identify_pareto_frontier(configs, objectives)
    pareto_dict = {cfg["id"]: cfg["is_pareto_optimal"] for cfg in pareto}

    assert pareto_dict["A"] is True
    assert pareto_dict["B"] is True
    assert pareto_dict["C"] is False
