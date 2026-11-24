"""Unit tests for EUTERPE recommendation and discovery science metrics."""

import math
import numpy as np
import pytest
from euterpe.discovery.metrics import (
    recall_at_k,
    capped_recall_at_k,
    precision_at_k,
    ndcg_at_k,
    mrr_at_k,
    novelty_at_k,
    intra_list_diversity,
    catalog_coverage,
    artist_coverage,
    gini_coefficient,
    serendipity_at_k,
    paired_bootstrap_ci,
)


def test_recall_at_k():
    actual = {10, 20, 30}
    predicted = [10, 99, 20, 88, 77]
    
    # Conventional Recall@K: |predicted[:k] ∩ actual| / |actual|
    # In top-3: [10, 99, 20], hits = {10, 20} (2 hits) out of len(actual) = 3 -> 2/3
    assert pytest.approx(recall_at_k(actual, predicted, k=3)) == 2.0 / 3.0
    
    # In top-1: [10], hit = {10} (1 hit) out of len(actual) = 3 -> 1/3
    assert pytest.approx(recall_at_k(actual, predicted, k=1)) == 1.0 / 3.0
    
    # Zero hits
    assert recall_at_k(actual, [1, 2, 3], k=3) == 0.0


def test_capped_recall_at_k():
    actual = {10, 20, 30}
    predicted = [10, 99, 20, 88, 77]
    
    # Capped Recall@K: |predicted[:k] ∩ actual| / min(k, |actual|)
    # In top-1: [10], hit = {10} out of min(1, 3) = 1 -> 1/1 = 1.0
    assert pytest.approx(capped_recall_at_k(actual, predicted, k=1)) == 1.0
    assert pytest.approx(capped_recall_at_k(actual, predicted, k=3)) == 2.0 / 3.0


def test_precision_at_k():
    actual = {10, 20}
    predicted = [10, 99, 88, 77]
    # In top-4: 1 hit out of 4 -> 0.25
    assert pytest.approx(precision_at_k(actual, predicted, k=4)) == 0.25


def test_ndcg_at_k():
    actual = {10, 20}
    
    # Perfect ranking: [10, 20] -> NDCG = 1.0
    predicted_perfect = [10, 20, 30]
    assert pytest.approx(ndcg_at_k(actual, predicted_perfect, k=2)) == 1.0
    
    # Hit at rank 1 only: DCG = 1/log2(2) = 1.0. IDCG for 2 items = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    predicted_rank1 = [10, 99, 88]
    expected_ndcg = (1.0 / 1.0) / (1.0 + 1.0 / math.log2(3))
    assert pytest.approx(ndcg_at_k(actual, predicted_rank1, k=2)) == expected_ndcg
    
    # No hits
    assert ndcg_at_k(actual, [99, 88], k=2) == 0.0


def test_mrr_at_k():
    actual = {42}
    assert pytest.approx(mrr_at_k(actual, [42, 1, 2], k=3)) == 1.0
    assert pytest.approx(mrr_at_k(actual, [1, 42, 2], k=3)) == 0.5
    assert pytest.approx(mrr_at_k(actual, [1, 2, 42], k=3)) == 1.0 / 3.0
    assert mrr_at_k(actual, [1, 2, 3], k=3) == 0.0


def test_novelty_at_k():
    # item 1 is mainstream (P=0.5, self-info = 1 bit)
    # item 2 is long-tail (P=0.03125, self-info = 5 bits)
    item_probs = {1: 0.5, 2: 0.03125}
    predicted = [1, 2]
    
    # Novelty = (1 + 5) / 2 = 3.0 bits
    assert pytest.approx(novelty_at_k(predicted, item_probs, k=2)) == 3.0


def test_intra_list_diversity():
    # 2 orthogonal 2D vectors: [1, 0] and [0, 1]
    # Cosine similarity = 0 -> distance = 1.0 -> ILD = 1.0
    embeddings = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.0, 1.0]),
    }
    assert pytest.approx(intra_list_diversity([1, 2], embeddings, k=2)) == 1.0
    
    # 2 identical vectors: Cosine similarity = 1 -> distance = 0.0 -> ILD = 0.0
    embeddings_same = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
    }
    assert pytest.approx(intra_list_diversity([1, 2], embeddings_same, k=2)) == 0.0


def test_catalog_and_artist_coverage():
    all_recs = [[1, 2], [2, 3], [3, 4]]
    # Unique recs: {1, 2, 3, 4} (4 items) out of catalog of 10 items -> 0.4
    assert pytest.approx(catalog_coverage(all_recs, total_catalog_items=10)) == 0.4
    
    item_to_artist = {1: 100, 2: 100, 3: 200, 4: 300}
    # Unique artists: {100, 200, 300} (3 artists) out of 5 catalog artists -> 3/5 = 0.6
    assert pytest.approx(artist_coverage(all_recs, item_to_artist, total_catalog_artists=5)) == 0.6


def test_gini_coefficient():
    # Completely equal distribution: [10, 10, 10, 10] -> Gini = 0.0
    equal_counts = np.array([10, 10, 10, 10])
    assert pytest.approx(gini_coefficient(equal_counts), abs=1e-5) == 0.0
    
    # Highly unequal distribution: one item gets 1000 recs, others get 0
    unequal_counts = np.array([0, 0, 0, 0, 1000])
    assert gini_coefficient(unequal_counts) > 0.75


def test_bootstrap_ci():
    values = np.array([0.2, 0.22, 0.21, 0.23, 0.19, 0.24, 0.20])
    mean, lower, upper = paired_bootstrap_ci(values, n_bootstraps=500, ci=0.95, seed=42)
    assert lower <= mean <= upper
    assert pytest.approx(mean) == float(np.mean(values))
