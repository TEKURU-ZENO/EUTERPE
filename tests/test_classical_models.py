"""
Unit tests for EUTERPE Phase 2 Classical Recommendation Models.
Verifies popularity baselines, ALS, BPR, consumed-item masking, and checkpoint persistence.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import polars as pl
import scipy.sparse as sp
import tempfile
from pathlib import Path

from euterpe.models.popularity import (
    GlobalPopularityRecommender,
    RecencyDecayedPopularityRecommender,
)
from euterpe.models.classical.als import ImplicitALSRecommender
from euterpe.models.classical.bpr import BPRRecommender


def create_toy_dataset():
    """Create a small toy interaction matrix for fast deterministic testing."""
    # 5 users, 8 items
    # User 0 has listened to items 0, 1
    # User 1 has listened to items 0, 2
    # User 2 has listened to items 0, 1, 3
    # User 3 has listened to items 4, 5
    # User 4 has listened to items 6, 7
    rows = [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    cols = [0, 1, 0, 2, 0, 1, 3, 4, 5, 6, 7]
    data = [10.0, 5.0, 8.0, 2.0, 15.0, 7.0, 3.0, 4.0, 1.0, 6.0, 2.0]
    
    mat = sp.csr_matrix((data, (rows, cols)), shape=(5, 8), dtype=np.float32)
    return mat


def test_global_popularity_consumed_masking():
    mat = create_toy_dataset()
    model = GlobalPopularityRecommender(mode="play_count")
    model.fit(mat)
    
    # Recommend top 3 items for User 0 (who consumed items 0 and 1)
    recs = model.recommend_all(mat, k=3)
    
    assert 0 in recs
    user_0_recs = recs[0]
    assert len(user_0_recs) == 3
    # Items 0 and 1 must NOT be in recommendations
    assert 0 not in user_0_recs
    assert 1 not in user_0_recs
    # Most popular remaining items: item 6 (6.0), item 4 (4.0), item 3 (3.0)
    assert user_0_recs[0] == 6
    assert user_0_recs[1] == 4
    assert user_0_recs[2] == 3


def test_recency_popularity_decay():
    t_now = 1_000_000
    # Create training events dataframe
    # Item 0 played 5 times 20 days ago (1,728,000s ago)
    # Item 1 played 3 times 1 day ago (86,400s ago)
    df = pl.DataFrame({
        "recording_key": ["track_0"] * 5 + ["track_1"] * 3,
        "epoch_seconds": [t_now - 20 * 86400] * 5 + [t_now - 1 * 86400] * 3,
    })
    item_to_idx = {"track_0": 0, "track_1": 1}
    
    model = RecencyDecayedPopularityRecommender(half_life_days=10.0)
    model.fit_from_dataframe(df, item_to_idx, t_cutoff_epoch=t_now)
    
    # Item 0 decayed score: 5 * 2^(-20/10) = 5 * 0.25 = 1.25
    # Item 1 decayed score: 3 * 2^(-1/10) ≈ 3 * 0.933 = 2.799
    assert model.global_scores[1] > model.global_scores[0]
    assert model.ranked_item_indices[0] == 1


def test_als_fit_predict_save_load():
    mat = create_toy_dataset()
    als = ImplicitALSRecommender(factors=8, regularization=0.01, iterations=10, random_state=42)
    als.fit(mat, show_progress=False)
    
    recs = als.recommend_all(mat, k=3)
    assert len(recs) == 5
    assert len(recs[0]) == 3
    
    # User 0 consumed items 0, 1 -> verify masked
    assert 0 not in recs[0]
    assert 1 not in recs[0]
    
    # Checkpoint save/load test
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "als_test.npz"
        als.save(path)
        
        loaded = ImplicitALSRecommender()
        loaded.load(path)
        assert loaded.factors == als.factors
        np.testing.assert_allclose(loaded.user_factors, als.user_factors, atol=1e-5)
        np.testing.assert_allclose(loaded.item_factors, als.item_factors, atol=1e-5)


def test_bpr_fit_predict_save_load():
    mat = create_toy_dataset()
    bpr = BPRRecommender(factors=8, regularization=0.01, iterations=10, random_state=42)
    bpr.fit(mat, show_progress=False)
    
    recs = bpr.recommend_all(mat, k=2)
    assert len(recs) == 5
    assert len(recs[0]) == 2
    # Verify consumed item masked
    assert 0 not in recs[0]
    assert 1 not in recs[0]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bpr_test.npz"
        bpr.save(path)
        
        loaded = BPRRecommender()
        loaded.load(path)
        np.testing.assert_allclose(loaded.user_factors, bpr.user_factors, atol=1e-5)
        np.testing.assert_allclose(loaded.item_factors, bpr.item_factors, atol=1e-5)
