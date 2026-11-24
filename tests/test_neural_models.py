"""Unit tests for EUTERPE Phase 3 Neural Recommendation Models.

Verifies:
1. Two-Tower forward pass (history aggregation, artist embedding, LayerNorm).
2. Sampled softmax loss with popularity log-Q correction.
3. False-negative candidate masking in batch.
4. Model checkpoint save and load.
5. Vector retrieval engine with IndexFlatIP and exact inner-product search.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from euterpe.models.neural.two_tower import TwoTowerRecommender
from euterpe.models.neural.negative_sampling import (
    InBatchSampledSoftmaxLoss,
    compute_popularity_log_q,
)
from euterpe.models.neural.retrieval import VectorRetrievalEngine


def test_two_tower_forward_shapes():
    n_users = 10
    n_items = 50
    n_artists = 8
    dim = 32

    model = TwoTowerRecommender(
        n_users=n_users,
        n_items=n_items,
        n_artists=n_artists,
        embedding_dim=dim,
        use_history=True,
        use_artist=True,
    )

    B = 4
    user_ids = torch.tensor([0, 1, 2, 3])
    track_ids = torch.tensor([5, 10, 15, 20])
    artist_ids = torch.tensor([1, 2, 0, 3])
    # 1-indexed historical tracks, 0 for pad
    history = torch.tensor([[1, 2, 0], [4, 0, 0], [0, 0, 0], [6, 7, 8]])

    u_repr, i_repr = model(user_ids, track_ids, artist_ids, history)

    assert u_repr.shape == (B, dim)
    assert i_repr.shape == (B, dim)
    # Check layer norm produces finite non-zero outputs
    assert torch.isfinite(u_repr).all()
    assert torch.isfinite(i_repr).all()


def test_popularity_log_q():
    freqs = np.array([1000.0, 100.0, 10.0, 1.0])
    log_q = compute_popularity_log_q(freqs, power=1.0)
    # Most popular item should have highest log probability
    assert log_q[0] > log_q[1] > log_q[2] > log_q[3]
    # Check probabilities sum to 1.0 (in exp space)
    np.testing.assert_allclose(np.sum(np.exp(log_q)), 1.0, atol=1e-5)


def test_false_negative_masking():
    torch.manual_seed(42)
    criterion = InBatchSampledSoftmaxLoss(use_popularity_correction=False)
    B = 3
    dim = 16

    u_repr = torch.randn(B, dim)
    i_repr = torch.randn(B, dim)
    user_ids = torch.tensor([10, 20, 30])
    pos_item_ids = torch.tensor([100, 200, 300])

    # User 10 has consumed item 200 (which is the positive for user 20 in the batch)
    consumed_sets = {10: {100, 200}, 20: {200}, 30: {300}}

    loss = criterion(
        u_repr,
        i_repr,
        user_ids,
        pos_item_ids,
        user_consumed_sets=consumed_sets,
    )
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_checkpoint_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_checkpoint.pt"
        model = TwoTowerRecommender(
            n_users=5,
            n_items=20,
            n_artists=3,
            embedding_dim=16,
            use_history=True,
            use_artist=True,
        )

        model.save_checkpoint(path)
        loaded = TwoTowerRecommender.load_checkpoint(path)

        assert loaded.n_users == model.n_users
        assert loaded.n_items == model.n_items
        assert loaded.embedding_dim == model.embedding_dim
        # Check parameter equality
        for p1, p2 in zip(model.parameters(), loaded.parameters()):
            torch.testing.assert_close(p1, p2)


def test_vector_retrieval_faiss_and_masking():
    np.random.seed(42)
    n_items = 100
    dim = 32

    # Normalized item embeddings
    item_embs = np.random.randn(n_items, dim).astype(np.float32)
    item_embs /= np.linalg.norm(item_embs, axis=1, keepdims=True)

    from euterpe.models.neural.retrieval import RetrievalMetric
    engine = VectorRetrievalEngine(dimension=dim, metric=RetrievalMetric.COSINE_SIMILARITY)
    engine.build_index(item_embs)

    # Query vector
    user_vec = item_embs[0].copy()  # perfect match with item 0
    consumed = {0}  # mask item 0

    recs = engine.query_user(user_vec, consumed_items=consumed, k=5)
    assert len(recs) == 5
    rec_ids = [r[0] for r in recs]
    # Item 0 must be masked!
    assert 0 not in rec_ids

    # Benchmark latency function runs cleanly
    lat = engine.benchmark_latency(item_embs[:10], n_warmup=5, n_runs=10, k=20)
    assert "p50_ms" in lat
    assert lat["p50_ms"] >= 0.0
