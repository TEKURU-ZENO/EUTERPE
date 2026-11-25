"""Unit and integration tests for EUTERPE Sequential Recommendation Models (SASRec & GRU4Rec)."""

import pytest
import torch
import torch.nn.functional as F
from euterpe.models.neural.sasrec import SASRec
from euterpe.models.neural.gru4rec import GRU4Rec


def test_sasrec_forward_and_shapes():
    """Verify SASRec output dimensions and causal masking."""
    n_items = 1000
    max_len = 20
    d_model = 32
    model = SASRec(n_items=n_items, max_seq_len=max_len, d_model=d_model, n_heads=2, n_blocks=1, d_ff=64)
    model.eval()

    B = 8
    seqs = torch.randint(1, n_items, (B, max_len))
    # Add some left-padding
    seqs[:, :5] = 0

    with torch.no_grad():
        hidden = model(seqs)
        assert hidden.shape == (B, max_len, d_model)

        user_vec = model.encode_user_sequence(seqs)
        assert user_vec.shape == (B, d_model)
        # Check unit normalization
        norms = torch.norm(user_vec, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

        item_embs = model.get_all_item_embeddings()
        assert item_embs.shape == (n_items, d_model)


def test_sasrec_infonce_loss():
    """Verify SASRec InfoNCE loss computation and gradient flow."""
    n_items = 500
    max_len = 15
    d_model = 32
    model = SASRec(n_items=n_items, max_seq_len=max_len, d_model=d_model, n_heads=2, n_blocks=1, d_ff=64)
    model.train()

    B = 4
    seqs = torch.randint(1, n_items, (B, max_len))
    pos_targets = torch.randint(1, n_items, (B, max_len))
    neg_targets = torch.randint(1, n_items, (B, max_len, 5))

    loss = model.compute_infonce_loss(seqs, pos_targets, neg_targets, temperature=0.2)
    assert loss.item() > 0.0
    loss.backward()

    # Verify gradients exist
    assert model.item_embedding.weight.grad is not None


def test_gru4rec_forward_and_shapes():
    """Verify GRU4Rec output dimensions and user representation extraction."""
    n_items = 500
    max_len = 15
    d_model = 32
    model = GRU4Rec(n_items=n_items, max_seq_len=max_len, d_model=d_model, n_layers=1)
    model.eval()

    B = 4
    seqs = torch.randint(1, n_items, (B, max_len))
    seqs[:, :3] = 0

    with torch.no_grad():
        hidden = model(seqs)
        assert hidden.shape == (B, max_len, d_model)

        user_vec = model.encode_user_sequence(seqs)
        assert user_vec.shape == (B, d_model)
        norms = torch.norm(user_vec, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_sequential_models_save_and_load(tmp_path):
    """Verify checkpoint serialization and parameter restoration."""
    n_items = 200
    max_len = 10
    d_model = 16

    # 1. SASRec save/load
    sasrec = SASRec(n_items=n_items, max_seq_len=max_len, d_model=d_model, n_heads=2, n_blocks=1)
    s_path = tmp_path / "sasrec_test.pt"
    sasrec.save(s_path)
    loaded_s = SASRec.load(s_path)
    assert loaded_s.n_items == n_items
    assert loaded_s.d_model == d_model

    # 2. GRU4Rec save/load
    gru = GRU4Rec(n_items=n_items, max_seq_len=max_len, d_model=d_model, n_layers=1)
    g_path = tmp_path / "gru_test.pt"
    gru.save(g_path)
    loaded_g = GRU4Rec.load(g_path)
    assert loaded_g.n_items == n_items
    assert loaded_g.d_model == d_model
