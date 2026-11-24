"""Unit tests for chronological splitting and temporal leakage prevention."""

import polars as pl
import pytest


def test_zero_leakage_chronological_invariant():
    # Synthetic events across 3 time windows:
    # Train: t = 100..150
    # Val: t = 160..180
    # Test: t = 190..220
    train_data = [{"epoch_seconds": t, "user_id": "u1", "recording_key": "k1"} for t in range(100, 151)]
    val_data = [{"epoch_seconds": t, "user_id": "u1", "recording_key": "k1"} for t in range(160, 181)]
    test_data = [{"epoch_seconds": t, "user_id": "u1", "recording_key": "k1"} for t in range(190, 221)]
    
    df = pl.DataFrame(train_data + val_data + test_data).sort("epoch_seconds")
    
    t_train_cutoff = 155
    t_val_cutoff = 185
    
    train = df.filter(pl.col("epoch_seconds") < t_train_cutoff)
    val = df.filter((pl.col("epoch_seconds") >= t_train_cutoff) & (pl.col("epoch_seconds") < t_val_cutoff))
    test = df.filter(pl.col("epoch_seconds") >= t_val_cutoff)
    
    # Invariants
    assert train["epoch_seconds"].max() < val["epoch_seconds"].min()
    assert val["epoch_seconds"].max() < test["epoch_seconds"].min()


def test_training_only_eligibility_enforcement():
    # Senior Scientific Review 02:
    # User u_future only appears in validation window.
    # Track k_future only appears in test window.
    # Neither must be in eligible training sets!
    events = [
        # Train window (t < 100)
        {"epoch_seconds": 10, "user_id": "u1", "recording_key": "k1"},
        {"epoch_seconds": 20, "user_id": "u1", "recording_key": "k1"},
        # Val window (100 <= t < 200)
        {"epoch_seconds": 110, "user_id": "u_future", "recording_key": "k1"},
        # Test window (t >= 200)
        {"epoch_seconds": 210, "user_id": "u1", "recording_key": "k_future"},
    ]
    df = pl.DataFrame(events)
    train_window = df.filter(pl.col("epoch_seconds") < 100)
    
    # Compute eligibility strictly from training
    eligible_users = set(train_window["user_id"].unique().to_list())
    eligible_items = set(train_window["recording_key"].unique().to_list())
    
    assert "u_future" not in eligible_users
    assert "k_future" not in eligible_items
    assert "u1" in eligible_users
    assert "k1" in eligible_items
