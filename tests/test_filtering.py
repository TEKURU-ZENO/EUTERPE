"""Unit tests for scrobble filtering and rapid spam detection."""

import polars as pl
import pytest


def test_rapid_duplicate_filtering_logic():
    # Synthetic events: user_001 listens to track_A at t=100, and again at t=110 (10s delta -> duplicate)
    # user_001 listens to track_A at t=200 (90s delta -> valid)
    # user_001 listens to track_B at t=210 (different track -> valid even if 10s delta)
    data = [
        {"user_id": "u1", "epoch_seconds": 100, "recording_key": "art:::track_a"},
        {"user_id": "u1", "epoch_seconds": 110, "recording_key": "art:::track_a"},  # duplicate (<30s)
        {"user_id": "u1", "epoch_seconds": 200, "recording_key": "art:::track_a"},  # valid (>30s)
        {"user_id": "u1", "epoch_seconds": 210, "recording_key": "art:::track_b"},  # valid (different track)
        {"user_id": "u2", "epoch_seconds": 105, "recording_key": "art:::track_a"},  # valid (different user)
    ]
    df = pl.DataFrame(data).sort(["user_id", "epoch_seconds"])
    
    df = df.with_columns([
        pl.col("epoch_seconds").diff().over("user_id").alias("delta_seconds"),
        (pl.col("recording_key") == pl.col("recording_key").shift(1).over("user_id")).alias("is_same_track")
    ])
    
    is_dup = (
        (df["is_same_track"] == True) &
        (df["delta_seconds"] >= 0) &
        (df["delta_seconds"] < 30)
    ).fill_null(False)
    
    filtered = df.filter(~is_dup)
    assert len(filtered) == 4
    # The duplicate at t=110 must be omitted
    assert 110 not in filtered["epoch_seconds"].to_list()
