"""Long-tail curve and catalog exposure distribution analysis for EUTERPE.

Workstream 1.7:
Quantifies the Head, Mid, and Long-Tail partitions of the catalog and measures
power-law slope characteristics.
"""

from pathlib import Path
import numpy as np
import polars as pl


def analyze_long_tail(df_events: pl.DataFrame) -> dict:
    """Classify catalog into Head, Mid, and Long Tail partitions."""
    print("Computing catalog long-tail partitions...")
    track_counts_df = df_events.group_by("recording_key").len().sort("len", descending=True)
    counts = track_counts_df["len"].to_numpy()
    total_events = float(np.sum(counts))
    n_tracks = len(counts)
    
    # Cumulative sum
    cumsum = np.cumsum(counts)
    
    # Pareto point: how many tracks account for 80% of listening?
    idx_80pct = int(np.searchsorted(cumsum, 0.80 * total_events))
    pct_catalog_for_80pct_plays = (idx_80pct / n_tracks) * 100
    
    # Partitions:
    # Head: top 1% tracks
    # Mid: next 19% (1% to 20%)
    # Tail: remaining 80% (20% to 100%)
    head_cutoff = max(1, int(n_tracks * 0.01))
    mid_cutoff = max(1, int(n_tracks * 0.20))
    
    head_plays = float(cumsum[head_cutoff - 1] / total_events) * 100
    mid_plays = float((cumsum[mid_cutoff - 1] - cumsum[head_cutoff - 1]) / total_events) * 100
    tail_plays = float((total_events - cumsum[mid_cutoff - 1]) / total_events) * 100
    
    return {
        "n_catalog_tracks": n_tracks,
        "tracks_for_80pct_plays_count": idx_80pct,
        "tracks_for_80pct_plays_pct": round(pct_catalog_for_80pct_plays, 2),
        "head_partition_pct_catalog": 1.0,
        "head_partition_pct_plays": round(head_plays, 2),
        "mid_partition_pct_catalog": 19.0,
        "mid_partition_pct_plays": round(mid_plays, 2),
        "tail_partition_pct_catalog": 80.0,
        "tail_partition_pct_plays": round(tail_plays, 2)
    }
