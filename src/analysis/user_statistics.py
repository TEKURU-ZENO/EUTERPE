"""User activity analysis module for EUTERPE.

Workstream 1.7:
Computes listener activity distributions, scrobble count percentiles,
active listening day distributions, and listener engagement variance.
"""

from pathlib import Path
import numpy as np
import polars as pl

BASE_DIR = Path(__file__).resolve().parents[2]


def analyze_user_activity(df_events: pl.DataFrame) -> dict:
    """Compute comprehensive user listening activity statistics."""
    print("Computing user activity distribution...")
    user_counts = df_events.group_by("user_id").agg([
        pl.len().alias("scrobble_count"),
        pl.col("epoch_seconds").min().alias("first_seen"),
        pl.col("epoch_seconds").max().alias("last_seen"),
        pl.col("recording_key").n_unique().alias("unique_tracks"),
        pl.col("artist_name").n_unique().alias("unique_artists")
    ])
    
    counts = user_counts["scrobble_count"].to_numpy()
    
    stats = {
        "n_users": len(user_counts),
        "mean_scrobbles": float(np.mean(counts)),
        "std_scrobbles": float(np.std(counts)),
        "min_scrobbles": int(np.min(counts)),
        "p25_scrobbles": float(np.percentile(counts, 25)),
        "median_scrobbles": float(np.median(counts)),
        "p75_scrobbles": float(np.percentile(counts, 75)),
        "p90_scrobbles": float(np.percentile(counts, 90)),
        "p99_scrobbles": float(np.percentile(counts, 99)),
        "max_scrobbles": int(np.max(counts)),
        "mean_unique_tracks": float(user_counts["unique_tracks"].mean()),
        "mean_unique_artists": float(user_counts["unique_artists"].mean()),
    }
    return stats
