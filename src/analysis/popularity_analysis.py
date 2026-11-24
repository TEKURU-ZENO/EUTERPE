"""Track and artist popularity analysis for EUTERPE.

Workstream 1.7:
Computes empirical popularity distributions, concentration ratios, and
measures the Matthew Effect in raw listening logs.
"""

from pathlib import Path
import numpy as np
import polars as pl
from euterpe.discovery.metrics import gini_coefficient


def analyze_popularity(df_events: pl.DataFrame) -> dict:
    """Analyze track and artist popularity skew and concentration."""
    print("Computing track and artist popularity distributions...")
    
    # Track level popularity
    track_counts_df = df_events.group_by("recording_key").len().sort("len", descending=True)
    track_counts = track_counts_df["len"].to_numpy()
    total_scrobbles = float(np.sum(track_counts))
    n_tracks = len(track_counts)
    
    # Cumulative shares
    top_1pct_idx = max(1, int(n_tracks * 0.01))
    top_5pct_idx = max(1, int(n_tracks * 0.05))
    top_10pct_idx = max(1, int(n_tracks * 0.10))
    top_20pct_idx = max(1, int(n_tracks * 0.20))
    
    top_1pct_share = float(np.sum(track_counts[:top_1pct_idx]) / total_scrobbles) * 100
    top_5pct_share = float(np.sum(track_counts[:top_5pct_idx]) / total_scrobbles) * 100
    top_10pct_share = float(np.sum(track_counts[:top_10pct_idx]) / total_scrobbles) * 100
    top_20pct_share = float(np.sum(track_counts[:top_20pct_idx]) / total_scrobbles) * 100
    
    track_gini = gini_coefficient(track_counts)
    
    # Artist level popularity
    artist_counts_df = df_events.group_by("artist_name").len().sort("len", descending=True)
    artist_counts = artist_counts_df["len"].to_numpy()
    n_artists = len(artist_counts)
    
    top_artist_1pct = max(1, int(n_artists * 0.01))
    top_artist_10pct = max(1, int(n_artists * 0.10))
    
    top_artist_1pct_share = float(np.sum(artist_counts[:top_artist_1pct]) / total_scrobbles) * 100
    top_artist_10pct_share = float(np.sum(artist_counts[:top_artist_10pct]) / total_scrobbles) * 100
    artist_gini = gini_coefficient(artist_counts)
    
    return {
        "n_unique_tracks": n_tracks,
        "n_unique_artists": n_artists,
        "track_gini": round(track_gini, 4),
        "artist_gini": round(artist_gini, 4),
        "track_top_1pct_share_pct": round(top_1pct_share, 2),
        "track_top_5pct_share_pct": round(top_5pct_share, 2),
        "track_top_10pct_share_pct": round(top_10pct_share, 2),
        "track_top_20pct_share_pct": round(top_20pct_share, 2),
        "artist_top_1pct_share_pct": round(top_artist_1pct_share, 2),
        "artist_top_10pct_share_pct": round(top_artist_10pct_share, 2),
    }
