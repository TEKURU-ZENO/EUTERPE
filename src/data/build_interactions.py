"""User-item interaction construction module for EUTERPE.

Workstream 1.4:
Aggregates clean scrobbles into user-item interaction records, computing:
1. Binary preference: whether user interacted with the track (p_ui = 1).
2. Weighted frequency: play count r_ui = sum of scrobbles.
3. Implicit confidence: c_ui = 1 + alpha * log(1 + r_ui) with alpha = 40.0.
Exports `data/processed/interactions.parquet`.
"""

import math
import time
from pathlib import Path
import polars as pl
from euterpe.config import default_config

BASE_DIR = Path(__file__).resolve().parents[2]
FILTERED_PARQUET = BASE_DIR / "data" / "interim" / "filtered_events.parquet"
INTERACTIONS_PARQUET = BASE_DIR / "data" / "processed" / "interactions.parquet"


def build_user_item_interactions(
    input_parquet: Path,
    output_parquet: Path,
    alpha: float = 40.0
) -> pl.DataFrame:
    """Aggregate listening history into user-item interaction pairs."""
    print(f"Building user-item interaction table from {input_parquet}...")
    t0 = time.time()
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    
    df = pl.read_parquet(
        input_parquet,
        columns=["user_id", "recording_key", "normalized_track_key", "artist_name", "track_name", "epoch_seconds"]
    )
    
    total_events = len(df)
    print(f"Aggregating {total_events:,} events by (user_id, recording_key)...")
    
    # Group by user_id and recording_key
    interactions = df.group_by(["user_id", "recording_key"]).agg([
        pl.len().alias("play_count"),
        pl.col("normalized_track_key").first().alias("normalized_track_key"),
        pl.col("artist_name").first().alias("display_artist"),
        pl.col("track_name").first().alias("display_track"),
        pl.col("epoch_seconds").min().alias("first_timestamp"),
        pl.col("epoch_seconds").max().alias("last_timestamp"),
    ])
    
    # Compute binary preference and Hu et al. implicit confidence
    print(f"Computing implicit confidence c_ui = 1 + {alpha} * log(1 + r_ui)...")
    interactions = interactions.with_columns([
        pl.lit(1).alias("binary_preference"),
        (1.0 + alpha * (pl.col("play_count") + 1.0).log()).alias("confidence")
    ])
    
    # Sort by user_id and play_count DESC
    interactions = interactions.sort(["user_id", "play_count"], descending=[False, True])
    
    print(f"Total unique (user, item) interaction pairs: {len(interactions):,}")
    print(f"Saving interactions table to {output_parquet}...")
    interactions.write_parquet(output_parquet, compression="zstd")
    
    elapsed = time.time() - t0
    print(f"Interactions construction complete in {elapsed:.2f}s.")
    return interactions


if __name__ == "__main__":
    if FILTERED_PARQUET.exists():
        build_user_item_interactions(FILTERED_PARQUET, INTERACTIONS_PARQUET)
    else:
        print(f"Required file {FILTERED_PARQUET} not found.")
