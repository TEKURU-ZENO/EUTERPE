"""Raw scrobble ingestion pipeline for Last.fm 1K.

Workstream 1.2: Parses the full 19.15M listening events, parses ISO-8601 timestamps,
records parsing anomalies, sorts events strictly chronologically, and exports
data/interim/raw_events.parquet.
"""

import datetime
import json
import os
import sys
import time
from pathlib import Path
import polars as pl

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_FILE = BASE_DIR / "data" / "raw" / "userid-timestamp-artid-artname-traid-traname.tsv"
OUTPUT_PARQUET = BASE_DIR / "data" / "interim" / "raw_events.parquet"
INGEST_MANIFEST = BASE_DIR / "artifacts" / "manifests" / "ingestion_summary.json"


def ingest_raw_scrobbles(input_tsv: Path, output_parquet: Path) -> dict:
    """Parse raw TSV stream into a typed, chronologically ordered Parquet dataset."""
    print(f"Starting raw scrobble ingestion from {input_tsv}...")
    t0 = time.time()
    
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    INGEST_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    malformed_lines = 0
    empty_lines = 0
    missing_track_or_artist = 0
    
    # We parse the file line by line or via polars with robust schema
    # Last.fm 1K has 6 tab-separated columns
    print("Reading and parsing raw TSV stream via Polars LazyFrame...")
    
    try:
        # Schema definition
        schema = {
            "column_1": pl.Utf8,  # user_id
            "column_2": pl.Utf8,  # timestamp
            "column_3": pl.Utf8,  # mbid_artist
            "column_4": pl.Utf8,  # artist_name
            "column_5": pl.Utf8,  # mbid_track
            "column_6": pl.Utf8,  # track_name
        }
        
        # Scan csv with tab separator, NO quote character (quotes are literal text in song titles)
        df = pl.read_csv(
            input_tsv,
            separator="\t",
            has_header=False,
            schema=schema,
            quote_char=None,
            truncate_ragged_lines=True,
            ignore_errors=True
        )
        
        total_parsed = len(df)
        print(f"Loaded {total_parsed:,} rows into DataFrame in {time.time() - t0:.2f}s.")
        
        # Rename columns to standard names
        df = df.rename({
            "column_1": "user_id",
            "column_2": "raw_timestamp",
            "column_3": "mbid_artist",
            "column_4": "artist_name",
            "column_5": "mbid_track",
            "column_6": "track_name"
        })
        
        # Filter null or empty artists or track names
        null_filter = (
            df["user_id"].is_not_null() &
            df["raw_timestamp"].is_not_null() &
            df["artist_name"].is_not_null() &
            df["track_name"].is_not_null() &
            (df["artist_name"].str.strip_chars() != "") &
            (df["track_name"].str.strip_chars() != "")
        )
        
        valid_df = df.filter(null_filter)
        discarded_null_count = total_parsed - len(valid_df)
        print(f"Valid scrobbles with non-empty user/artist/track: {len(valid_df):,} (Dropped nulls: {discarded_null_count:,})")
        
        # Parse timestamp to Datetime
        print("Parsing ISO-8601 timestamps and extracting epoch seconds...")
        valid_df = valid_df.with_columns([
            pl.col("raw_timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_unit="ms", strict=False).alias("timestamp")
        ])
        
        # Filter unparseable timestamps
        timestamp_valid = valid_df["timestamp"].is_not_null()
        invalid_ts_count = len(valid_df) - timestamp_valid.sum()
        if invalid_ts_count > 0:
            print(f"Warning: {invalid_ts_count} rows had invalid timestamps.")
            valid_df = valid_df.filter(timestamp_valid)
            
        # Add epoch timestamp in seconds for fast filtering and boundary evaluation
        valid_df = valid_df.with_columns([
            (pl.col("timestamp").dt.epoch("ms") / 1000).cast(pl.Int64).alias("epoch_seconds")
        ])
        
        # Sort chronologically by timestamp ASC, then user_id
        print("Sorting complete event log chronologically (timestamp ASC)...")
        valid_df = valid_df.sort(["epoch_seconds", "user_id"])
        
        # Write to Parquet
        print(f"Writing parsed dataset to {output_parquet}...")
        valid_df.write_parquet(output_parquet, compression="zstd")
        
        elapsed = time.time() - t0
        min_ts = str(valid_df["timestamp"].min())
        max_ts = str(valid_df["timestamp"].max())
        unique_users = valid_df["user_id"].n_unique()
        unique_artists = valid_df["artist_name"].n_unique()
        
        summary = {
            "input_file": str(input_tsv),
            "output_parquet": str(output_parquet),
            "total_parsed_rows": total_parsed,
            "retained_valid_rows": len(valid_df),
            "dropped_missing_fields": discarded_null_count,
            "dropped_invalid_timestamps": invalid_ts_count,
            "unique_users": unique_users,
            "unique_raw_artists": unique_artists,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
            "processing_time_seconds": round(elapsed, 2)
        }
        
        with open(INGEST_MANIFEST, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
        print(f"Ingestion finished in {elapsed:.2f}s.")
        print(f"Retained: {len(valid_df):,} events | Users: {unique_users:,} | Range: {min_ts} -> {max_ts}")
        return summary
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise


if __name__ == "__main__":
    if not RAW_FILE.exists():
        print(f"Error: {RAW_FILE} not found. Please run download_dataset.py first.")
        sys.exit(1)
    ingest_raw_scrobbles(RAW_FILE, OUTPUT_PARQUET)
