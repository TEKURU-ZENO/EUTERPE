"""Unit tests for raw scrobble parsing and timestamp ingestion."""

import datetime
from pathlib import Path
import polars as pl
import pytest
from euterpe.data.normalize import normalize_text_broad


def test_timestamp_parsing():
    raw_ts = "2009-04-08T01:57:47Z"
    dt = datetime.datetime.strptime(raw_ts, "%Y-%m-%dT%H:%M:%SZ")
    epoch_sec = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
    assert epoch_sec == 1239155867
    assert dt.year == 2009
    assert dt.month == 4
    assert dt.day == 8


def test_malformed_row_detection():
    # Simulate rows with missing fields
    data = [
        ("user_001", "2009-01-01T00:00:00Z", "mbid_a", "Radiohead", "mbid_t", "Karma Police"),
        ("user_001", "2009-01-01T00:01:00Z", "", "Radiohead", "", ""),  # missing track
        ("", "2009-01-01T00:02:00Z", "", "Radiohead", "", "Creep"),  # missing user
        ("user_001", "INVALID_DATE", "", "Radiohead", "", "Creep"),  # invalid date
    ]
    df = pl.DataFrame(
        data,
        schema=["user_id", "raw_timestamp", "mbid_artist", "artist_name", "mbid_track", "track_name"],
        orient="row"
    )
    
    # Check valid filter logic
    valid = df.filter(
        (pl.col("user_id").str.strip_chars() != "") &
        (pl.col("artist_name").str.strip_chars() != "") &
        (pl.col("track_name").str.strip_chars() != "")
    )
    assert len(valid) == 2
