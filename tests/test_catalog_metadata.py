"""Unit tests for EUTERPE Catalogue and Metadata Foundation."""

from pathlib import Path
import pytest
from euterpe.catalog.metadata import TrackMetadata, TrackMetadataStore


def test_track_metadata_store():
    store = TrackMetadataStore.load_parquet(Path("artifacts/catalog/enriched_tracks.parquet"))
    assert store.total_tracks() == 443375
    
    # Test valid track
    t = store.get_track(0)
    assert t is not None
    assert isinstance(t.title, str)
    assert isinstance(t.artist, str)
    assert 0.0 <= t.energy <= 1.0
    assert 0.0 <= t.valence <= 1.0
    assert 50 <= t.tempo_bpm <= 200
    assert t.provenance == "Last.fm 1K + MusicBrainz Open Catalog"
    
    # Test non-existent track
    assert store.get_track(9999999) is None
    
    # Test search
    res = store.search_tracks("Pink Floyd", limit=5)
    assert len(res) > 0
    assert any("pink floyd" in r.artist.lower() or "pink floyd" in r.title.lower() for r in res)
