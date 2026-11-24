"""Catalogue and Metadata Foundation for EUTERPE.

Provides high-performance, validated music metadata mapping internal track integer IDs
to displayable titles, artists, MusicBrainz IDs, Last.fm URLs, popularity statistics,
novelty surprisal bits, and provenance guarantees.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import polars as pl


@dataclass(frozen=True)
class AudioFeatures:
    """Artist-level estimated acoustic characteristics calibrated for exploratory visualization.
    
    NOTE: These features are statistical proxy estimates conditioned on historical artist profiles,
    NOT measured audio signal processing from raw waveform recordings.
    """
    energy: float
    valence: float
    danceability: float
    acousticness: float
    tempo_bpm: int
    provenance_type: str = "artist_level_proxy_estimate"


@dataclass(frozen=True)
class TrackMetadata:
    """Complete metadata record for a single catalogue track."""
    track_id: int
    recording_key: str
    title: str
    artist: str
    artist_id: int
    mbid_track: Optional[str]
    mbid_artist: Optional[str]
    playcount: int
    novelty_bits: float
    is_head_decile: bool
    musicbrainz_url: Optional[str]
    lastfm_url: str
    cover_art_url: str
    energy: float
    valence: float
    danceability: float
    acousticness: float
    tempo_bpm: int
    acoustic_feature_source: str = "artist_level_proxy_estimate"
    temporal_cutoff_boundary: str = "2008-10-16 18:26:38 UTC"
    provenance: str = "Last.fm 1K + MusicBrainz Open Catalog"
    license: str = "Academic Research / CC BY-NC-SA 3.0"
    preview_status: str = "available_simulated_sonification"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrackMetadataStore:
    """High-performance in-memory and columnar metadata store."""

    def __init__(self, df: Optional[pl.DataFrame] = None):
        self._df = df
        self._by_id: Dict[int, TrackMetadata] = {}
        if df is not None:
            self._index_dataframe(df)

    def _index_dataframe(self, df: pl.DataFrame) -> None:
        """Build fast in-memory array and hash index for sub-millisecond retrieval."""
        for row in df.iter_rows(named=True):
            t = TrackMetadata(
                track_id=int(row["track_id"]),
                recording_key=row["recording_key"],
                title=row["title"],
                artist=row["artist"],
                artist_id=int(row["artist_id"]),
                mbid_track=row.get("mbid_track"),
                mbid_artist=row.get("mbid_artist"),
                playcount=int(row["playcount"]),
                novelty_bits=float(row["novelty_bits"]),
                is_head_decile=bool(row["is_head_decile"]),
                musicbrainz_url=row.get("musicbrainz_url"),
                lastfm_url=row["lastfm_url"],
                cover_art_url=row["cover_art_url"],
                energy=float(row["energy"]),
                valence=float(row["valence"]),
                danceability=float(row["danceability"]),
                acousticness=float(row["acousticness"]),
                tempo_bpm=int(row["tempo_bpm"]),
                provenance=row.get("provenance", "Last.fm 1K + MusicBrainz Open Catalog"),
                license=row.get("license", "Academic Research / CC BY-NC-SA 3.0"),
                preview_status=row.get("preview_status", "available_simulated"),
            )
            self._by_id[t.track_id] = t

    def get_track(self, track_id: int) -> Optional[TrackMetadata]:
        """Fetch metadata for a single track by internal integer ID."""
        return self._by_id.get(track_id)

    def get_tracks_batch(self, track_ids: List[int]) -> List[TrackMetadata]:
        """Batch fetch metadata for multiple track IDs."""
        result = []
        for tid in track_ids:
            item = self._by_id.get(tid)
            if item is not None:
                result.append(item)
        return result

    def search_tracks(self, query: str, limit: int = 10) -> List[TrackMetadata]:
        """Sub-string search over title and artist."""
        if not query or len(query.strip()) < 2:
            return []
        q = query.strip().lower()
        matches = []
        for meta in self._by_id.values():
            if q in meta.title.lower() or q in meta.artist.lower():
                matches.append(meta)
                if len(matches) >= limit:
                    break
        return matches

    def total_tracks(self) -> int:
        return len(self._by_id)

    @classmethod
    def load_parquet(cls, path: Path = Path("artifacts/catalog/enriched_tracks.parquet")) -> "TrackMetadataStore":
        """Load enriched catalogue from parquet."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Catalogue file not found at {path}")
        df = pl.read_parquet(path)
        return cls(df)

    def save_parquet(self, path: Path = Path("artifacts/catalog/enriched_tracks.parquet")) -> None:
        """Persist current metadata to parquet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._df is not None:
            self._df.write_parquet(path)
        else:
            records = [m.to_dict() for m in self._by_id.values()]
            pl.DataFrame(records).write_parquet(path)


_STORE_INSTANCE: Optional[TrackMetadataStore] = None


def get_metadata_store() -> TrackMetadataStore:
    """Singleton provider for TrackMetadataStore."""
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        catalog_path = Path("artifacts/catalog/enriched_tracks.parquet")
        if catalog_path.exists():
            _STORE_INSTANCE = TrackMetadataStore.load_parquet(catalog_path)
        else:
            raise RuntimeError("Catalogue not generated. Run scripts/build_catalog_metadata.py first.")
    return _STORE_INSTANCE
