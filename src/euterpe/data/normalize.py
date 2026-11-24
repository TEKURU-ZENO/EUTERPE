"""Track and artist normalization module for EUTERPE.

Implements conservative artist & title cleaning and generates dual identities:
1. `recording_key`: Conservative identity preserving live, acoustic, remix, remaster, and features.
2. `normalized_track_key`: Broader title/artist identity for exploratory grouping.
"""

import re
import unicodedata
from typing import Tuple


# Regex patterns for broader track title grouping (stripping acoustic suffixes)
_BROAD_STRIP_PATTERNS = [
    # Remastered indicators
    r"\s*[\(\[]\s*(\d{4}\s+)?remaster(ed)?(\s+\d{4})?\s*[\)\]]",
    # Live indicators
    r"\s*[\(\[]\s*live(\s+[^)\]]*)?\s*[\)\]]",
    # Featured artists
    r"\s*[\(\[]\s*feat(\.|uring)?\s+[^)\]]+[\)\]]",
    # Audio versions / bonus
    r"\s*[\(\[]\s*(album\s+version|radio\s+edit|bonus\s+track|deluxe(\s+edition)?|original\s+mix|instrumental)\s*[\)\]]",
    # Trailing hyphens
    r"\s*-\s*(remaster(ed)?|live|radio\s+edit|bonus\s+track).*$",
]
_COMPILED_BROAD_PATTERNS = [re.compile(p, flags=re.IGNORECASE) for p in _BROAD_STRIP_PATTERNS]


def normalize_text_conservative(text: str) -> str:
    """Conservative normalization: lowercases, decomposes accents, collapses spaces,
    but PRESERVES acoustic qualifiers (live, remix, acoustic, remaster).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.lower()
    # Strip combining diacritical marks
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Replace disruptive punctuation with space, but keep alphanumerics
    text = re.sub(r"[^\w\s\(\)\[\]\-]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_broad(text: str) -> str:
    """Broad normalization: strips punctuation, diacritics, and noise suffixes."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.lower()
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_artist_name(artist: str) -> str:
    """Normalize artist name consistently."""
    return normalize_text_broad(artist)


def build_recording_key(artist: str, track: str) -> str:
    """Conservative identity preserving meaningful recording distinctions.
    
    Example:
        'Radiohead', 'Karma Police (Live at Glastonbury)' -> 'radiohead:::karma police (live at glastonbury)'
        'Radiohead', 'Karma Police' -> 'radiohead:::karma police'
    """
    norm_artist = normalize_text_broad(artist)
    norm_track = normalize_text_conservative(track)
    return f"{norm_artist}:::{norm_track}"


def build_normalized_track_key(artist: str, track: str) -> str:
    """Broader identity for exploratory grouping and candidate aggregation.
    
    Example:
        'Radiohead', 'Karma Police (Live at Glastonbury)' -> 'radiohead:::karma police'
        'Radiohead', 'Karma Police (2011 Remaster)' -> 'radiohead:::karma police'
    """
    norm_artist = normalize_text_broad(artist)
    
    # Strip noise patterns
    cleaned = str(track).strip() if track else ""
    for pat in _COMPILED_BROAD_PATTERNS:
        cleaned = pat.sub("", cleaned).strip()
        
    norm_track = normalize_text_broad(cleaned)
    return f"{norm_artist}:::{norm_track}"


def resolve_track_identities(artist: str, track: str) -> Tuple[str, str, str, str]:
    """Generate both recording_key and normalized_track_key along with display labels.
    
    Returns:
        (recording_key, normalized_track_key, display_artist, display_track)
    """
    rec_key = build_recording_key(artist, track)
    track_key = build_normalized_track_key(artist, track)
    disp_artist = str(artist).strip() if artist else "Unknown Artist"
    disp_track = str(track).strip() if track else "Unknown Track"
    return rec_key, track_key, disp_artist, disp_track
