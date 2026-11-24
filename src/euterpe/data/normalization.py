"""Data normalization and canonical entity resolution for EUTERPE.

Ensures tracks and artists are identified consistently without false splits
(e.g., splitting "Song (Live)" vs "Song") or false merges across distinct recordings.
"""

import re
import unicodedata
from typing import Tuple


# Regex patterns to detect and strip non-canonical metadata suffixes
_NOISE_PATTERNS = [
    # Remastered indicators: (Remastered 2011), [2009 Remaster], (2015 Remaster)
    r"\s*[\(\[]\s*(\d{4}\s+)?remaster(ed)?(\s+\d{4})?\s*[\)\]]",
    # Live indicators: (Live at ...), [Live 2008], (Live)
    r"\s*[\(\[]\s*live(\s+[^)\]]*)?\s*[\)\]]",
    # Featured artists: (feat. Drake), [Featuring Eminem]
    r"\s*[\(\[]\s*feat(\.|uring)?\s+[^)\]]+[\)\]]",
    # Audio versions: (Album Version), (Radio Edit), [Bonus Track], (Deluxe Edition)
    r"\s*[\(\[]\s*(album\s+version|radio\s+edit|bonus\s+track|deluxe(\s+edition)?|original\s+mix|instrumental)\s*[\)\]]",
    # Trailing hyphens / dashes often appended to track names in Last.fm
    r"\s*-\s*(remaster(ed)?|live|radio\s+edit|bonus\s+track).*$",
]

_COMPILED_PATTERNS = [re.compile(p, flags=re.IGNORECASE) for p in _NOISE_PATTERNS]


def normalize_string(text: str) -> str:
    """Normalize text using Unicode NFKD, lowercasing, and whitespace collapsing.
    
    Args:
        text: Raw input string.
        
    Returns:
        Normalized clean string.
    """
    if not text:
        return ""
    
    # Unicode NFKD normalization (decomposes accents, e.g., é -> e + accent)
    text = unicodedata.normalize("NFKD", text)
    # Lowercase
    text = text.lower()
    # Strip non-ASCII diacritics while preserving characters
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Replace non-alphanumeric (except standard space) with whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_track_title(title: str) -> str:
    """Clean track title by removing noise patterns (remastered, live, feat).
    
    Args:
        title: Raw track title.
        
    Returns:
        Cleaned canonical title.
    """
    if not title:
        return ""
    
    cleaned = title.strip()
    for pattern in _COMPILED_PATTERNS:
        cleaned = pattern.sub("", cleaned).strip()
    
    return normalize_string(cleaned)


def clean_artist_name(artist: str) -> str:
    """Clean artist name by normalizing unicode and stripping surrounding artifacts.
    
    Args:
        artist: Raw artist name.
        
    Returns:
        Cleaned canonical artist name.
    """
    if not artist:
        return ""
    
    return normalize_string(artist)


def build_canonical_track_key(artist: str, track: str) -> str:
    """Construct a composite canonical track key to prevent metadata collisions.
    
    Args:
        artist: Raw or partially cleaned artist name.
        track: Raw or partially cleaned track title.
        
    Returns:
        Deterministic canonical key in the format: "normalized_artist:::normalized_track".
    """
    norm_artist = clean_artist_name(artist)
    norm_track = clean_track_title(track)
    return f"{norm_artist}:::{norm_track}"


def canonicalize_interaction(
    artist_name: str,
    track_name: str,
    raw_mbid_artist: str = "",
    raw_mbid_track: str = ""
) -> Tuple[str, str, str]:
    """Process interaction metadata and return canonical keys and display labels.
    
    Returns:
        Tuple of (canonical_key, display_artist, display_track)
    """
    key = build_canonical_track_key(artist_name, track_name)
    display_artist = artist_name.strip() if artist_name else "Unknown Artist"
    display_track = track_name.strip() if track_name else "Unknown Track"
    return key, display_artist, display_track
