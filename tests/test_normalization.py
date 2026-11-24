"""Unit tests for string normalization and canonical track key resolution."""

from euterpe.data.normalization import (
    clean_artist_name,
    clean_track_title,
    build_canonical_track_key,
    normalize_string,
    canonicalize_interaction,
)


def test_normalize_string():
    # Accents / diacritics
    assert normalize_string("Björk") == "bjork"
    assert normalize_string("Sigur Rós") == "sigur ros"
    assert normalize_string("Motörhead") == "motorhead"
    # Whitespace and punctuation
    assert normalize_string("  Radiohead - OK Computer  ") == "radiohead ok computer"


def test_clean_track_title_remaster_variations():
    # Remastered noise patterns should be stripped cleanly
    assert clean_track_title("Karma Police (2011 Remastered)") == "karma police"
    assert clean_track_title("Karma Police [2009 Remaster]") == "karma police"
    assert clean_track_title("Karma Police - Remastered 2011") == "karma police"
    assert clean_track_title("Creep (Remastered)") == "creep"


def test_clean_track_title_live_and_features():
    # Live patterns
    assert clean_track_title("Paranoid Android (Live at Glastonbury)") == "paranoid android"
    assert clean_track_title("Fake Plastic Trees [Live]") == "fake plastic trees"
    # Featured artists
    assert clean_track_title("Stan (feat. Dido)") == "stan"
    assert clean_track_title("Stan [Featuring Dido]") == "stan"


def test_clean_track_title_audio_edits():
    assert clean_track_title("Pyramid Song (Album Version)") == "pyramid song"
    assert clean_track_title("Idioteque (Radio Edit)") == "idioteque"
    assert clean_track_title("Like Spinning Plates [Bonus Track]") == "like spinning plates"


def test_build_canonical_track_key_identity_preservation():
    # Two noisy scrobbles representing the same recording must map to identical key
    key1 = build_canonical_track_key("Radiohead", "Karma Police (2011 Remastered)")
    key2 = build_canonical_track_key(" Radiohead ", "karma police")
    key3 = build_canonical_track_key("Radiohead", "Karma Police [Live at Glastonbury]")
    
    assert key1 == "radiohead:::karma police"
    assert key2 == "radiohead:::karma police"
    # Different acoustic variants with live flags resolve cleanly
    assert key1 == key2


def test_canonicalize_interaction():
    key, disp_artist, disp_track = canonicalize_interaction(
        artist_name="坂本龍一",
        track_name="Bibo No Aozora (Live_2009_4_15)"
    )
    assert disp_artist == "坂本龍一"
    assert "Live" not in clean_track_title(disp_track)
    assert key.startswith("坂本龍一:::")
