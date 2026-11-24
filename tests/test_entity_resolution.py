"""Unit tests for entity resolution and dual identity mappings."""

from euterpe.data.normalize import (
    build_recording_key,
    build_normalized_track_key,
    resolve_track_identities,
    clean_artist_name,
)


def test_dual_identities_live_distinction():
    # Senior Scientific Review 03:
    # Live versions must be preserved as distinct recording_keys, but grouped under normalized_track_key
    studio = "Radiohead", "Creep"
    live = "Radiohead", "Creep (Live at Reading)"
    
    rec_studio = build_recording_key(*studio)
    rec_live = build_recording_key(*live)
    
    track_studio = build_normalized_track_key(*studio)
    track_live = build_normalized_track_key(*live)
    
    # recording_key must preserve live distinction
    assert rec_studio != rec_live
    assert "live" in rec_live
    
    # normalized_track_key must collapse them for broad exploratory discovery
    assert track_studio == track_live
    assert track_studio == "radiohead:::creep"


def test_dual_identities_remaster_distinction():
    # Remastered cuts
    original = "Pink Floyd", "Comfortably Numb"
    remaster = "Pink Floyd", "Comfortably Numb (2011 Remastered)"
    
    rec_orig = build_recording_key(*original)
    rec_rem = build_recording_key(*remaster)
    
    norm_orig = build_normalized_track_key(*original)
    norm_rem = build_normalized_track_key(*remaster)
    
    # Conservative preserves remaster tag
    assert rec_orig != rec_rem
    # Broad collapses
    assert norm_orig == norm_rem
    assert norm_orig == "pink floyd:::comfortably numb"


def test_resolve_track_identities_tuple():
    rec_key, norm_key, disp_art, disp_trk = resolve_track_identities(
        artist="Sigur Rós",
        track="Hoppípolla (Radio Edit)"
    )
    assert rec_key == "sigur ros:::hoppipolla (radio edit)"
    assert norm_key == "sigur ros:::hoppipolla"
    assert disp_art == "Sigur Rós"
    assert disp_trk == "Hoppípolla (Radio Edit)"
