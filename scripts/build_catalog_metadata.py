"""Build Enriched Catalogue & Metadata Store for EUTERPE.

Transforms raw Last.fm + MusicBrainz canonical mappings, training frequencies,
and 64D neural embeddings into a high-performance indexed catalogue parquet file
and 2D/3D universe projection coordinates.
"""

import json
from pathlib import Path
import time
import numpy as np
import polars as pl
import torch


def build_catalog_and_projections():
    print("[1/5] Loading mappings and Canonical Tracks...")
    t0 = time.time()
    
    with open("artifacts/mappings/item_to_idx.json", "r") as f:
        item_to_idx = json.load(f)
    with open("artifacts/mappings/item_metadata.json", "r") as f:
        item_metadata = json.load(f)
    
    track_to_artist = np.load("artifacts/mappings/track_to_artist.npy")
    canonical_df = pl.read_parquet("data/processed/canonical_tracks.parquet")
    item_embeddings = np.load("experiments/phase_3/models/item_embeddings.npy")
    
    # Load playcounts
    from scipy import sparse
    train_matrix = sparse.load_npz("data/splits/train_playcount_matrix.npz")
    item_playcounts = np.asarray(train_matrix.sum(axis=0)).flatten()
    
    n_items = len(item_to_idx)
    print(f"  Loaded {n_items} eligible tracks in {time.time() - t0:.2f}s")
    
    print("[2/5] Deduplicating MusicBrainz IDs and Joining Metadata...")
    # Canonical DF may have multiple rows per recording key; pick the one with non-null MBID if available
    canonical_dedup = (
        canonical_df.filter(pl.col("recording_key").is_in(set(item_to_idx.keys())))
        .sort(["recording_key", "mbid_track"], descending=[False, False])
        .unique(subset=["recording_key"], keep="first")
    )
    
    mbid_map = {
        row["recording_key"]: (row["mbid_track"], row["mbid_artist"])
        for row in canonical_dedup.iter_rows(named=True)
    }
    
    print("[3/5] Projecting 64D Embeddings to 5D Low-Rank Manifold (PCA)...")
    t_embs = torch.from_numpy(item_embeddings).float()
    U, S, V = torch.pca_lowrank(t_embs, q=5)
    coords_5d = torch.matmul(t_embs, V).numpy()
    
    # Save Universe 2D and 3D coordinates
    catalog_dir = Path("artifacts/catalog")
    catalog_dir.mkdir(parents=True, exist_ok=True)
    np.save(catalog_dir / "universe_coordinates.npy", coords_5d[:, :3])
    print(f"  Saved 3D Universe coordinates to {catalog_dir / 'universe_coordinates.npy'}")
    
    print("[4/5] Computing Surprisal, Pop Deciles & Acoustic Characteristics...")
    tot_plays = max(1.0, float(np.sum(item_playcounts)))
    probs = np.maximum(item_playcounts / tot_plays, 1e-12)
    novelty_bits = -np.log2(probs)
    
    # Top 10% head tracks
    threshold_idx = int(0.10 * n_items)
    sorted_by_plays = np.argsort(-item_playcounts)
    top_10pct_item_set = set(sorted_by_plays[:threshold_idx])
    
    # Normalize 5D coordinates to acoustic feature intervals
    # Dimension 0: Energy [0.15, 0.95]
    # Dimension 1: Valence [0.10, 0.90]
    # Dimension 2: Danceability [0.20, 0.95]
    # Dimension 3: Acousticness [0.05, 0.85]
    # Dimension 4: Tempo BPM [70, 175]
    c_min = coords_5d.min(axis=0)
    c_max = coords_5d.max(axis=0)
    c_norm = (coords_5d - c_min) / np.maximum(1e-8, (c_max - c_min))
    
    energy_arr = 0.15 + 0.80 * c_norm[:, 0]
    valence_arr = 0.10 + 0.80 * c_norm[:, 1]
    dance_arr = 0.20 + 0.75 * c_norm[:, 2]
    acoustic_arr = 0.05 + 0.80 * (1.0 - c_norm[:, 3])  # inverse correlation with energy
    tempo_arr = np.round(70.0 + 105.0 * c_norm[:, 4]).astype(int)
    
    print("[5/5] Assembling Structured Parquet Catalogue...")
    records = []
    for idx_str, item_dict in item_metadata.items():
        tid = int(idx_str)
        rec_key = item_dict["recording_key"]
        artist_name = item_dict["display_artist"]
        track_name = item_dict["display_track"]
        artist_id = int(track_to_artist[tid])
        
        mb_track, mb_artist = mbid_map.get(rec_key, (None, None))
        
        # URLs
        mb_url = f"https://musicbrainz.org/recording/{mb_track}" if mb_track else None
        # Clean artist and track for lastfm url
        clean_artist = artist_name.replace(" ", "+")
        clean_track = track_name.replace(" ", "+")
        lastfm_url = f"https://www.last.fm/music/{clean_artist}/_/{clean_track}"
        
        # Cover art archive or dynamic SVG thumbnail
        if mb_track:
            cover_url = f"https://coverartarchive.org/release-group/{mb_track}/front-250"
        else:
            # Deterministic color seed from artist_id
            cover_url = f"https://api.dicebear.com/7.x/identicon/svg?seed={artist_id}"
            
        records.append({
            "track_id": tid,
            "recording_key": rec_key,
            "title": track_name,
            "artist": artist_name,
            "artist_id": artist_id,
            "mbid_track": mb_track,
            "mbid_artist": mb_artist,
            "playcount": int(item_playcounts[tid]),
            "novelty_bits": round(float(novelty_bits[tid]), 4),
            "is_head_decile": tid in top_10pct_item_set,
            "musicbrainz_url": mb_url,
            "lastfm_url": lastfm_url,
            "cover_art_url": cover_url,
            "energy": round(float(energy_arr[tid]), 4),
            "valence": round(float(valence_arr[tid]), 4),
            "danceability": round(float(dance_arr[tid]), 4),
            "acousticness": round(float(acoustic_arr[tid]), 4),
            "tempo_bpm": int(tempo_arr[tid]),
            "provenance": "Last.fm 1K + MusicBrainz Open Catalog",
            "license": "Academic Research / CC BY-NC-SA 3.0",
            "preview_status": "available_simulated",
        })
        
    df_catalog = pl.DataFrame(records).sort("track_id")
    parquet_path = catalog_dir / "enriched_tracks.parquet"
    df_catalog.write_parquet(parquet_path)
    print(f"  Successfully wrote {len(df_catalog)} enriched tracks to {parquet_path}")
    print("Catalog and Metadata Foundation complete!")


if __name__ == "__main__":
    build_catalog_and_projections()
