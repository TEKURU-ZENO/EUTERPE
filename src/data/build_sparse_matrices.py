"""Sparse matrix construction and index mapping module for EUTERPE.

Workstream 1.6:
Constructs scipy.sparse.csr_matrix interaction representations for train, validation,
and test sets. Generates explicit bidirectional index mappings and persists
metrics on matrix density, sparsity, and memory footprint.
"""

import json
import time
from pathlib import Path
import numpy as np
import polars as pl
from scipy import sparse
from euterpe.config import default_config

BASE_DIR = Path(__file__).resolve().parents[2]
SPLITS_DIR = BASE_DIR / "data" / "splits"
MAPPINGS_DIR = BASE_DIR / "artifacts" / "mappings"
MATRIX_REPORT = BASE_DIR / "reports" / "phase_1" / "sparse_matrix_report.md"


def build_sparse_matrices_and_mappings(
    splits_dir: Path,
    mappings_dir: Path,
    alpha: float = 40.0
) -> dict:
    """Build CSR sparse matrices and persistent index mappings."""
    print("Beginning sparse matrix construction and mapping serialization...")
    t0 = time.time()
    mappings_dir.mkdir(parents=True, exist_ok=True)
    MATRIX_REPORT.parent.mkdir(parents=True, exist_ok=True)
    
    # Read training set to define the universe of users and items
    train_path = splits_dir / "train.parquet"
    val_path = splits_dir / "validation.parquet"
    test_path = splits_dir / "test.parquet"
    
    print(f"Loading training split from {train_path}...")
    df_train = pl.read_parquet(train_path)
    
    # Aggregate train interactions: user, track -> count
    train_agg = df_train.group_by(["user_id", "recording_key"]).agg([
        pl.len().alias("play_count"),
        pl.col("artist_name").first().alias("display_artist"),
        pl.col("track_name").first().alias("display_track"),
        pl.col("normalized_track_key").first().alias("normalized_track_key")
    ])
    
    # Build deterministic index mappings based on training items and users
    unique_users = sorted(train_agg["user_id"].unique().to_list())
    unique_items = sorted(train_agg["recording_key"].unique().to_list())
    
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    print(f"Index Mapping Universe: {n_users:,} Users | {n_items:,} Items")
    
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    idx_to_user = {i: u for i, u in enumerate(unique_users)}
    item_to_idx = {it: i for i, it in enumerate(unique_items)}
    idx_to_item = {i: it for i, it in enumerate(unique_items)}
    
    # Save index mappings
    print(f"Persisting mappings to {mappings_dir}...")
    with open(mappings_dir / "user_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(user_to_idx, f)
    with open(mappings_dir / "idx_to_user.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_user, f)
    with open(mappings_dir / "item_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(item_to_idx, f)
    with open(mappings_dir / "idx_to_item.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_item, f)
        
    # Build item metadata lookup table
    item_meta_df = train_agg.unique(subset=["recording_key"]).select([
        "recording_key", "display_artist", "display_track", "normalized_track_key"
    ])
    item_metadata = {}
    for row in item_meta_df.iter_rows(named=True):
        it_idx = item_to_idx[row["recording_key"]]
        item_metadata[it_idx] = {
            "recording_key": row["recording_key"],
            "normalized_track_key": row["normalized_track_key"],
            "display_artist": row["display_artist"],
            "display_track": row["display_track"]
        }
    with open(mappings_dir / "item_metadata.json", "w", encoding="utf-8") as f:
        json.dump(item_metadata, f)
        
    # Build Training Matrices:
    # 1. Raw play counts r_ui
    train_users = np.array([user_to_idx[u] for u in train_agg["user_id"]], dtype=np.int32)
    train_items = np.array([item_to_idx[it] for it in train_agg["recording_key"]], dtype=np.int32)
    train_counts = train_agg["play_count"].to_numpy().astype(np.float32)
    
    train_playcount_matrix = sparse.csr_matrix(
        (train_counts, (train_users, train_items)),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    sparse.save_npz(splits_dir / "train_playcount_matrix.npz", train_playcount_matrix)
    
    # 2. Log play counts: ln(1 + r_ui)
    train_log_counts = np.log1p(train_counts)
    train_logcount_matrix = sparse.csr_matrix(
        (train_log_counts, (train_users, train_items)),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    sparse.save_npz(splits_dir / "train_logcount_matrix.npz", train_logcount_matrix)
    
    # 3. Precomputed confidence: c_ui = 1 + alpha * ln(1 + r_ui)
    train_confs = 1.0 + alpha * train_log_counts
    train_confidence_matrix = sparse.csr_matrix(
        (train_confs, (train_users, train_items)),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    train_matrix = train_confidence_matrix
    sparse.save_npz(splits_dir / "train_matrix.npz", train_confidence_matrix)
    sparse.save_npz(splits_dir / "train_confidence_matrix.npz", train_confidence_matrix)
    
    # 4. Binary preference matrix
    train_binary_matrix = sparse.csr_matrix(
        (np.ones_like(train_counts), (train_users, train_items)),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    sparse.save_npz(splits_dir / "train_binary_matrix.npz", train_binary_matrix)
    
    # Build Validation Matrix
    df_val = pl.read_parquet(val_path)
    val_agg = df_val.group_by(["user_id", "recording_key"]).len().rename({"len": "play_count"})
    val_users = np.array([user_to_idx[u] for u in val_agg["user_id"] if u in user_to_idx and val_agg["recording_key"][0] in item_to_idx], dtype=np.int32)
    
    val_rows, val_cols, val_data = [], [], []
    for row in val_agg.iter_rows(named=True):
        u, it, c = row["user_id"], row["recording_key"], row["play_count"]
        if u in user_to_idx and it in item_to_idx:
            val_rows.append(user_to_idx[u])
            val_cols.append(item_to_idx[it])
            val_data.append(c)
            
    val_matrix = sparse.csr_matrix(
        (np.array(val_data, dtype=np.float32), (np.array(val_rows, dtype=np.int32), np.array(val_cols, dtype=np.int32))),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    sparse.save_npz(splits_dir / "validation_matrix.npz", val_matrix)
    
    # Build Test Matrix
    df_test = pl.read_parquet(test_path)
    test_agg = df_test.group_by(["user_id", "recording_key"]).len().rename({"len": "play_count"})
    test_rows, test_cols, test_data = [], [], []
    for row in test_agg.iter_rows(named=True):
        u, it, c = row["user_id"], row["recording_key"], row["play_count"]
        if u in user_to_idx and it in item_to_idx:
            test_rows.append(user_to_idx[u])
            test_cols.append(item_to_idx[it])
            test_data.append(c)
            
    test_matrix = sparse.csr_matrix(
        (np.array(test_data, dtype=np.float32), (np.array(test_rows, dtype=np.int32), np.array(test_cols, dtype=np.int32))),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    sparse.save_npz(splits_dir / "test_matrix.npz", test_matrix)
    
    # Compute Density and Sparsity Statistics
    total_cells = float(n_users * n_items)
    train_nnz = train_matrix.nnz
    val_nnz = val_matrix.nnz
    test_nnz = test_matrix.nnz
    
    train_density = (train_nnz / total_cells) * 100
    train_sparsity = 100.0 - train_density
    
    val_density = (val_nnz / total_cells) * 100
    test_density = (test_nnz / total_cells) * 100
    
    mem_train_mb = (train_matrix.data.nbytes + train_matrix.indices.nbytes + train_matrix.indptr.nbytes) / (1024 * 1024)
    
    elapsed = time.time() - t0
    
    report = f"""# EUTERPE — Sparse Interaction Matrix Report
**Phase:** 1 — Data Engineering & Listening Intelligence  
**Matrix Architecture:** Compressed Sparse Row (`scipy.sparse.csr_matrix`)  

---

## 1. Matrix Topology & Dimensionality

| Parameter | Value | Interpretation |
|---|:---:|---|
| **Users ($|\\mathcal{{U}}|$)** | **{n_users:,}** | Eligible training listeners |
| **Items ($|\\mathcal{{I}}|$)** | **{n_items:,}** | Eligible training track recordings |
| **Total Possible Matrix Cells** | **{int(total_cells):,}** | $|\\mathcal{{U}}| \\times |\\mathcal{{I}}|$ interaction space |
| **Confidence Weighting** | $c_{{ui}} = 1 + {alpha} \\cdot \\ln(1 + r_{{ui}})$ | Hu, Koren & Volinsky (2008) formulation |

---

## 2. Partition Sparsity & Density Analysis

| Matrix Split | Non-Zero Entries (`nnz`) | Matrix Density (%) | Matrix Sparsity (%) | In-Memory Footprint |
|---|:---:|:---:|:---:|:---:|
| **Train Matrix (`train_matrix.npz`)** | **{train_nnz:,}** | **{train_density:.4f}%** | **{train_sparsity:.4f}%** | **{mem_train_mb:.2f} MB** |
| **Validation (`validation_matrix.npz`)** | **{val_nnz:,}** | **{val_density:.4f}%** | **{100.0 - val_density:.4f}%** | ~{(val_matrix.data.nbytes + val_matrix.indices.nbytes)/(1024*1024):.2f} MB |
| **Test Matrix (`test_matrix.npz`)** | **{test_nnz:,}** | **{test_density:.4f}%** | **{100.0 - test_density:.4f}%** | ~{(test_matrix.data.nbytes + test_matrix.indices.nbytes)/(1024*1024):.2f} MB |

---

## 3. Engineering Assessment & Index Mapping Integrity
- All user and track indices strictly adhere to 0-indexed contiguous integers.
- Exact bidirectional lookups (`user_to_idx.json`, `item_to_idx.json`, `item_metadata.json`) are preserved in `artifacts/mappings/` for sub-millisecond API retrieval and model serving.
"""
    with open(MATRIX_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"Sparse matrix report written: {MATRIX_REPORT}")
    return {
        "n_users": n_users,
        "n_items": n_items,
        "train_nnz": train_nnz,
        "val_nnz": val_nnz,
        "test_nnz": test_nnz,
        "train_density_pct": train_density,
        "train_sparsity_pct": train_sparsity,
        "memory_mb": round(mem_train_mb, 2),
        "elapsed_seconds": round(elapsed, 2)
    }


if __name__ == "__main__":
    build_sparse_matrices_and_mappings(SPLITS_DIR, MAPPINGS_DIR)
