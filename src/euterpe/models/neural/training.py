"""Training, ablation, and validation engine for EUTERPE Two-Tower recommendation architecture.

Implements Workstream 3.3 & 3.4:
- In-batch sampled softmax loss with popularity log-Q correction.
- Strict chronological user history generation (target never in history).
- False negative masking on batch interaction matrices.
- Controlled ablation suite (N0 through N5).
- Comprehensive validation ranking metrics (Recall@20, NDCG@20, MRR@20, Novelty@20, Coverage, Gini).
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from scipy import sparse
import torch
import torch.nn as nn
from tqdm import tqdm

from euterpe.discovery.metrics import (
    artist_coverage,
    capped_recall_at_k,
    catalog_coverage,
    gini_coefficient,
    mrr_at_k,
    ndcg_at_k,
    novelty_at_k,
    paired_bootstrap_ci,
    recall_at_k,
)
from euterpe.models.neural.two_tower import TwoTowerRecommender
from euterpe.models.neural.negative_sampling import (
    InBatchSampledSoftmaxLoss,
    compute_popularity_log_q,
)
from euterpe.models.neural.retrieval import VectorRetrievalEngine


def load_dataset_artifacts():
    """Load mappings, item metadata, track-to-artist array, and matrices."""
    with open("artifacts/mappings/user_to_idx.json", "r") as f:
        user_to_idx = json.load(f)
    with open("artifacts/mappings/item_to_idx.json", "r") as f:
        item_to_idx = json.load(f)
    with open("artifacts/mappings/item_metadata.json", "r") as f:
        item_metadata = json.load(f)

    track_to_artist = np.load("artifacts/mappings/track_to_artist.npy")
    train_matrix = sparse.load_npz("data/splits/train_playcount_matrix.npz")
    val_matrix = sparse.load_npz("data/splits/validation_matrix.npz")
    test_matrix = sparse.load_npz("data/splits/test_matrix.npz")

    # Item frequencies in training matrix for log-Q calculation
    item_playcounts = np.asarray(train_matrix.sum(axis=0)).flatten()
    log_q_table = torch.tensor(
        compute_popularity_log_q(item_playcounts, power=0.75),
        dtype=torch.float32,
    )

    # Item artist array
    artist_list = [item_metadata[str(i)]["display_artist"] for i in range(len(item_metadata))]

    # Build consumed set per user
    user_consumed_sets = {}
    for u in range(train_matrix.shape[0]):
        user_consumed_sets[u] = set(
            train_matrix.indices[train_matrix.indptr[u] : train_matrix.indptr[u + 1]]
        )

    return {
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "item_metadata": item_metadata,
        "track_to_artist": track_to_artist,
        "train_matrix": train_matrix,
        "val_matrix": val_matrix,
        "test_matrix": test_matrix,
        "item_playcounts": item_playcounts,
        "log_q_table": log_q_table,
        "artist_list": artist_list,
        "user_consumed_sets": user_consumed_sets,
        "n_users": len(user_to_idx),
        "n_items": len(item_to_idx),
        "n_artists": int(track_to_artist.max()) + 1,
    }


def build_training_sequences(
    user_to_idx: Dict[str, int],
    item_to_idx: Dict[str, int],
    track_to_artist: np.ndarray,
    max_history_len: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build chronological training instances: (user_id, item_id, artist_id, history_seq).
    
    Invariant: Target track NEVER appears in history. History strictly contains items listened
    to chronologically prior to target track.
    Returns:
        user_ids: (N,)
        track_ids: (N,)
        artist_ids: (N,)
        histories: (N, max_history_len)  # 1-indexed, 0 is pad
        eval_histories: (n_users, max_history_len) # User's full training history for evaluation
    """
    df = pl.read_parquet(
        "data/splits/train.parquet",
        columns=["user_id", "recording_key", "epoch_seconds"],
    )

    # Group by (user, item) to get first listen timestamp
    df_first = (
        df.group_by(["user_id", "recording_key"])
        .agg(pl.col("epoch_seconds").min().alias("first_listen"))
        .sort(["user_id", "first_listen"])
    )

    df_mapped = df_first.with_columns([
        pl.col("user_id").replace_strict(user_to_idx, default=None).cast(pl.Int32).alias("user_idx"),
        pl.col("recording_key").replace_strict(item_to_idx, default=None).cast(pl.Int32).alias("item_idx"),
    ])

    user_arr = df_mapped["user_idx"].to_numpy()
    item_arr = df_mapped["item_idx"].to_numpy()

    # Pre-allocate arrays
    N = len(user_arr)
    n_users = len(user_to_idx)
    histories = np.zeros((N, max_history_len), dtype=np.int32)
    eval_histories = np.zeros((n_users, max_history_len), dtype=np.int32)
    artist_ids = track_to_artist[item_arr]

    # Populate sequences per user
    user_splits = np.split(
        np.arange(N),
        np.where(np.diff(user_arr) != 0)[0] + 1,
    )

    for split in user_splits:
        u = user_arr[split[0]]
        u_items = item_arr[split]
        m = len(u_items)

        for i in range(m):
            # Target item is u_items[i]
            # History is previous items up to max_history_len (1-indexed)
            start_idx = max(0, i - max_history_len)
            prev_items = u_items[start_idx:i] + 1  # 1-indexed
            if len(prev_items) > 0:
                histories[split[i], : len(prev_items)] = prev_items

        # Evaluation history: last items of user's training history
        start_eval = max(0, m - max_history_len)
        eval_prev = u_items[start_eval:m] + 1
        if len(eval_prev) > 0:
            eval_histories[u, : len(eval_prev)] = eval_prev

    return user_arr, item_arr, artist_ids, histories, eval_histories


def evaluate_two_tower_model(
    model: TwoTowerRecommender,
    eval_histories: np.ndarray,
    target_matrix: sparse.csr_matrix,
    train_matrix: sparse.csr_matrix,
    item_playcounts: np.ndarray,
    artist_list: List[str],
    track_to_artist: np.ndarray,
    k: int = 20,
    batch_size: int = 128,
) -> Dict[str, float]:
    """Evaluate Two-Tower model on target split (validation or test) with training consumed-item masking."""
    model.eval()
    n_users = target_matrix.shape[0]
    n_items = model.n_items

    # 1. Precompute all item embeddings
    with torch.no_grad():
        all_tracks = torch.arange(n_items, dtype=torch.long)
        all_artists = torch.from_numpy(track_to_artist).long()
        
        # Batch encode items
        item_embs = []
        for start in range(0, n_items, 4096):
            end = min(start + 4096, n_items)
            t_batch = all_tracks[start:end]
            a_batch = all_artists[start:end]
            i_repr = model.encode_items(t_batch, artist_ids=a_batch)
            item_embs.append(i_repr.cpu().numpy())
        item_embeddings = np.concatenate(item_embs, axis=0)  # (n_items, d)

    # 2. Build FAISS index for candidate retrieval
    engine = VectorRetrievalEngine(dimension=model.embedding_dim)
    engine.build_index(item_embeddings)

    # 3. Encode users
    with torch.no_grad():
        user_tensor = torch.arange(n_users, dtype=torch.long)
        history_tensor = torch.from_numpy(eval_histories).long()
        user_embeddings = model.encode_users(user_tensor, history_items=history_tensor).cpu().numpy()

    # 4. Generate recommendations for warm users in target split
    warm_users = [u for u in range(n_users) if target_matrix.indptr[u + 1] > target_matrix.indptr[u]]
    recommendations = {}

    for u in warm_users:
        consumed = set(train_matrix.indices[train_matrix.indptr[u] : train_matrix.indptr[u + 1]])
        u_recs = engine.query_user(user_embeddings[u], consumed_items=consumed, k=k, candidate_pool_size=150)
        recommendations[u] = [item_id for item_id, _ in u_recs]

    # 5. Compute metrics across all evaluated warm users
    recall_list = []
    capped_recall_list = []
    ndcg_list = []
    mrr_list = []
    novelty_list = []
    all_recs_list = []

    rec_counts = np.zeros(n_items, dtype=np.int32)
    tot_plays = max(1.0, float(np.sum(item_playcounts)))
    item_probabilities = {i: item_playcounts[i] / tot_plays for i in range(n_items)}
    item_to_artist = {i: int(track_to_artist[i]) for i in range(n_items)}
    total_artists = len(set(track_to_artist))

    threshold_idx = int(0.10 * n_items)
    sorted_by_plays = np.argsort(-item_playcounts)
    top_10pct_item_set = set(sorted_by_plays[:threshold_idx])

    for u in warm_users:
        actual = set(target_matrix.indices[target_matrix.indptr[u] : target_matrix.indptr[u + 1]])
        preds = recommendations.get(u, [])[:k]
        all_recs_list.append(preds)

        for it in preds:
            rec_counts[it] += 1

        recall_list.append(recall_at_k(actual, preds, k=k))
        capped_recall_list.append(capped_recall_at_k(actual, preds, k=k))
        ndcg_list.append(ndcg_at_k(actual, preds, k=k))
        mrr_list.append(mrr_at_k(actual, preds, k=k))
        novelty_list.append(novelty_at_k(preds, item_probabilities, k=k))

    cat_cov = catalog_coverage(all_recs_list, n_items) * 100.0
    art_cov = artist_coverage(all_recs_list, item_to_artist, total_artists) * 100.0
    gini = gini_coefficient(rec_counts)

    total_rec_slots = float(len(warm_users) * k)
    aggregate_top10_exposure = float(np.sum([rec_counts[it] for it in top_10pct_item_set]) / total_rec_slots) * 100.0

    return {
        "recall_at_20_conv": float(np.mean(recall_list)),
        "recall_at_20_capped": float(np.mean(capped_recall_list)),
        "ndcg_at_20": float(np.mean(ndcg_list)),
        "mrr_at_20": float(np.mean(mrr_list)),
        "novelty_at_20": float(np.mean(novelty_list)),
        "catalog_coverage": float(cat_cov),
        "artist_coverage": float(art_cov),
        "exposure_gini": float(gini),
        "top_10_decile_share": float(aggregate_top10_exposure),
        "n_evaluated_users": len(warm_users),
    }


def train_two_tower_epoch(
    model: TwoTowerRecommender,
    user_arr: np.ndarray,
    item_arr: np.ndarray,
    artist_arr: np.ndarray,
    histories: np.ndarray,
    criterion: InBatchSampledSoftmaxLoss,
    optimizer_sparse: torch.optim.Optimizer,
    optimizer_dense: torch.optim.Optimizer,
    log_q_table: Optional[torch.Tensor] = None,
    train_csr_matrix: Optional[sparse.csr_matrix] = None,
    batch_size: int = 1024,
    extra_negatives: int = 0,
    n_items: int = 443375,
    track_to_artist: Optional[np.ndarray] = None,
) -> float:
    """Train model for one epoch."""
    model.train()
    N = len(user_arr)
    indices = np.random.permutation(N)

    total_loss = 0.0
    n_batches = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_idx = indices[start:end]
        B = len(batch_idx)

        u_batch = torch.from_numpy(user_arr[batch_idx]).long()
        t_batch = torch.from_numpy(item_arr[batch_idx]).long()
        a_batch = torch.from_numpy(artist_arr[batch_idx]).long()
        h_batch = torch.from_numpy(histories[batch_idx]).long()

        u_repr, i_repr = model(u_batch, t_batch, artist_ids=a_batch, history_items=h_batch)

        # Optional uniform extra negative sampling (N5)
        if extra_negatives > 0 and track_to_artist is not None:
            extra_ids = np.random.randint(0, n_items, size=extra_negatives)
            extra_t = torch.from_numpy(extra_ids).long()
            extra_a = torch.from_numpy(track_to_artist[extra_ids]).long()
            extra_repr = model.encode_items(extra_t, artist_ids=extra_a)
            extra_ids_t = extra_t
        else:
            extra_repr = None
            extra_ids_t = None

        loss = criterion(
            user_repr=u_repr,
            pos_item_repr=i_repr,
            user_ids=u_batch,
            pos_item_ids=t_batch,
            log_q_table=log_q_table,
            extra_neg_repr=extra_repr,
            extra_neg_ids=extra_ids_t,
            train_csr_matrix=train_csr_matrix,
        )

        loss.backward()
        optimizer_sparse.step()
        optimizer_dense.step()
        optimizer_sparse.zero_grad()
        optimizer_dense.zero_grad()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)
