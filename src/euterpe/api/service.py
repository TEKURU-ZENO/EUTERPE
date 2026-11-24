"""Discovery Service Engine for EUTERPE FastAPI Backend.

Coordinates in-memory retrieval, score calibration, multi-objective reranking,
explainability attributions, taste profile analysis, and universe projections.
"""

from functools import lru_cache
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import polars as pl
from scipy import sparse
import torch

from euterpe.catalog.metadata import TrackMetadata, TrackMetadataStore
from euterpe.discovery.explainability import ExplainabilityEngine, TrackExplanation
from euterpe.discovery.feedback import get_feedback_store
from euterpe.discovery.reranking import min_max_normalize


class DiscoveryService:
    """Core recommendation and analytics engine for EUTERPE API."""

    def __init__(self):
        self.catalog_store: Optional[TrackMetadataStore] = None
        self.als_user_factors: Optional[np.ndarray] = None
        self.als_item_factors: Optional[np.ndarray] = None
        self.tt_user_embeddings: Optional[np.ndarray] = None
        self.tt_item_embeddings: Optional[np.ndarray] = None
        self.universe_coords: Optional[np.ndarray] = None
        self.train_matrix: Optional[sparse.csr_matrix] = None
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.track_to_artist: Optional[np.ndarray] = None
        self.item_playcounts: Optional[np.ndarray] = None
        self.novelty_bits: Optional[np.ndarray] = None
        self.top_10pct_item_set: set = set()
        self.precomputed_recs: Dict[int, List[int]] = {}
        self.artist_names: Dict[int, str] = {}
        self._user_consumed_cache: Dict[int, set] = {}

    def initialize(self) -> None:
        """Load all model checkpoints, indices, and mappings once at startup."""
        print("[DiscoveryService] Initializing EUTERPE Engine...")
        t0 = time.time()

        # 1. Metadata store
        self.catalog_store = TrackMetadataStore.load_parquet(Path("artifacts/catalog/enriched_tracks.parquet"))

        # 2. Mappings
        with open("artifacts/mappings/user_to_idx.json", "r") as f:
            self.user_to_idx = json.load(f)
        self.idx_to_user = {v: k for k, v in self.user_to_idx.items()}

        with open("artifacts/mappings/item_to_idx.json", "r") as f:
            self.item_to_idx = json.load(f)
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}

        self.track_to_artist = np.load("artifacts/mappings/track_to_artist.npy")

        # 3. Train matrix and item frequencies
        self.train_matrix = sparse.load_npz("data/splits/train_playcount_matrix.npz")
        self.item_playcounts = np.asarray(self.train_matrix.sum(axis=0)).flatten()
        tot_plays = max(1.0, float(np.sum(self.item_playcounts)))
        probs = np.maximum(self.item_playcounts / tot_plays, 1e-12)
        self.novelty_bits = -np.log2(probs)
        
        n_items = len(self.item_to_idx)
        top10_idx = int(0.10 * n_items)
        self.top_10pct_item_set = set(np.argsort(-self.item_playcounts)[:top10_idx])

        # 4. Latent models
        als_data = np.load("experiments/phase_2/models/als_model.npz")
        self.als_user_factors = als_data["user_factors"]
        self.als_item_factors = als_data["item_factors"]

        self.tt_item_embeddings = np.load("experiments/phase_3/models/item_embeddings.npy")
        
        # Load precomputed Two-Tower user embeddings (instant sub-millisecond startup)
        user_emb_path = Path("experiments/phase_3/models/user_embeddings.npy")
        if user_emb_path.exists():
            self.tt_user_embeddings = np.load(user_emb_path)
        else:
            from euterpe.models.neural.two_tower import TwoTowerRecommender
            from euterpe.models.neural.training import build_training_sequences
            tt_model = TwoTowerRecommender.load_checkpoint("experiments/phase_3/models/neural_model_checkpoint.pt")
            _, _, _, _, eval_histories = build_training_sequences(
                self.user_to_idx,
                self.item_to_idx,
                self.track_to_artist,
                max_history_len=20,
            )
            with torch.no_grad():
                u_tensor = torch.arange(len(self.user_to_idx), dtype=torch.long)
                h_tensor = torch.from_numpy(eval_histories).long()
                self.tt_user_embeddings = tt_model.encode_users(u_tensor, history_items=h_tensor).cpu().numpy()
            np.save(user_emb_path, self.tt_user_embeddings)

        # 5. Universe 3D coordinates
        self.universe_coords = np.load("artifacts/catalog/universe_coordinates.npy")

        # 6. Precomputed frozen test recommendations
        precomputed_df = pl.read_parquet("experiments/phase_4/predictions/fused_reranked.parquet")
        for row in precomputed_df.iter_rows(named=True):
            self.precomputed_recs[int(row["user_idx"])] = list(row["recommended_items"])

        # 7. Artist names mapping
        for m in self.catalog_store._by_id.values():
            if m.artist_id not in self.artist_names:
                self.artist_names[m.artist_id] = m.artist

        t1 = time.time()
        print(f"[DiscoveryService] Initialized in {t1 - t0:.2f}s across {len(self.user_to_idx)} users and {n_items} tracks.")

    def resolve_user_idx(self, user_identifier: str) -> Optional[int]:
        """Convert username string or integer ID to internal integer index."""
        if user_identifier in self.user_to_idx:
            return self.user_to_idx[user_identifier]
        try:
            val = int(user_identifier)
            if 0 <= val < len(self.user_to_idx):
                return val
        except ValueError:
            pass
        return None

    def get_user_consumed_items(self, user_idx: int) -> set:
        """Fetch historical training consumed tracks for masking."""
        if user_idx in self._user_consumed_cache:
            return self._user_consumed_cache[user_idx]
        if self.train_matrix is None:
            return set()
        consumed = set(self.train_matrix.indices[self.train_matrix.indptr[user_idx] : self.train_matrix.indptr[user_idx + 1]])
        self._user_consumed_cache[user_idx] = consumed
        return consumed

    def get_popular_fallback(self, limit: int = 20, max_per_artist: int = 2) -> List[int]:
        """High-quality popular discovery fallback for cold-start or anonymous users."""
        sorted_indices = np.argsort(-self.item_playcounts)
        recs = []
        artist_counts = {}
        for idx in sorted_indices:
            art = self.track_to_artist[idx]
            if artist_counts.get(art, 0) < max_per_artist:
                recs.append(int(idx))
                artist_counts[art] = artist_counts.get(art, 0) + 1
                if len(recs) == limit:
                    break
        return recs

    def compute_discovery_recommendations(
        self,
        user_idx: int,
        alpha: float = 0.70,
        beta: float = 0.20,
        max_per_artist: int = 2,
        limit: int = 20,
        offset: int = 0,
        pool_size: int = 100,
        enable_feedback: bool = True,
        session_id: Optional[str] = None,
    ) -> Tuple[List[int], List[TrackExplanation], str]:
        """Execute full real-time hybrid retrieval, fusion, novelty bonus, and bounded artist repetition."""
        consumed = set(self.get_user_consumed_items(user_idx))

        # Closed-loop feedback integration (P7.4) with optional session/condition isolation
        user_id_str = self.idx_to_user.get(user_idx)
        positive_tracks = set()
        if enable_feedback and user_id_str:
            feedback_store = get_feedback_store()
            neg_tracks = feedback_store.get_user_negative_tracks(user_id_str, session_id=session_id)
            if neg_tracks:
                consumed = consumed | neg_tracks
            positive_tracks = feedback_store.get_user_positive_tracks(user_id_str, session_id=session_id)

        # 1. ALS candidates
        u_als = self.als_user_factors[user_idx]
        s_als = np.dot(self.als_item_factors, u_als)
        if consumed:
            s_als[list(consumed)] = -np.inf
        top_als_idx = np.argpartition(-s_als, pool_size)[:pool_size]
        top_als_idx = top_als_idx[np.argsort(-s_als[top_als_idx])]

        # 2. Two-Tower candidates (MIPS inner product)
        u_tt = self.tt_user_embeddings[user_idx]
        s_tt = np.dot(self.tt_item_embeddings, u_tt)
        if consumed:
            s_tt[list(consumed)] = -np.inf
        top_tt_idx = np.argpartition(-s_tt, pool_size)[:pool_size]
        top_tt_idx = top_tt_idx[np.argsort(-s_tt[top_tt_idx])]

        # 3. Deduplicated candidate union
        candidate_pool = list(dict.fromkeys(list(top_als_idx) + list(top_tt_idx)))

        # 4. Standardize candidate scores
        als_cands = s_als[candidate_pool]
        tt_cands = s_tt[candidate_pool]

        als_std = (als_cands - np.mean(als_cands)) / (np.std(als_cands) + 1e-8)
        tt_std = (tt_cands - np.mean(tt_cands)) / (np.std(tt_cands) + 1e-8)

        # 5. Linear fusion + Novelty bonus + Feedback adaptation
        fused_relevance = alpha * als_std + (1.0 - alpha) * tt_std
        novelty_bonus = beta * self.novelty_bits[candidate_pool]
        final_scores = fused_relevance + novelty_bonus

        # Apply gentle positive feedback affinity bonus if user provided positive feedback
        if positive_tracks and self.tt_item_embeddings is not None:
            valid_pos = [pid for pid in positive_tracks if 0 <= pid < len(self.tt_item_embeddings)]
            if valid_pos:
                pos_embs = self.tt_item_embeddings[valid_pos[-5:]]
                cand_embs = self.tt_item_embeddings[candidate_pool]
                sims = np.dot(cand_embs, pos_embs.T)
                max_sim = np.max(sims, axis=1)
                gamma = 0.15
                norm_sim = (max_sim - np.mean(max_sim)) / (np.std(max_sim) + 1e-8)
                final_scores = final_scores + gamma * norm_sim

        # Store raw scores map for explainability
        als_scores_map = {candidate_pool[i]: float(als_std[i]) for i in range(len(candidate_pool))}
        tt_scores_map = {candidate_pool[i]: float(tt_std[i]) for i in range(len(candidate_pool))}

        # 6. Ranked selection with bounded artist repetition
        ranked_indices = np.argsort(-final_scores)
        selected_tracks = []
        artist_counts = {}
        artist_cap_events = {}

        for idx in ranked_indices:
            track_id = candidate_pool[idx]
            art_id = int(self.track_to_artist[track_id])
            if artist_counts.get(art_id, 0) < max_per_artist:
                selected_tracks.append(track_id)
                artist_counts[art_id] = artist_counts.get(art_id, 0) + 1
            else:
                art_name = self.artist_names.get(art_id, f"Artist #{art_id}")
                artist_cap_events[track_id] = art_name

            if len(selected_tracks) >= (offset + limit):
                break

        final_slate = selected_tracks[offset : offset + limit]

        # 7. Generate Explainability Attributions
        explanations = ExplainabilityEngine.generate_explanations(
            recommended_tracks=final_slate,
            als_scores=als_scores_map,
            tt_scores=tt_scores_map,
            novelty_bits=self.novelty_bits,
            track_to_artist=self.track_to_artist,
            alpha=alpha,
            beta=beta,
            artist_cap_events=artist_cap_events,
            artist_names=self.artist_names,
        )

        return final_slate, explanations, "realtime_hybrid_inference"

    def get_hidden_gems(
        self,
        user_idx: Optional[int] = None,
        min_novelty_bits: float = 16.0,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve personalized high-novelty, low-popularity tracks."""
        if user_idx is not None and 0 <= user_idx < len(self.user_to_idx):
            # Personalized candidate pool filtered for high novelty
            consumed = self.get_user_consumed_items(user_idx)
            u_als = self.als_user_factors[user_idx]
            s_als = np.dot(self.als_item_factors, u_als)
            if consumed:
                s_als[list(consumed)] = -np.inf
            # Search top 500 candidates
            top_candidates = np.argpartition(-s_als, 500)[:500]
            top_candidates = top_candidates[np.argsort(-s_als[top_candidates])]
            
            gems = []
            seen_artists = {}
            for t in top_candidates:
                n_bits = float(self.novelty_bits[t])
                if n_bits >= min_novelty_bits:
                    art = self.track_to_artist[t]
                    if seen_artists.get(art, 0) < 2:
                        meta = self.catalog_store.get_track(int(t))
                        if meta:
                            gems.append({
                                **meta.to_dict(),
                                "relevance_score": round(float(s_als[t]), 4),
                                "gem_tag": f"Long-Tail Rarity ({n_bits:.1f} bits)",
                            })
                            seen_artists[art] = seen_artists.get(art, 0) + 1
                            if len(gems) == limit:
                                break
            if gems:
                return gems

        # Global high-quality hidden gems fallback
        rare_indices = np.where(self.novelty_bits >= min_novelty_bits)[0]
        # Sort by moderate playcount (at least 5 plays in training)
        qual_mask = (self.item_playcounts[rare_indices] >= 5)
        qual_indices = rare_indices[qual_mask]
        selected_rare = np.random.choice(qual_indices, size=min(limit, len(qual_indices)), replace=False)
        
        gems = []
        for tid in selected_rare:
            meta = self.catalog_store.get_track(int(tid))
            if meta:
                gems.append({
                    **meta.to_dict(),
                    "relevance_score": 1.0,
                    "gem_tag": f"Catalogue Discovery ({meta.novelty_bits:.1f} bits)",
                })
        return gems

    def get_universe_viewport(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        limit: int = 250,
        highlight_user_idx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return 2D/3D projection points filtered by viewport bounding box."""
        coords = self.universe_coords
        xs = coords[:, 0]
        ys = coords[:, 1]
        zs = coords[:, 2]

        mask = np.ones(len(coords), dtype=bool)
        if x_min is not None:
            mask &= (xs >= x_min)
        if x_max is not None:
            mask &= (xs <= x_max)
        if y_min is not None:
            mask &= (ys >= y_min)
        if y_max is not None:
            mask &= (ys <= y_max)

        valid_indices = np.where(mask)[0]
        if len(valid_indices) > limit:
            # Subsample evenly across popularity spectrum
            selected = np.random.choice(valid_indices, size=limit, replace=False)
        else:
            selected = valid_indices

        # Check recommended tracks for highlighting
        rec_set = set()
        if highlight_user_idx is not None and highlight_user_idx in self.precomputed_recs:
            rec_set = set(self.precomputed_recs[highlight_user_idx])

        points = []
        for tid in selected:
            meta = self.catalog_store.get_track(int(tid))
            if meta:
                points.append({
                    "track_id": int(tid),
                    "title": meta.title,
                    "artist": meta.artist,
                    "x": round(float(xs[tid]), 4),
                    "y": round(float(ys[tid]), 4),
                    "z": round(float(zs[tid]), 4),
                    "energy": meta.energy,
                    "valence": meta.valence,
                    "novelty_bits": meta.novelty_bits,
                    "is_head_decile": meta.is_head_decile,
                    "is_recommended": int(tid) in rec_set,
                    "cover_art_url": meta.cover_art_url,
                })
        return points

    def get_taste_profile(self, user_idx: int) -> Dict[str, Any]:
        """Compute transparent listening metrics and acoustic radar for listener."""
        consumed_items = list(self.get_user_consumed_items(user_idx))
        if not consumed_items:
            return {
                "user_id": self.idx_to_user.get(user_idx, f"user_{user_idx}"),
                "total_listens": 0,
                "unique_tracks": 0,
                "unique_artists": 0,
                "openness_score": 50.0,
                "mainstream_index": 50.0,
                "artist_diversity_score": 50.0,
                "acoustic_radar": {"energy": 0.5, "valence": 0.5, "danceability": 0.5, "acousticness": 0.5},
                "top_artists": [],
                "history_sample": [],
            }

        user_nov = self.novelty_bits[consumed_items]
        mean_nov = float(np.mean(user_nov))
        # Openness score: 10 bits is 0%, 20 bits is 100%
        openness = float(np.clip((mean_nov - 10.0) / 10.0 * 100.0, 0.0, 100.0))

        # Mainstream index: % of plays in top-10% head
        n_head = sum(1 for it in consumed_items if it in self.top_10pct_item_set)
        mainstream = float((n_head / max(1, len(consumed_items))) * 100.0)

        # Artist counts and Shannon diversity
        artists = [self.track_to_artist[it] for it in consumed_items]
        unique_artists = len(set(artists))
        art_counts = {}
        for a in artists:
            art_counts[a] = art_counts.get(a, 0) + 1
        
        art_probs = np.array(list(art_counts.values())) / float(len(artists))
        shannon_entropy = -float(np.sum(art_probs * np.log2(art_probs + 1e-12)))
        # Normalize entropy: log2(100) = 6.64 bits max
        diversity_score = float(np.clip((shannon_entropy / 6.64) * 100.0, 0.0, 100.0))

        # Top 5 artists by frequency
        top_art_ids = sorted(art_counts.keys(), key=lambda a: art_counts[a], reverse=True)[:5]
        top_artists = [
            {"artist_id": int(aid), "artist_name": self.artist_names.get(aid, f"Artist #{aid}"), "plays": art_counts[aid]}
            for aid in top_art_ids
        ]

        # Acoustic radar averages
        metas = [self.catalog_store.get_track(it) for it in consumed_items[:50]]
        metas = [m for m in metas if m is not None]
        avg_energy = float(np.mean([m.energy for m in metas])) if metas else 0.5
        avg_valence = float(np.mean([m.valence for m in metas])) if metas else 0.5
        avg_dance = float(np.mean([m.danceability for m in metas])) if metas else 0.5
        avg_acoustic = float(np.mean([m.acousticness for m in metas])) if metas else 0.5

        # Sample history
        history_sample = [m.to_dict() for m in metas[:8]]

        return {
            "user_id": self.idx_to_user.get(user_idx, f"user_{user_idx}"),
            "user_idx": user_idx,
            "total_listens": len(consumed_items),
            "unique_tracks": len(consumed_items),
            "unique_artists": unique_artists,
            "openness_score": round(openness, 1),
            "mainstream_index": round(mainstream, 1),
            "artist_diversity_score": round(diversity_score, 1),
            "mean_novelty_bits": round(mean_nov, 2),
            "acoustic_radar": {
                "energy": round(avg_energy, 2),
                "valence": round(avg_valence, 2),
                "danceability": round(avg_dance, 2),
                "acousticness": round(avg_acoustic, 2),
            },
            "top_artists": top_artists,
            "history_sample": history_sample,
        }


_SERVICE_INSTANCE: Optional[DiscoveryService] = None


def get_discovery_service() -> DiscoveryService:
    """Singleton getter for DiscoveryService."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = DiscoveryService()
        _SERVICE_INSTANCE.initialize()
    return _SERVICE_INSTANCE
