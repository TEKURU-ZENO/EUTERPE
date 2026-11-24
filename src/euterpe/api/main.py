"""FastAPI Production Discovery Service for EUTERPE.

Exposes 5 core interactive discovery endpoints:
1. /api/discover: Hybrid fused personalized discovery with explainability attributions.
2. /api/hidden-gems: Long-tail curated discovery stream with rarity safeguards.
3. /api/universe: Precomputed 2D/3D latent space projections and viewport query.
4. /api/taste-profile: Transparent listener openness, mainstream, and acoustic metrics.
5. /api/lab: Interactive algorithmic sandbox for real-time reranking & trade-off simulation.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from euterpe.api.service import DiscoveryService, get_discovery_service
from euterpe.catalog.metadata import TrackMetadata
from euterpe.catalog.previews import get_preview_resolver
from euterpe.discovery.feedback import (
    FeedbackAction,
    FeedbackEvent,
    FamiliarityLevel,
    PreferenceRating,
    ImpressionEvent,
    QualitativeFeedbackEvent,
    get_feedback_store,
    get_impression_store,
)


class FeedbackSubmitRequest(BaseModel):
    user_id: str
    track_id: int
    action: str
    familiarity: Optional[str] = None
    preference: Optional[str] = None
    slate_rank: Optional[int] = None
    session_id: Optional[str] = None
    impression_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ImpressionItemModel(BaseModel):
    track_id: int
    slate_rank: int
    model_version: Optional[str] = "euterpe-v1.0.0"
    exposure_condition: Optional[str] = "default"
    has_audio_preview: Optional[bool] = False
    preview_url: Optional[str] = None
    preview_source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchImpressionRequest(BaseModel):
    participant_id: str
    session_id: str
    items: List[ImpressionItemModel]


class AudioPlaybackRecordRequest(BaseModel):
    impression_id: str
    started: bool = True
    completed: bool = False
    replay_increment: int = 0
    play_duration_sec: float = 0.0


class QualitativeFeedbackRequest(BaseModel):
    participant_id: str
    session_id: str
    exposure_condition: str
    model_version: Optional[str] = "unknown"
    question: str
    response_text: str


def get_app_service() -> DiscoveryService:
    """Safely obtain initialized DiscoveryService singleton."""
    if not hasattr(app.state, "service") or app.state.service is None:
        app.state.service = get_discovery_service()
    return app.state.service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager loading models and indices at startup."""
    print("=" * 60)
    print("Starting EUTERPE Discovery Intelligence Service...")
    app.state.service = get_discovery_service()
    print("EUTERPE Service Ready for Production Requests.")
    print("=" * 60)
    yield
    print("Shutting down EUTERPE Service...")


app = FastAPI(
    title="EUTERPE Discovery Intelligence API",
    description="High-performance recommendation, discovery science, and latent-space API for music discovery.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for interactive React UI frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    """System health check and loaded catalog inventory."""
    service: DiscoveryService = get_app_service()
    return {
        "status": "healthy",
        "service": "EUTERPE Discovery Intelligence",
        "version": "1.0.0",
        "eligible_tracks": service.catalog_store.total_tracks(),
        "registered_users": len(service.user_to_idx),
        "precomputed_test_users": len(service.precomputed_recs),
        "timestamp": time.time(),
    }


@app.get("/api/users")
def list_sample_users(limit: int = 25):
    """List sample supported listener accounts for UI selection."""
    service: DiscoveryService = get_app_service()
    sample_users = []
    
    # Stratified sample: sparse, medium, heavy users
    all_users = sorted(service.precomputed_recs.keys())[:limit]
    for uid_idx in all_users:
        u_str = service.idx_to_user[uid_idx]
        consumed = len(service.get_user_consumed_items(uid_idx))
        sample_users.append({
            "user_id": u_str,
            "user_idx": uid_idx,
            "training_scrobbles": consumed,
            "cohort": "Sparse (Q1)" if consumed <= 3625 else ("Medium (Q2-Q3)" if consumed <= 23823 else "Heavy (Q4)"),
        })
    return {"users": sample_users}


@app.get("/api/discover")
def discover_recommendations(
    user_id: str = "user_000001",
    limit: int = 20,
    offset: int = 0,
    alpha: float = 0.70,
    beta: float = 0.20,
    max_per_artist: int = 2,
    enable_feedback: bool = True,
    session_id: Optional[str] = None,
):
    """Personalized hybrid discovery recommendations with traceable mathematical attributions and audio previews."""
    if not (0.0 <= alpha <= 1.0):
        raise HTTPException(status_code=400, detail="alpha parameter must be bounded in [0.0, 1.0]")
    if not (0.0 <= beta <= 1.0):
        raise HTTPException(status_code=400, detail="beta parameter must be bounded in [0.0, 1.0]")
    if not (1 <= max_per_artist <= 20):
        raise HTTPException(status_code=400, detail="max_per_artist must be between 1 and 20")
    if not (1 <= limit <= 100):
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")

    t0 = time.perf_counter()
    service: DiscoveryService = get_app_service()
    preview_resolver = get_preview_resolver()

    uid_idx = service.resolve_user_idx(user_id)
    if uid_idx is None:
        # Graceful cold-start fallback
        fallback_tracks = service.get_popular_fallback(limit=limit, max_per_artist=max_per_artist)
        metas = []
        for tid in fallback_tracks:
            m = service.catalog_store.get_track(tid)
            if m:
                m_dict = m.to_dict()
                p_url, p_src = preview_resolver.resolve_preview(tid, m.artist, m.title)
                m_dict["preview_url"] = p_url
                m_dict["preview_source"] = p_src
                m_dict["has_audio_preview"] = bool(p_url)
                metas.append(m_dict)
        t1 = time.perf_counter()
        return {
            "user_id": user_id,
            "resolved_user_idx": None,
            "inference_mode": "cold_start_fallback",
            "total_items": len(metas),
            "latency_ms": round((t1 - t0) * 1000, 2),
            "recommendations": metas,
            "explanations": [],
        }

    # Execute Hybrid Inference
    tracks, explanations, mode = service.compute_discovery_recommendations(
        user_idx=uid_idx,
        alpha=alpha,
        beta=beta,
        max_per_artist=max_per_artist,
        limit=limit,
        offset=offset,
        enable_feedback=enable_feedback,
        session_id=session_id,
    )

    enriched_recs = []
    for rank_i, tid in enumerate(tracks):
        meta = service.catalog_store.get_track(tid)
        if meta:
            meta_dict = meta.to_dict()
            if rank_i < len(explanations):
                meta_dict["attribution"] = explanations[rank_i].to_dict()
            p_url, p_src = preview_resolver.resolve_preview(tid, meta.artist, meta.title)
            meta_dict["preview_url"] = p_url
            meta_dict["preview_source"] = p_src
            meta_dict["has_audio_preview"] = bool(p_url)
            enriched_recs.append(meta_dict)

    # Compute slate summary metrics
    nov_vals = [r["novelty_bits"] for r in enriched_recs]
    mean_nov = float(np.mean(nov_vals)) if nov_vals else 0.0
    unique_artists = len(set(r["artist_id"] for r in enriched_recs))
    head_share = (sum(1 for r in enriched_recs if r["is_head_decile"]) / max(1, len(enriched_recs))) * 100.0

    t1 = time.perf_counter()
    return {
        "user_id": user_id,
        "resolved_user_idx": uid_idx,
        "inference_mode": mode,
        "total_items": len(enriched_recs),
        "latency_ms": round((t1 - t0) * 1000, 2),
        "parameters": {
            "alpha": alpha,
            "beta": beta,
            "max_per_artist": max_per_artist,
        },
        "slate_metrics": {
            "mean_novelty_bits": round(mean_nov, 2),
            "unique_artists": unique_artists,
            "top_10pct_share": round(head_share, 1),
        },
        "recommendations": enriched_recs,
    }


@app.get("/api/hidden-gems")
def get_hidden_gems(
    user_id: Optional[str] = None,
    min_novelty_bits: float = 16.0,
    limit: int = 20,
):
    """Retrieve long-tail, high-surprisal tracks protected by quality and relevance safeguards."""
    t0 = time.perf_counter()
    service: DiscoveryService = get_app_service()

    uid_idx = service.resolve_user_idx(user_id) if user_id else None
    gems = service.get_hidden_gems(user_idx=uid_idx, min_novelty_bits=min_novelty_bits, limit=limit)
    t1 = time.perf_counter()

    return {
        "user_id": user_id,
        "min_novelty_bits": min_novelty_bits,
        "total_gems": len(gems),
        "latency_ms": round((t1 - t0) * 1000, 2),
        "gems": gems,
    }


@app.get("/api/universe")
def get_music_universe(
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    limit: int = 300,
    user_id: Optional[str] = None,
):
    """Retrieve precomputed 2D/3D latent coordinates for interactive vector space exploration."""
    t0 = time.perf_counter()
    service: DiscoveryService = get_app_service()

    uid_idx = service.resolve_user_idx(user_id) if user_id else None
    points = service.get_universe_viewport(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        limit=limit,
        highlight_user_idx=uid_idx,
    )
    t1 = time.perf_counter()

    return {
        "viewport": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "total_points": len(points),
        "highlighted_user": user_id,
        "latency_ms": round((t1 - t0) * 1000, 2),
        "points": points,
    }


@app.get("/api/taste-profile")
def get_taste_profile(user_id: str = "user_000001"):
    """Compute transparent listening metrics and acoustic radar for listener."""
    t0 = time.perf_counter()
    service: DiscoveryService = get_app_service()

    uid_idx = service.resolve_user_idx(user_id)
    if uid_idx is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    profile = service.get_taste_profile(uid_idx)
    t1 = time.perf_counter()
    profile["latency_ms"] = round((t1 - t0) * 1000, 2)
    return profile


@app.get("/api/lab")
def discovery_lab_rerank(
    user_id: str = "user_000001",
    alpha: float = 0.70,
    beta: float = 0.20,
    max_per_artist: int = 2,
    limit: int = 20,
):
    """Interactive algorithmic sandbox returning live reranked items with delta trade-offs."""
    if not (0.0 <= alpha <= 1.0):
        raise HTTPException(status_code=400, detail="alpha parameter must be bounded in [0.0, 1.0]")
    if not (0.0 <= beta <= 1.0):
        raise HTTPException(status_code=400, detail="beta parameter must be bounded in [0.0, 1.0]")
    if not (1 <= max_per_artist <= 20):
        raise HTTPException(status_code=400, detail="max_per_artist must be between 1 and 20")
    if not (1 <= limit <= 100):
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")

    t0 = time.perf_counter()
    service: DiscoveryService = get_app_service()

    uid_idx = service.resolve_user_idx(user_id)
    if uid_idx is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    # 1. Baseline ALS recommendations (alpha=1.0, beta=0.0, max_per_artist=20)
    baseline_tracks, _, _ = service.compute_discovery_recommendations(
        user_idx=uid_idx, alpha=1.0, beta=0.0, max_per_artist=20, limit=limit
    )

    # 2. Experimental Discovery recommendations
    exp_tracks, explanations, _ = service.compute_discovery_recommendations(
        user_idx=uid_idx, alpha=alpha, beta=beta, max_per_artist=max_per_artist, limit=limit
    )

    # Calculate live deltas
    base_nov = np.mean([service.novelty_bits[t] for t in baseline_tracks])
    exp_nov = np.mean([service.novelty_bits[t] for t in exp_tracks])
    base_art_count = len(set(service.track_to_artist[t] for t in baseline_tracks))
    exp_art_count = len(set(service.track_to_artist[t] for t in exp_tracks))

    enriched_exp = []
    for rank_i, tid in enumerate(exp_tracks):
        meta = service.catalog_store.get_track(tid)
        if meta:
            meta_dict = meta.to_dict()
            if rank_i < len(explanations):
                meta_dict["attribution"] = explanations[rank_i].to_dict()
            enriched_exp.append(meta_dict)

    t1 = time.perf_counter()
    return {
        "user_id": user_id,
        "latency_ms": round((t1 - t0) * 1000, 2),
        "controls": {
            "alpha": alpha,
            "beta": beta,
            "max_per_artist": max_per_artist,
        },
        "live_deltas": {
            "novelty_bits": round(float(exp_nov), 2),
            "delta_novelty": round(float(exp_nov - base_nov), 2),
            "unique_artists": exp_art_count,
            "delta_artists": exp_art_count - base_art_count,
            "artist_cap_interventions": sum(1 for e in explanations if e.artist_cap_triggered),
        },
        "recommendations": enriched_exp,
    }


@app.post("/api/feedback")
def submit_discovery_feedback(req: FeedbackSubmitRequest):
    """Record listener interaction & discovery feedback signal."""
    try:
        act = FeedbackAction(req.action)
    except ValueError:
        valid_actions = [a.value for a in FeedbackAction]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{req.action}'. Must be one of: {valid_actions}",
        )

    fam = None
    if req.familiarity:
        try:
            fam = FamiliarityLevel(req.familiarity)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid familiarity '{req.familiarity}'")

    pref = None
    if req.preference:
        try:
            pref = PreferenceRating(req.preference)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid preference '{req.preference}'")

    store = get_feedback_store()
    event = store.record_feedback(
        user_id=req.user_id,
        track_id=req.track_id,
        action=act,
        familiarity=fam,
        preference=pref,
        slate_rank=req.slate_rank,
        session_id=req.session_id,
        impression_id=req.impression_id,
        metadata=req.metadata,
    )
    return {"status": "success", "event": event.to_dict()}


@app.delete("/api/feedback/user/{user_id}")
def delete_user_feedback_history(user_id: str, session_id: Optional[str] = None):
    """Clear all feedback history for a specific listener, or within a specific condition session."""
    store = get_feedback_store()
    if session_id:
        removed = store.clear_user_session_feedback(user_id, session_id)
    else:
        removed = store.clear_user_feedback(user_id)
    return {"status": "success", "user_id": user_id, "session_id": session_id, "removed_events": removed}


@app.get("/api/feedback/summary")
def get_feedback_summary():
    """Aggregated discovery feedback metrics across all recorded sessions."""
    store = get_feedback_store()
    return store.get_summary_metrics()


@app.get("/api/feedback/user/{user_id}")
def get_user_feedback_history(user_id: str):
    """Retrieve full feedback history for a specific listener."""
    store = get_feedback_store()
    events = store.get_user_feedback(user_id)
    return {"user_id": user_id, "events": [e.to_dict() for e in events]}


@app.post("/api/feedback/qualitative")
def submit_qualitative_feedback(req: QualitativeFeedbackRequest):
    """Record qualitative post-slate open-ended response."""
    store = get_impression_store()
    qev = store.record_qualitative_feedback(
        participant_id=req.participant_id,
        session_id=req.session_id,
        exposure_condition=req.exposure_condition,
        model_version=req.model_version or "unknown",
        question=req.question,
        response_text=req.response_text,
    )
    return {"status": "success", "event": qev.to_dict()}


@app.get("/api/feedback/qualitative")
def list_qualitative_feedback(participant_id: Optional[str] = None):
    """Retrieve recorded qualitative responses."""
    store = get_impression_store()
    events = store.get_qualitative_feedback(participant_id=participant_id)
    return {"total": len(events), "responses": [e.to_dict() for e in events]}


@app.post("/api/impressions")
def record_batch_impressions(req: BatchImpressionRequest):
    """Record impressions for a recommendation slate presented to a participant."""
    store = get_impression_store()
    recorded = []
    for item in req.items:
        imp = store.record_impression(
            participant_id=req.participant_id,
            session_id=req.session_id,
            track_id=item.track_id,
            slate_rank=item.slate_rank,
            model_version=item.model_version or "euterpe-v1.0.0",
            exposure_condition=item.exposure_condition or "default",
            has_audio_preview=bool(item.has_audio_preview),
            preview_url=item.preview_url,
            preview_source=item.preview_source,
            metadata=item.metadata,
        )
        recorded.append(imp.to_dict())
    return {
        "status": "success",
        "participant_id": req.participant_id,
        "session_id": req.session_id,
        "recorded_impressions": len(recorded),
        "impressions": recorded,
    }


@app.post("/api/impressions/audio")
def record_audio_telemetry(req: AudioPlaybackRecordRequest):
    """Record audio playback telemetry event on an impression."""
    store = get_impression_store()
    imp = store.record_audio_playback(
        impression_id=req.impression_id,
        started=req.started,
        completed=req.completed,
        replay_increment=req.replay_increment,
        play_duration_sec=req.play_duration_sec,
    )
    if not imp:
        raise HTTPException(status_code=404, detail=f"Impression '{req.impression_id}' not found.")
    return {"status": "success", "impression": imp.to_dict()}


@app.get("/api/impressions/summary")
def get_impressions_summary():
    """Retrieve aggregated impression metrics using unique impressions as denominator."""
    store = get_impression_store()
    return store.get_summary_metrics()


@app.get("/api/impressions/participant/{participant_id}")
def get_participant_impressions(participant_id: str):
    """Retrieve impression logs for a specific study participant."""
    store = get_impression_store()
    impressions = store.get_impressions_by_participant(participant_id)
    return {"participant_id": participant_id, "impressions": [imp.to_dict() for imp in impressions]}


@app.get("/api/metadata/track/{track_id}")
def get_track_detail(track_id: int):
    """Retrieve full metadata for a single track."""
    service: DiscoveryService = get_app_service()
    meta = service.catalog_store.get_track(track_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Track ID {track_id} not found.")
    return meta.to_dict()


# Mount production UI distribution if built
UI_DIST_PATH = Path(__file__).resolve().parents[3] / "ui" / "dist"
if UI_DIST_PATH.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIST_PATH), html=True), name="ui")
