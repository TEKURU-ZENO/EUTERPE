"""Comprehensive Automated Integration Tests for EUTERPE FastAPI Discovery Endpoints."""

import pytest
from euterpe.api.main import (
    app,
    health_check,
    list_sample_users,
    discover_recommendations,
    get_hidden_gems,
    get_music_universe,
    get_taste_profile,
    discovery_lab_rerank,
    get_track_detail,
    submit_discovery_feedback,
    get_feedback_summary,
    get_user_feedback_history,
    delete_user_feedback_history,
    record_batch_impressions,
    get_impressions_summary,
    get_participant_impressions,
    record_audio_telemetry,
    submit_qualitative_feedback,
    list_qualitative_feedback,
    FeedbackSubmitRequest,
    BatchImpressionRequest,
    ImpressionItemModel,
    AudioPlaybackRecordRequest,
    QualitativeFeedbackRequest,
    UI_DIST_PATH,
)
from fastapi import HTTPException
from euterpe.api.service import get_discovery_service


@pytest.fixture(scope="session", autouse=True)
def init_service():
    """Ensure service is initialized once for the entire test session."""
    service = get_discovery_service()
    return service


def test_api_health():
    res = health_check()
    assert res["status"] == "healthy"
    assert res["eligible_tracks"] == 443375
    assert res["registered_users"] == 884


def test_api_users():
    res = list_sample_users(limit=10)
    assert len(res["users"]) == 10
    assert "user_id" in res["users"][0]
    assert "cohort" in res["users"][0]


def test_api_discover_supported_user():
    res = discover_recommendations(
        user_id="user_000001",
        limit=20,
        alpha=0.70,
        beta=0.20,
        max_per_artist=2,
    )
    assert res["user_id"] == "user_000001"
    assert len(res["recommendations"]) == 20
    assert res["latency_ms"] < 100.0  # sub-100ms real-time SLA
    
    # Check explainability attribution
    first_rec = res["recommendations"][0]
    assert "attribution" in first_rec
    assert "has_audio_preview" in first_rec
    attr = first_rec["attribution"]
    assert "attribution_tag" in attr
    assert "rationale" in attr
    assert "total_score" in attr
    assert "als_component" in attr
    assert "tt_component" in attr
    assert "novelty_component" in attr
    assert attr["novelty_bits"] > 0


def test_api_discover_cold_start():
    res = discover_recommendations(
        user_id="unknown_guest_listener",
        limit=10,
        max_per_artist=2,
    )
    assert res["inference_mode"] == "cold_start_fallback"
    assert len(res["recommendations"]) == 10


def test_api_hidden_gems():
    res = get_hidden_gems(
        user_id="user_000001",
        min_novelty_bits=15.0,
        limit=10,
    )
    assert len(res["gems"]) == 10
    for gem in res["gems"]:
        assert gem["novelty_bits"] >= 15.0
        assert "title" in gem
        assert "artist" in gem


def test_api_universe():
    res = get_music_universe(limit=25, user_id="user_000001")
    assert len(res["points"]) == 25
    pt = res["points"][0]
    assert "x" in pt
    assert "y" in pt
    assert "z" in pt
    assert "title" in pt
    assert "artist" in pt
    assert "is_recommended" in pt


def test_api_taste_profile():
    res = get_taste_profile(user_id="user_000001")
    assert res["user_id"] == "user_000001"
    assert 0.0 <= res["openness_score"] <= 100.0
    assert 0.0 <= res["mainstream_index"] <= 100.0
    assert 0.0 <= res["artist_diversity_score"] <= 100.0
    assert "acoustic_radar" in res
    assert "energy" in res["acoustic_radar"]
    assert len(res["top_artists"]) > 0


def test_api_lab():
    res = discovery_lab_rerank(
        user_id="user_000001",
        alpha=0.6,
        beta=0.3,
        max_per_artist=2,
        limit=10,
    )
    assert "live_deltas" in res
    assert "delta_novelty" in res["live_deltas"]
    assert "unique_artists" in res["live_deltas"]
    assert len(res["recommendations"]) == 10


def test_api_track_detail():
    res = get_track_detail(track_id=0)
    assert res["track_id"] == 0
    assert "title" in res
    assert "artist" in res
    assert "provenance" in res


def test_frontend_static_serving():
    assert UI_DIST_PATH.exists()
    index_file = UI_DIST_PATH / "index.html"
    assert index_file.exists()
    content = index_file.read_text(encoding="utf-8")
    assert "EUTERPE" in content
    assert '<div id="root"></div>' in content
    # Assert static mount is active on FastAPI app
    has_ui_mount = any(getattr(route, "name", "") == "ui" for route in app.routes)
    assert has_ui_mount


def test_api_feedback_submission():
    req = FeedbackSubmitRequest(
        user_id="user_000001",
        track_id=1042,
        action="discovered_something_new",
        slate_rank=1,
    )
    res = submit_discovery_feedback(req)
    assert res["status"] == "success"
    assert res["event"]["action"] == "discovered_something_new"
    assert res["event"]["track_id"] == 1042

    summary = get_feedback_summary()
    assert summary["total_events"] >= 1
    assert "action_counts" in summary

    history = get_user_feedback_history("user_000001")
    assert len(history["events"]) >= 1


def test_api_feedback_invalid_action():
    req = FeedbackSubmitRequest(
        user_id="user_000001",
        track_id=1042,
        action="invalid_random_action",
    )
    with pytest.raises(HTTPException) as exc_info:
        submit_discovery_feedback(req)
    assert exc_info.value.status_code == 400


def test_api_parameter_bounding_validation():
    with pytest.raises(HTTPException) as exc_info:
        discover_recommendations(alpha=1.5)
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        discover_recommendations(beta=-0.2)
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        discover_recommendations(max_per_artist=0)
    assert exc_info.value.status_code == 400


def test_closed_loop_feedback_adaptation():
    # 1. Fetch initial recommendations for user_000002
    initial_recs = discover_recommendations(user_id="user_000002", limit=10)
    first_track_id = initial_recs["recommendations"][0]["track_id"]

    # 2. Submit negative feedback ("dislike")
    submit_discovery_feedback(FeedbackSubmitRequest(
        user_id="user_000002",
        track_id=first_track_id,
        action="dislike",
        preference="negative",
    ))

    # 3. Query recommendations again - first_track_id must be strictly masked out!
    adapted_recs = discover_recommendations(user_id="user_000002", limit=10)
    adapted_track_ids = [r["track_id"] for r in adapted_recs["recommendations"]]
    assert first_track_id not in adapted_track_ids, f"Track {first_track_id} was disliked but appeared in recommendations!"


def test_batch_impressions_and_feedback_linking():
    import uuid
    p_id = f"participant_{uuid.uuid4().hex[:8]}"
    s_id = f"session_{uuid.uuid4().hex[:8]}"
    batch_req = BatchImpressionRequest(
        participant_id=p_id,
        session_id=s_id,
        items=[
            ImpressionItemModel(track_id=101, slate_rank=1, exposure_condition="system_A"),
            ImpressionItemModel(track_id=102, slate_rank=2, exposure_condition="system_A"),
            ImpressionItemModel(track_id=103, slate_rank=3, exposure_condition="system_A"),
        ],
    )
    rec_res = record_batch_impressions(batch_req)
    assert rec_res["status"] == "success"
    assert rec_res["recorded_impressions"] == 3
    first_imp_id = rec_res["impressions"][0]["impression_id"]

    # Submit feedback linked to first impression (Liked novel track -> Positive Discovery)
    fb_req = FeedbackSubmitRequest(
        user_id=p_id,
        track_id=101,
        action="discovered_something_new",
        familiarity="unknown",
        preference="positive",
        impression_id=first_imp_id,
    )
    fb_res = submit_discovery_feedback(fb_req)
    assert fb_res["status"] == "success"

    # Query impression metrics
    imp_summary = get_impressions_summary()
    assert imp_summary["total_impressions"] >= 3
    assert imp_summary["positive_discoveries"] >= 1
    assert "system_A" in imp_summary["condition_breakdown"]
    assert imp_summary["condition_breakdown"]["system_A"]["total_impressions"] >= 3

    # Participant query
    part_imps = get_participant_impressions(p_id)
    assert len(part_imps["impressions"]) == 3


def test_feedback_clear_and_reverse():
    # Submit feedback for user_000003
    submit_discovery_feedback(FeedbackSubmitRequest(
        user_id="user_000003",
        track_id=999,
        action="dislike",
        preference="negative",
    ))
    history_before = get_user_feedback_history("user_000003")
    assert len(history_before["events"]) >= 1

    # Clear feedback (reversal)
    del_res = delete_user_feedback_history("user_000003")
    assert del_res["status"] == "success"
    assert del_res["removed_events"] >= 1

    history_after = get_user_feedback_history("user_000003")
    assert len(history_after["events"]) == 0


def test_feedback_ablation_toggle():
    user = "user_000004"
    # Get a track that would be recommended
    initial_recs = discover_recommendations(user_id=user, limit=5, enable_feedback=False)
    first_track = initial_recs["recommendations"][0]["track_id"]

    # Submit dislike feedback
    submit_discovery_feedback(FeedbackSubmitRequest(
        user_id=user,
        track_id=first_track,
        action="dislike",
        preference="negative",
    ))

    # With feedback enabled: track is masked
    with_fb = discover_recommendations(user_id=user, limit=5, enable_feedback=True)
    assert first_track not in [r["track_id"] for r in with_fb["recommendations"]]

    # With feedback disabled (ablation mode): track remains unmasked
    without_fb = discover_recommendations(user_id=user, limit=5, enable_feedback=False)
    assert first_track == without_fb["recommendations"][0]["track_id"]


def test_audio_playback_telemetry_and_qualitative():
    import uuid
    p_id = f"participant_{uuid.uuid4().hex[:8]}"
    s_id = f"session_{uuid.uuid4().hex[:8]}"

    # Record impression with audio preview
    batch_req = BatchImpressionRequest(
        participant_id=p_id,
        session_id=s_id,
        items=[
            ImpressionItemModel(
                track_id=2024,
                slate_rank=1,
                exposure_condition="engine_alpha",
                has_audio_preview=True,
                preview_url="https://cdn.euterpe.discovery.internal/previews/30s/2024.mp3",
                preview_source="sample_provider",
            )
        ],
    )
    res = record_batch_impressions(batch_req)
    imp_id = res["impressions"][0]["impression_id"]

    # Record audio playback events
    audio_req = AudioPlaybackRecordRequest(
        impression_id=imp_id,
        started=True,
        completed=True,
        replay_increment=1,
        play_duration_sec=30.0,
    )
    audio_res = record_audio_telemetry(audio_req)
    assert audio_res["status"] == "success"
    assert audio_res["impression"]["audio_playback_started"] is True
    assert audio_res["impression"]["audio_playback_completed"] is True
    assert audio_res["impression"]["audio_replay_count"] >= 1

    # Record qualitative feedback
    q_req = QualitativeFeedbackRequest(
        participant_id=p_id,
        session_id=s_id,
        exposure_condition="engine_alpha",
        model_version="euterpe-v1.0.0",
        question="What did this recommendation experience do particularly well or poorly?",
        response_text="Discovered a stellar post-rock instrumental band I had never heard of before. Mood matched my evening listening state.",
    )
    q_res = submit_qualitative_feedback(q_req)
    assert q_res["status"] == "success"
    assert "post-rock" in q_res["event"]["response_text"]

    # Verify retrieval
    q_list = list_qualitative_feedback(participant_id=p_id)
    assert q_list["total"] == 1
    assert q_list["responses"][0]["exposure_condition"] == "engine_alpha"
