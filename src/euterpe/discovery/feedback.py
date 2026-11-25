"""Discovery Feedback and Interaction Tracking for EUTERPE.

Provides an empirical user-feedback collection layer bridging offline
historical interaction prediction with real-time discovery intelligence.

PHASE 7 ORTHOGONAL 2D TAXONOMY:
Separates Familiarity from Preference to avoid conflating unfamiliarity with enjoyment:
1. Familiarity Axis: Was this track previously known or unknown to the listener?
   - UNKNOWN: Novel track not previously encountered by listener.
   - FAMILIAR: Listener is already familiar with this track (prevents false novelty claims).
2. Preference Axis: How does the listener evaluate the track?
   - POSITIVE: Liked, loved, or recommended for similar tracks.
   - NEGATIVE: Disliked or explicitly rejected ("Not my taste").
   - SAVE_FOR_LATER: High-commitment retention intent.
   - NEUTRAL: Passive exposure or neutral assessment.

TRUE SERENDIPITY FORMULATION:
Serendipity is defined as a derived joint condition:
    True Serendipity = (Familiarity == UNKNOWN) AND (Preference in {POSITIVE, SAVE_FOR_LATER})
"""

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional


class FamiliarityLevel(str, Enum):
    UNKNOWN = "unknown"
    FAMILIAR = "familiar"
    UNSPECIFIED = "unspecified"


class PreferenceRating(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    SAVE_FOR_LATER = "save_for_later"
    UNSPECIFIED = "unspecified"


class FeedbackAction(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    ALREADY_KNOW = "already_know"
    NOT_MY_TASTE = "not_my_taste"
    MORE_LIKE_THIS = "more_like_this"
    SAVE_FOR_LATER = "save_for_later"
    DISCOVERED_SOMETHING_NEW = "discovered_something_new"


# Default mapping from action to 2D orthogonal space
ACTION_TO_ORTHOGONAL = {
    FeedbackAction.LIKE: (FamiliarityLevel.UNSPECIFIED, PreferenceRating.POSITIVE),
    FeedbackAction.DISLIKE: (FamiliarityLevel.UNSPECIFIED, PreferenceRating.NEGATIVE),
    FeedbackAction.ALREADY_KNOW: (FamiliarityLevel.FAMILIAR, PreferenceRating.UNSPECIFIED),
    FeedbackAction.NOT_MY_TASTE: (FamiliarityLevel.UNSPECIFIED, PreferenceRating.NEGATIVE),
    FeedbackAction.MORE_LIKE_THIS: (FamiliarityLevel.UNSPECIFIED, PreferenceRating.POSITIVE),
    FeedbackAction.SAVE_FOR_LATER: (FamiliarityLevel.UNSPECIFIED, PreferenceRating.SAVE_FOR_LATER),
    FeedbackAction.DISCOVERED_SOMETHING_NEW: (FamiliarityLevel.UNKNOWN, PreferenceRating.POSITIVE),
}


@dataclass
class FeedbackEvent:
    event_id: str
    user_id: str
    track_id: int
    action: FeedbackAction
    familiarity: FamiliarityLevel
    preference: PreferenceRating
    timestamp: float
    slate_rank: Optional[int] = None
    session_id: Optional[str] = None
    impression_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["track_id"] = int(self.track_id)
        d["action"] = self.action.value
        d["familiarity"] = self.familiarity.value
        d["preference"] = self.preference.value
        if self.slate_rank is not None:
            d["slate_rank"] = int(self.slate_rank)
        return d


@dataclass
class ImpressionEvent:
    impression_id: str
    participant_id: str
    session_id: str
    track_id: int
    slate_rank: int
    model_version: str
    exposure_condition: str
    timestamp: float
    familiarity: FamiliarityLevel = FamiliarityLevel.UNSPECIFIED
    preference: PreferenceRating = PreferenceRating.UNSPECIFIED
    action: Optional[FeedbackAction] = None
    has_audio_preview: bool = False
    preview_url: Optional[str] = None
    preview_source: Optional[str] = None
    audio_playback_started: bool = False
    audio_playback_completed: bool = False
    audio_replay_count: int = 0
    audio_play_duration_sec: float = 0.0
    audio_verified: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["track_id"] = int(self.track_id)
        d["slate_rank"] = int(self.slate_rank)
        d["familiarity"] = self.familiarity.value
        d["preference"] = self.preference.value
        d["action"] = self.action.value if self.action else None
        return d


@dataclass
class QualitativeFeedbackEvent:
    event_id: str
    participant_id: str
    session_id: str
    exposure_condition: str
    model_version: str
    question: str
    response_text: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeedbackStore:
    """Thread-safe persistent feedback store logging discovery signals."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("artifacts/feedback/discovery_feedback.jsonl")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._events: List[FeedbackEvent] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            action_val = FeedbackAction(data["action"])
                            def_fam, def_pref = ACTION_TO_ORTHOGONAL.get(
                                action_val, (FamiliarityLevel.UNSPECIFIED, PreferenceRating.UNSPECIFIED)
                            )
                            fam_val = FamiliarityLevel(data.get("familiarity", def_fam.value))
                            pref_val = PreferenceRating(data.get("preference", def_pref.value))

                            ev = FeedbackEvent(
                                event_id=data["event_id"],
                                user_id=data["user_id"],
                                track_id=int(data["track_id"]),
                                action=action_val,
                                familiarity=fam_val,
                                preference=pref_val,
                                timestamp=float(data["timestamp"]),
                                slate_rank=data.get("slate_rank"),
                                session_id=data.get("session_id"),
                                impression_id=data.get("impression_id"),
                                metadata=data.get("metadata"),
                            )
                            self._events.append(ev)
                        except Exception:
                            continue

    def record_feedback(
        self,
        user_id: str,
        track_id: int,
        action: FeedbackAction,
        familiarity: Optional[FamiliarityLevel] = None,
        preference: Optional[PreferenceRating] = None,
        slate_rank: Optional[int] = None,
        session_id: Optional[str] = None,
        impression_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackEvent:
        """Record a listener discovery feedback action with 2D mapping."""
        import uuid
        def_fam, def_pref = ACTION_TO_ORTHOGONAL.get(
            action, (FamiliarityLevel.UNSPECIFIED, PreferenceRating.UNSPECIFIED)
        )
        resolved_fam = familiarity if familiarity is not None else def_fam
        resolved_pref = preference if preference is not None else def_pref

        ev = FeedbackEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            track_id=int(track_id),
            action=action,
            familiarity=resolved_fam,
            preference=resolved_pref,
            timestamp=time.time(),
            slate_rank=slate_rank,
            session_id=session_id,
            impression_id=impression_id,
            metadata=metadata,
        )
        with self._lock:
            self._events.append(ev)
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev.to_dict()) + "\n")

        # Update impression store if linked to an impression
        if impression_id:
            try:
                imp_store = get_impression_store()
                imp_store.update_impression_feedback(
                    impression_id=impression_id,
                    action=action,
                    familiarity=resolved_fam,
                    preference=resolved_pref,
                )
            except Exception:
                pass

        return ev

    def clear_user_feedback(self, user_id: str) -> int:
        """Clear all feedback events recorded for a specific user (reversal/reset)."""
        with self._lock:
            initial_count = len(self._events)
            self._events = [e for e in self._events if e.user_id != user_id]
            removed = initial_count - len(self._events)
            self._persist_all_unlocked()
            return removed

    def remove_track_feedback(self, user_id: str, track_id: int) -> int:
        """Remove feedback for a specific track for a user."""
        with self._lock:
            initial_count = len(self._events)
            self._events = [e for e in self._events if not (e.user_id == user_id and e.track_id == track_id)]
            removed = initial_count - len(self._events)
            self._persist_all_unlocked()
            return removed

    def _persist_all_unlocked(self) -> None:
        """Rewrite feedback file to reflect updates or deletions."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            for ev in self._events:
                f.write(json.dumps(ev.to_dict()) + "\n")

    def get_user_feedback(self, user_id: str) -> List[FeedbackEvent]:
        """Fetch all feedback events recorded for a specific user."""
        with self._lock:
            return [e for e in self._events if e.user_id == user_id]

    def get_user_negative_tracks(self, user_id: str, session_id: Optional[str] = None) -> set:
        """Fetch track IDs explicitly rejected or disliked by user for candidate masking.
        If session_id is provided, strictly isolates feedback to that condition session.
        """
        with self._lock:
            return {
                e.track_id for e in self._events
                if e.user_id == user_id
                and e.preference == PreferenceRating.NEGATIVE
                and (session_id is None or e.session_id == session_id)
            }

    def get_user_positive_tracks(self, user_id: str, session_id: Optional[str] = None) -> set:
        """Fetch track IDs liked, saved, or marked positive by user.
        If session_id is provided, strictly isolates feedback to that condition session.
        """
        with self._lock:
            return {
                e.track_id for e in self._events
                if e.user_id == user_id
                and e.preference in (PreferenceRating.POSITIVE, PreferenceRating.SAVE_FOR_LATER)
                and (session_id is None or e.session_id == session_id)
            }

    def clear_user_session_feedback(self, user_id: str, session_id: str) -> int:
        """Clear feedback events for a user within a specific experimental condition session."""
        with self._lock:
            initial_count = len(self._events)
            self._events = [e for e in self._events if not (e.user_id == user_id and e.session_id == session_id)]
            removed = initial_count - len(self._events)
            self._persist_all_unlocked()
            return removed

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Compute aggregated feedback metrics across all 7 actions and 2D dimensions."""
        with self._lock:
            total = len(self._events)
            counts = {a.value: 0 for a in FeedbackAction}
            fam_counts = {f.value: 0 for f in FamiliarityLevel}
            pref_counts = {p.value: 0 for p in PreferenceRating}

            unknown_and_positive = 0
            unknown_total = 0

            for e in self._events:
                counts[e.action.value] = counts.get(e.action.value, 0) + 1
                fam_counts[e.familiarity.value] = fam_counts.get(e.familiarity.value, 0) + 1
                pref_counts[e.preference.value] = pref_counts.get(e.preference.value, 0) + 1

                if e.familiarity == FamiliarityLevel.UNKNOWN:
                    unknown_total += 1
                    if e.preference in (PreferenceRating.POSITIVE, PreferenceRating.SAVE_FOR_LATER):
                        unknown_and_positive += 1

            novel_discovery_rate = (fam_counts[FamiliarityLevel.UNKNOWN.value] / total * 100.0) if total > 0 else 0.0
            familiarity_rate = (fam_counts[FamiliarityLevel.FAMILIAR.value] / total * 100.0) if total > 0 else 0.0
            rejection_rate = (pref_counts[PreferenceRating.NEGATIVE.value] / total * 100.0) if total > 0 else 0.0
            positive_rate = (
                (pref_counts[PreferenceRating.POSITIVE.value] + pref_counts[PreferenceRating.SAVE_FOR_LATER.value])
                / total * 100.0 if total > 0 else 0.0
            )
            true_serendipity_rate = (unknown_and_positive / max(1, unknown_total) * 100.0) if unknown_total > 0 else 0.0

            return {
                "total_events": total,
                "action_counts": counts,
                "familiarity_breakdown": fam_counts,
                "preference_breakdown": pref_counts,
                "novel_discovery_rate_pct": round(novel_discovery_rate, 2),
                "familiarity_rate_pct": round(familiarity_rate, 2),
                "rejection_rate_pct": round(rejection_rate, 2),
                "positive_preference_rate_pct": round(positive_rate, 2),
                "true_serendipity_rate_pct": round(true_serendipity_rate, 2),
            }


_FEEDBACK_STORE_INSTANCE: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    global _FEEDBACK_STORE_INSTANCE
    if _FEEDBACK_STORE_INSTANCE is None:
        _FEEDBACK_STORE_INSTANCE = FeedbackStore()
    return _FEEDBACK_STORE_INSTANCE


class ImpressionStore:
    """Thread-safe persistent storage and telemetry aggregation for recommendation impressions."""

    def __init__(self, storage_path: Optional[Path] = None, qualitative_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("artifacts/feedback/discovery_impressions.jsonl")
        self.qualitative_path = qualitative_path or Path("artifacts/feedback/discovery_qualitative.jsonl")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.qualitative_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._impressions: Dict[str, ImpressionEvent] = {}  # keyed by impression_id
        self._qualitative: List[QualitativeFeedbackEvent] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            action_val = FeedbackAction(data["action"]) if data.get("action") else None
                            fam_val = FamiliarityLevel(data.get("familiarity", FamiliarityLevel.UNSPECIFIED.value))
                            pref_val = PreferenceRating(data.get("preference", PreferenceRating.UNSPECIFIED.value))
                            imp = ImpressionEvent(
                                impression_id=data["impression_id"],
                                participant_id=data["participant_id"],
                                session_id=data["session_id"],
                                track_id=int(data["track_id"]),
                                slate_rank=int(data["slate_rank"]),
                                model_version=data.get("model_version", "euterpe-v1.0.0"),
                                exposure_condition=data.get("exposure_condition", "default"),
                                timestamp=float(data["timestamp"]),
                                familiarity=fam_val,
                                preference=pref_val,
                                action=action_val,
                                has_audio_preview=data.get("has_audio_preview", False),
                                preview_url=data.get("preview_url"),
                                preview_source=data.get("preview_source"),
                                audio_playback_started=data.get("audio_playback_started", False),
                                audio_playback_completed=data.get("audio_playback_completed", False),
                                audio_replay_count=int(data.get("audio_replay_count", 0)),
                                audio_play_duration_sec=float(data.get("audio_play_duration_sec", 0.0)),
                                audio_verified=bool(data.get("audio_verified", False)),
                                metadata=data.get("metadata"),
                            )
                            self._impressions[imp.impression_id] = imp
                        except Exception:
                            continue

        if self.qualitative_path.exists():
            with open(self.qualitative_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            qdata = json.loads(line)
                            qev = QualitativeFeedbackEvent(
                                event_id=qdata["event_id"],
                                participant_id=qdata["participant_id"],
                                session_id=qdata["session_id"],
                                exposure_condition=qdata["exposure_condition"],
                                model_version=qdata.get("model_version", "unknown"),
                                question=qdata["question"],
                                response_text=qdata["response_text"],
                                timestamp=float(qdata["timestamp"]),
                            )
                            self._qualitative.append(qev)
                        except Exception:
                            continue

    def record_impression(
        self,
        participant_id: str,
        session_id: str,
        track_id: int,
        slate_rank: int,
        model_version: str = "euterpe-v1.0.0",
        exposure_condition: str = "default",
        impression_id: Optional[str] = None,
        has_audio_preview: bool = False,
        preview_url: Optional[str] = None,
        preview_source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ImpressionEvent:
        """Record an individual recommendation impression with audio preview tracking."""
        import uuid
        imp_id = impression_id or str(uuid.uuid4())
        imp = ImpressionEvent(
            impression_id=imp_id,
            participant_id=participant_id,
            session_id=session_id,
            track_id=track_id,
            slate_rank=slate_rank,
            model_version=model_version,
            exposure_condition=exposure_condition,
            timestamp=time.time(),
            has_audio_preview=has_audio_preview,
            preview_url=preview_url,
            preview_source=preview_source,
            metadata=metadata,
        )
        with self._lock:
            self._impressions[imp_id] = imp
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(imp.to_dict()) + "\n")
        return imp

    def record_slate_impressions(
        self,
        participant_id: str,
        session_id: str,
        track_ids: List[int],
        model_version: str = "euterpe-v1.0.0",
        exposure_condition: str = "default",
        preview_urls: Optional[Dict[int, str]] = None,
        preview_sources: Optional[Dict[int, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ImpressionEvent]:
        """Record full slate of recommended items as individual impressions."""
        import uuid
        results = []
        preview_urls = preview_urls or {}
        preview_sources = preview_sources or {}
        with self._lock:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                for rank_i, tid in enumerate(track_ids, start=1):
                    imp_id = str(uuid.uuid4())
                    p_url = preview_urls.get(tid)
                    p_src = preview_sources.get(tid, "deezer" if p_url else "unavailable")
                    imp = ImpressionEvent(
                        impression_id=imp_id,
                        participant_id=participant_id,
                        session_id=session_id,
                        track_id=tid,
                        slate_rank=rank_i,
                        model_version=model_version,
                        exposure_condition=exposure_condition,
                        timestamp=time.time(),
                        has_audio_preview=bool(p_url),
                        preview_url=p_url,
                        preview_source=p_src,
                        metadata=metadata,
                    )
                    self._impressions[imp_id] = imp
                    results.append(imp)
                    f.write(json.dumps(imp.to_dict()) + "\n")
        return results

    def record_audio_playback(
        self,
        impression_id: str,
        started: bool = True,
        completed: bool = False,
        replay_increment: int = 0,
        play_duration_sec: float = 0.0,
        audio_verified: Optional[bool] = None,
    ) -> Optional[ImpressionEvent]:
        """Record audio playback telemetry event on an impression.
        
        Enforces Phase 7B Safeguard Item A: Playback qualification rule.
        Impression is certified as audio_verified if >= 5.0 seconds of audio
        is played or if playback is completed.
        """
        with self._lock:
            if impression_id not in self._impressions:
                return None
            imp = self._impressions[impression_id]
            if started:
                imp.audio_playback_started = True
            if completed:
                imp.audio_playback_completed = True
            if replay_increment > 0:
                imp.audio_replay_count += replay_increment
            if play_duration_sec > 0.0:
                imp.audio_play_duration_sec += play_duration_sec
            
            # Qualification rule: >= 5.0s playback duration or full completion
            if audio_verified is not None:
                imp.audio_verified = audio_verified
            elif imp.audio_play_duration_sec >= 5.0 or imp.audio_playback_completed:
                imp.audio_verified = True

            self._persist_all_unlocked()
            return imp

    def record_qualitative_feedback(
        self,
        participant_id: str,
        session_id: str,
        exposure_condition: str,
        model_version: str,
        question: str,
        response_text: str,
    ) -> QualitativeFeedbackEvent:
        """Record qualitative post-slate open-ended response."""
        import uuid
        qev = QualitativeFeedbackEvent(
            event_id=str(uuid.uuid4()),
            participant_id=participant_id,
            session_id=session_id,
            exposure_condition=exposure_condition,
            model_version=model_version,
            question=question,
            response_text=response_text.strip(),
            timestamp=time.time(),
        )
        with self._lock:
            self._qualitative.append(qev)
            with open(self.qualitative_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(qev.to_dict()) + "\n")
        return qev

    def get_qualitative_feedback(self, participant_id: Optional[str] = None) -> List[QualitativeFeedbackEvent]:
        with self._lock:
            if participant_id:
                return [q for q in self._qualitative if q.participant_id == participant_id]
            return list(self._qualitative)

    def update_impression_feedback(
        self,
        impression_id: str,
        action: FeedbackAction,
        familiarity: Optional[FamiliarityLevel] = None,
        preference: Optional[PreferenceRating] = None,
        persist: bool = True,
    ) -> Optional[ImpressionEvent]:
        """Update an existing impression with participant feedback."""
        with self._lock:
            if impression_id not in self._impressions:
                return None
            imp = self._impressions[impression_id]
            if action is not None:
                imp.action = FeedbackAction(action) if not isinstance(action, FeedbackAction) else action
            if familiarity is not None:
                imp.familiarity = FamiliarityLevel(familiarity) if not isinstance(familiarity, FamiliarityLevel) else familiarity
            if preference is not None:
                imp.preference = PreferenceRating(preference) if not isinstance(preference, PreferenceRating) else preference
            if persist:
                self._persist_all_unlocked()
            return imp

    def persist_all(self) -> None:
        """Explicitly write all impressions to disk."""
        with self._lock:
            self._persist_all_unlocked()

    def _persist_all_unlocked(self) -> None:
        """Rewrite impressions file to reflect updates."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            for imp in self._impressions.values():
                f.write(json.dumps(imp.to_dict()) + "\n")

    def get_impressions_by_participant(self, participant_id: str) -> List[ImpressionEvent]:
        with self._lock:
            return [imp for imp in self._impressions.values() if imp.participant_id == participant_id]

    def get_impressions_by_session(self, session_id: str) -> List[ImpressionEvent]:
        with self._lock:
            return [imp for imp in self._impressions.values() if imp.session_id == session_id]

    def clear(self) -> None:
        with self._lock:
            self._impressions.clear()
            self._qualitative.clear()
            if self.storage_path.exists():
                self.storage_path.unlink()
            if self.qualitative_path.exists():
                self.qualitative_path.unlink()

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Compute impression-level metrics with disambiguated conditional vs yield indicators."""
        with self._lock:
            total_impressions = len(self._impressions)
            if total_impressions == 0:
                return {
                    "total_impressions": 0,
                    "evaluated_impressions": 0,
                    "evaluation_coverage_pct": 0.0,
                    "unknown_evaluated_count": 0,
                    "positive_discoveries": 0,
                    "saved_discoveries": 0,
                    "rejected_discoveries": 0,
                    "familiar_impressions": 0,
                    # Primary Gate 1 Endpoint: Conditional Positive Discovery
                    "conditional_positive_discovery_rate_pct": 0.0,
                    # Secondary Product Metric: Positive Discovery Yield
                    "positive_discovery_yield_pct": 0.0,
                    "saved_discovery_yield_pct": 0.0,
                    "conditional_saved_discovery_rate_pct": 0.0,
                    "conditional_rejection_rate_pct": 0.0,
                    "rejection_yield_pct": 0.0,
                    "conditional_serendipity_rate_pct": 0.0,
                    "serendipity_yield_pct": 0.0,
                    # Audio metrics
                    "audio_preview_count": 0,
                    "audio_preview_coverage_pct": 0.0,
                    "audio_playback_started_count": 0,
                    "audio_playback_rate_pct": 0.0,
                    "audio_playback_completed_count": 0,
                    "audio_completion_rate_pct": 0.0,
                    "total_replay_count": 0,
                    # Safeguard Item A (Audio-Verified vs Metadata-Only)
                    "audio_verified_impressions_count": 0,
                    "audio_verified_evaluated_count": 0,
                    "audio_verified_unknown_evaluated_count": 0,
                    "audio_verified_positive_discoveries": 0,
                    "audio_verified_conditional_pdr_pct": 0.0,
                    "metadata_only_impressions_count": 0,
                    "metadata_only_evaluated_count": 0,
                    "metadata_only_unknown_evaluated_count": 0,
                    "metadata_only_positive_discoveries": 0,
                    "metadata_only_conditional_pdr_pct": 0.0,
                    # Safeguard Item B (LTE Prospective >= 16.0 bits)
                    "long_tail_impressions_count": 0,
                    "long_tail_evaluated_count": 0,
                    "long_tail_unknown_evaluated_count": 0,
                    "long_tail_engaged_count": 0,
                    "prospective_lte_pct": 0.0,
                    "condition_breakdown": {},
                }

            evaluated = 0
            pos_disc = 0
            saved_disc = 0
            rejected_disc = 0
            familiar_count = 0
            unknown_evaluated_count = 0

            audio_preview_count = 0
            audio_started_count = 0
            audio_completed_count = 0
            total_replays = 0

            # Safeguard Item A & B tracking
            audio_ver_total = 0
            audio_ver_eval = 0
            audio_ver_unk_eval = 0
            audio_ver_pos = 0

            meta_only_total = 0
            meta_only_eval = 0
            meta_only_unk_eval = 0
            meta_only_pos = 0

            lt_total = 0
            lt_eval = 0
            lt_unk_eval = 0
            lt_engaged = 0

            condition_map: Dict[str, Dict[str, Any]] = {}

            for imp in self._impressions.values():
                c = imp.exposure_condition
                if c not in condition_map:
                    condition_map[c] = {
                        "total": 0,
                        "evaluated": 0,
                        "unknown_evaluated": 0,
                        "pos_disc": 0,
                        "saved_disc": 0,
                        "rejected_disc": 0,
                        "familiar": 0,
                        "audio_preview": 0,
                        "audio_started": 0,
                        "audio_completed": 0,
                        "audio_replays": 0,
                        "audio_ver_total": 0,
                        "audio_ver_eval": 0,
                        "audio_ver_unk_eval": 0,
                        "audio_ver_pos": 0,
                        "meta_only_total": 0,
                        "meta_only_eval": 0,
                        "meta_only_unk_eval": 0,
                        "meta_only_pos": 0,
                        "lt_total": 0,
                        "lt_eval": 0,
                        "lt_unk_eval": 0,
                        "lt_engaged": 0,
                    }
                condition_map[c]["total"] += 1

                if imp.has_audio_preview:
                    audio_preview_count += 1
                    condition_map[c]["audio_preview"] += 1
                if imp.audio_playback_started:
                    audio_started_count += 1
                    condition_map[c]["audio_started"] += 1
                if imp.audio_playback_completed:
                    audio_completed_count += 1
                    condition_map[c]["audio_completed"] += 1
                total_replays += imp.audio_replay_count
                condition_map[c]["audio_replays"] += imp.audio_replay_count

                is_audio_ver = imp.audio_verified
                if is_audio_ver:
                    audio_ver_total += 1
                    condition_map[c]["audio_ver_total"] += 1
                else:
                    meta_only_total += 1
                    condition_map[c]["meta_only_total"] += 1

                nov = 0.0
                if imp.metadata:
                    nov = float(imp.metadata.get("novelty_bits", imp.metadata.get("novelty", 0.0)) or 0.0)
                is_lt = nov >= 16.0
                if is_lt:
                    lt_total += 1
                    condition_map[c]["lt_total"] += 1

                has_eval = (
                    imp.familiarity != FamiliarityLevel.UNSPECIFIED
                    or imp.preference != PreferenceRating.UNSPECIFIED
                    or imp.action is not None
                )
                if has_eval:
                    evaluated += 1
                    condition_map[c]["evaluated"] += 1
                    if is_audio_ver:
                        audio_ver_eval += 1
                        condition_map[c]["audio_ver_eval"] += 1
                    else:
                        meta_only_eval += 1
                        condition_map[c]["meta_only_eval"] += 1
                    if is_lt:
                        lt_eval += 1
                        condition_map[c]["lt_eval"] += 1

                if imp.familiarity == FamiliarityLevel.FAMILIAR:
                    familiar_count += 1
                    condition_map[c]["familiar"] += 1

                # Evaluate Unknown pool (impressions evaluated conditional on being unknown)
                is_unk_eval = (imp.familiarity == FamiliarityLevel.UNKNOWN and imp.preference != PreferenceRating.UNSPECIFIED)
                if is_unk_eval:
                    unknown_evaluated_count += 1
                    condition_map[c]["unknown_evaluated"] += 1
                    if is_audio_ver:
                        audio_ver_unk_eval += 1
                        condition_map[c]["audio_ver_unk_eval"] += 1
                    else:
                        meta_only_unk_eval += 1
                        condition_map[c]["meta_only_unk_eval"] += 1
                    if is_lt:
                        lt_unk_eval += 1
                        condition_map[c]["lt_unk_eval"] += 1

                # Positive Discovery: Unfamiliar + Liked
                if imp.familiarity == FamiliarityLevel.UNKNOWN and imp.preference == PreferenceRating.POSITIVE:
                    pos_disc += 1
                    condition_map[c]["pos_disc"] += 1
                    if is_audio_ver:
                        audio_ver_pos += 1
                        condition_map[c]["audio_ver_pos"] += 1
                    else:
                        meta_only_pos += 1
                        condition_map[c]["meta_only_pos"] += 1
                    if is_lt:
                        lt_engaged += 1
                        condition_map[c]["lt_engaged"] += 1

                # Saved Discovery: Unfamiliar + Saved for later (investigative intent)
                elif imp.familiarity == FamiliarityLevel.UNKNOWN and imp.preference == PreferenceRating.SAVE_FOR_LATER:
                    saved_disc += 1
                    condition_map[c]["saved_disc"] += 1
                    if is_lt:
                        lt_engaged += 1
                        condition_map[c]["lt_engaged"] += 1

                # Rejected Discovery: Unfamiliar + Disliked/Not my taste
                elif imp.familiarity == FamiliarityLevel.UNKNOWN and imp.preference == PreferenceRating.NEGATIVE:
                    rejected_disc += 1
                    condition_map[c]["rejected_disc"] += 1

            # Format condition breakdowns
            breakdowns = {}
            for c, counts in condition_map.items():
                c_total = max(1, counts["total"])
                c_unk_eval = max(1, counts["unknown_evaluated"])
                c_audio_prev = max(1, counts["audio_preview"])
                c_audio_started = max(1, counts["audio_started"])
                c_audio_ver_unk = max(1, counts["audio_ver_unk_eval"])
                c_meta_only_unk = max(1, counts["meta_only_unk_eval"])
                c_lt_unk = max(1, counts["lt_unk_eval"])

                breakdowns[c] = {
                    "total_impressions": counts["total"],
                    "evaluated_impressions": counts["evaluated"],
                    "unknown_evaluated_count": counts["unknown_evaluated"],
                    "positive_discoveries": counts["pos_disc"],
                    "saved_discoveries": counts["saved_disc"],
                    "rejected_discoveries": counts["rejected_disc"],
                    "familiar_impressions": counts["familiar"],
                    # Primary Gate 1 Endpoint: Conditional Positive Discovery Rate
                    "conditional_positive_discovery_rate_pct": round(counts["pos_disc"] / c_unk_eval * 100.0, 2) if counts["unknown_evaluated"] > 0 else 0.0,
                    # Secondary Product Metric: Positive Discovery Yield
                    "positive_discovery_yield_pct": round(counts["pos_disc"] / c_total * 100.0, 2),
                    "saved_discovery_yield_pct": round(counts["saved_disc"] / c_total * 100.0, 2),
                    "conditional_saved_discovery_rate_pct": round(counts["saved_disc"] / c_unk_eval * 100.0, 2) if counts["unknown_evaluated"] > 0 else 0.0,
                    "conditional_rejection_rate_pct": round(counts["rejected_disc"] / c_unk_eval * 100.0, 2) if counts["unknown_evaluated"] > 0 else 0.0,
                    "rejection_yield_pct": round(counts["rejected_disc"] / c_total * 100.0, 2),
                    "conditional_serendipity_rate_pct": round((counts["pos_disc"] + counts["saved_disc"]) / c_unk_eval * 100.0, 2) if counts["unknown_evaluated"] > 0 else 0.0,
                    "serendipity_yield_pct": round((counts["pos_disc"] + counts["saved_disc"]) / c_total * 100.0, 2),
                    # Audio metrics
                    "audio_preview_count": counts["audio_preview"],
                    "audio_preview_coverage_pct": round(counts["audio_preview"] / c_total * 100.0, 2),
                    "audio_playback_started_count": counts["audio_started"],
                    "audio_playback_rate_pct": round(counts["audio_started"] / c_audio_prev * 100.0, 2) if counts["audio_preview"] > 0 else 0.0,
                    "audio_playback_completed_count": counts["audio_completed"],
                    "audio_completion_rate_pct": round(counts["audio_completed"] / c_audio_started * 100.0, 2) if counts["audio_started"] > 0 else 0.0,
                    "total_replay_count": counts["audio_replays"],
                    # Safeguard Item A (Audio-Verified vs Metadata-Only)
                    "audio_verified_impressions_count": counts["audio_ver_total"],
                    "audio_verified_evaluated_count": counts["audio_ver_eval"],
                    "audio_verified_unknown_evaluated_count": counts["audio_ver_unk_eval"],
                    "audio_verified_positive_discoveries": counts["audio_ver_pos"],
                    "audio_verified_conditional_pdr_pct": round(counts["audio_ver_pos"] / c_audio_ver_unk * 100.0, 2) if counts["audio_ver_unk_eval"] > 0 else 0.0,
                    "metadata_only_impressions_count": counts["meta_only_total"],
                    "metadata_only_evaluated_count": counts["meta_only_eval"],
                    "metadata_only_unknown_evaluated_count": counts["meta_only_unk_eval"],
                    "metadata_only_positive_discoveries": counts["meta_only_pos"],
                    "metadata_only_conditional_pdr_pct": round(counts["meta_only_pos"] / c_meta_only_unk * 100.0, 2) if counts["meta_only_unk_eval"] > 0 else 0.0,
                    # Safeguard Item B (LTE Prospective >= 16.0 bits)
                    "long_tail_impressions_count": counts["lt_total"],
                    "long_tail_evaluated_count": counts["lt_eval"],
                    "long_tail_unknown_evaluated_count": counts["lt_unk_eval"],
                    "long_tail_engaged_count": counts["lt_engaged"],
                    "prospective_lte_pct": round(counts["lt_engaged"] / c_lt_unk * 100.0, 2) if counts["lt_unk_eval"] > 0 else 0.0,
                }

            global_unk_eval = max(1, unknown_evaluated_count)
            global_audio_prev = max(1, audio_preview_count)
            global_audio_started = max(1, audio_started_count)
            global_audio_ver_unk = max(1, audio_ver_unk_eval)
            global_meta_only_unk = max(1, meta_only_unk_eval)
            global_lt_unk = max(1, lt_unk_eval)

            return {
                "total_impressions": total_impressions,
                "evaluated_impressions": evaluated,
                "evaluation_coverage_pct": round(evaluated / total_impressions * 100.0, 2),
                "unknown_evaluated_count": unknown_evaluated_count,
                "positive_discoveries": pos_disc,
                "saved_discoveries": saved_disc,
                "rejected_discoveries": rejected_disc,
                "familiar_impressions": familiar_count,
                # Primary Gate 1 Endpoint: Conditional Positive Discovery Rate
                "conditional_positive_discovery_rate_pct": round(pos_disc / global_unk_eval * 100.0, 2) if unknown_evaluated_count > 0 else 0.0,
                # Secondary Product Metric: Positive Discovery Yield
                "positive_discovery_yield_pct": round(pos_disc / total_impressions * 100.0, 2),
                "saved_discovery_yield_pct": round(saved_disc / total_impressions * 100.0, 2),
                "conditional_saved_discovery_rate_pct": round(saved_disc / global_unk_eval * 100.0, 2) if unknown_evaluated_count > 0 else 0.0,
                "conditional_rejection_rate_pct": round(rejected_disc / global_unk_eval * 100.0, 2) if unknown_evaluated_count > 0 else 0.0,
                "rejection_yield_pct": round(rejected_disc / total_impressions * 100.0, 2),
                "conditional_serendipity_rate_pct": round((pos_disc + saved_disc) / global_unk_eval * 100.0, 2) if unknown_evaluated_count > 0 else 0.0,
                "serendipity_yield_pct": round((pos_disc + saved_disc) / total_impressions * 100.0, 2),
                # Audio metrics
                "audio_preview_count": audio_preview_count,
                "audio_preview_coverage_pct": round(audio_preview_count / total_impressions * 100.0, 2),
                "audio_playback_started_count": audio_started_count,
                "audio_playback_rate_pct": round(audio_started_count / global_audio_prev * 100.0, 2) if audio_preview_count > 0 else 0.0,
                "audio_playback_completed_count": audio_completed_count,
                "audio_completion_rate_pct": round(audio_completed_count / global_audio_started * 100.0, 2) if audio_started_count > 0 else 0.0,
                "total_replay_count": total_replays,
                # Safeguard Item A (Audio-Verified vs Metadata-Only)
                "audio_verified_impressions_count": audio_ver_total,
                "audio_verified_evaluated_count": audio_ver_eval,
                "audio_verified_unknown_evaluated_count": audio_ver_unk_eval,
                "audio_verified_positive_discoveries": audio_ver_pos,
                "audio_verified_conditional_pdr_pct": round(audio_ver_pos / global_audio_ver_unk * 100.0, 2) if audio_ver_unk_eval > 0 else 0.0,
                "metadata_only_impressions_count": meta_only_total,
                "metadata_only_evaluated_count": meta_only_eval,
                "metadata_only_unknown_evaluated_count": meta_only_unk_eval,
                "metadata_only_positive_discoveries": meta_only_pos,
                "metadata_only_conditional_pdr_pct": round(meta_only_pos / global_meta_only_unk * 100.0, 2) if meta_only_unk_eval > 0 else 0.0,
                # Safeguard Item B (LTE Prospective >= 16.0 bits)
                "long_tail_impressions_count": lt_total,
                "long_tail_evaluated_count": lt_eval,
                "long_tail_unknown_evaluated_count": lt_unk_eval,
                "long_tail_engaged_count": lt_engaged,
                "prospective_lte_pct": round(lt_engaged / global_lt_unk * 100.0, 2) if lt_unk_eval > 0 else 0.0,
                "condition_breakdown": breakdowns,
            }


_IMPRESSION_STORE_INSTANCE: Optional[ImpressionStore] = None


def get_impression_store() -> ImpressionStore:
    global _IMPRESSION_STORE_INSTANCE
    if _IMPRESSION_STORE_INSTANCE is None:
        _IMPRESSION_STORE_INSTANCE = ImpressionStore()
    return _IMPRESSION_STORE_INSTANCE
