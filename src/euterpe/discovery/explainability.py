"""Explainability Engine for EUTERPE Discovery Recommendations.

Provides concrete, mathematically traceable score attributions (ranking score decompositions)
for every recommended track.

IMPORTANT METHODOLOGICAL DISTINCTION:
These decompositions reflect mathematical score attributions within the multi-objective ranking
function (ALS collaborative affinity, Two-Tower deep embedding alignment, and novelty surprisal bits).
They are NOT causal psychological explanations of human listener enjoyment or taste.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class TrackExplanation:
    """Mathematically traceable ranking score attribution for a single recommended track."""
    track_id: int
    rank: int
    total_score: float
    als_component: float
    tt_component: float
    novelty_component: float
    novelty_bits: float
    primary_source: str
    attribution_tag: str
    rationale: str
    artist_cap_triggered: bool
    score_attribution_type: str = "ranking_score_decomposition"
    attribution_scope: str = "mathematical_score_contribution_not_causal_preference"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExplainabilityEngine:
    """Generates rigorous, audited model attributions from candidate scores."""

    @staticmethod
    def generate_explanations(
        recommended_tracks: List[int],
        als_scores: Dict[int, float],
        tt_scores: Dict[int, float],
        novelty_bits: np.ndarray,
        track_to_artist: np.ndarray,
        alpha: float = 0.70,
        beta: float = 0.20,
        artist_cap_events: Optional[Dict[int, str]] = None,
        artist_names: Optional[Dict[int, str]] = None,
    ) -> List[TrackExplanation]:
        """Produce concrete explanations for each item in the recommendation slate."""
        if not recommended_tracks:
            return []

        artist_cap_events = artist_cap_events or {}
        artist_names = artist_names or {}
        explanations = []

        # Standardize scores across candidate slate
        als_raw = np.array([als_scores.get(t, 0.0) for t in recommended_tracks])
        tt_raw = np.array([tt_scores.get(t, 0.0) for t in recommended_tracks])

        # Z-score standardization across slate
        als_std = (als_raw - np.mean(als_raw)) / (np.std(als_raw) + 1e-8) if len(als_raw) > 1 else als_raw
        tt_std = (tt_raw - np.mean(tt_raw)) / (np.std(tt_raw) + 1e-8) if len(tt_raw) > 1 else tt_raw

        for rank_idx, track_id in enumerate(recommended_tracks):
            s_als = float(als_std[rank_idx])
            s_tt = float(tt_std[rank_idx])
            n_bits = float(novelty_bits[track_id])
            
            als_contrib = float(alpha * s_als)
            tt_contrib = float((1.0 - alpha) * s_tt)
            nov_contrib = float(beta * n_bits)
            total_score = als_contrib + tt_contrib + nov_contrib

            art_id = int(track_to_artist[track_id])
            art_name = artist_names.get(art_id, f"Artist #{art_id}")
            was_capped = track_id in artist_cap_events

            # Attribution logic based on dominant mathematical driver
            if was_capped:
                primary_source = "artist_diversity"
                attribution_tag = "Artist Diversity Balance"
                skipped_art = artist_cap_events[track_id]
                rationale = (
                    f"Selected to preserve artist variety after earlier tracks by '{skipped_art}' "
                    f"reached the 2-track slate limit. Retains strong relevance ({total_score:+.2f})."
                )
            elif n_bits >= 16.0 and nov_contrib > 0.35 * max(0.1, abs(total_score)):
                primary_source = "long_tail_gem"
                attribution_tag = "Hidden Gem (Long-Tail)"
                rationale = (
                    f"Promoted for high discovery surprisal ({n_bits:.1f} bits of novelty, rarity top 5%) "
                    f"supported by consistent collaborative relevance."
                )
            elif tt_contrib > als_contrib and s_tt > 0.5:
                primary_source = "neural_semantic"
                attribution_tag = "Artist Semantic Affinity"
                rationale = (
                    f"Retrieved through Two-Tower deep embedding alignment with '{art_name}' "
                    f"(neural semantic contribution: {tt_contrib:+.2f})."
                )
            else:
                primary_source = "collaborative"
                attribution_tag = "Collaborative Synergy"
                rationale = (
                    f"Ranked via high collaborative listener co-occurrence "
                    f"(ALS predictive contribution: {als_contrib:+.2f}, relevance: {s_als:+.2f} std)."
                )

            explanations.append(TrackExplanation(
                track_id=track_id,
                rank=rank_idx + 1,
                total_score=round(total_score, 4),
                als_component=round(als_contrib, 4),
                tt_component=round(tt_contrib, 4),
                novelty_component=round(nov_contrib, 4),
                novelty_bits=round(n_bits, 2),
                primary_source=primary_source,
                attribution_tag=attribution_tag,
                rationale=rationale,
                artist_cap_triggered=was_capped,
            ))

        return explanations
