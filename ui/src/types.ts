export interface Attribution {
  track_id: number;
  rank: number;
  total_score: number;
  als_component: number;
  tt_component: number;
  novelty_component: number;
  novelty_bits: number;
  primary_source: string;
  attribution_tag: string;
  rationale: string;
  artist_cap_triggered: boolean;
}

export interface Track {
  track_id: number;
  recording_key: string;
  title: string;
  artist: string;
  artist_id: number;
  mbid_track?: string | null;
  mbid_artist?: string | null;
  playcount: number;
  novelty_bits: number;
  is_head_decile: boolean;
  musicbrainz_url?: string | null;
  lastfm_url: string;
  cover_art_url: string;
  energy: number;
  valence: number;
  danceability: number;
  acousticness: number;
  tempo_bpm: number;
  provenance: string;
  license: string;
  preview_status: string;
  has_audio_preview?: boolean;
  preview_url?: string | null;
  preview_source?: string | null;
  attribution?: Attribution;
  relevance_score?: number;
  gem_tag?: string;
}

export interface SlateMetrics {
  mean_novelty_bits: number;
  unique_artists: number;
  top_10pct_share: number;
}

export interface DiscoverResponse {
  user_id: string;
  resolved_user_idx: number | null;
  inference_mode: string;
  total_items: number;
  latency_ms: number;
  parameters: {
    alpha: number;
    beta: number;
    max_per_artist: number;
  };
  slate_metrics: SlateMetrics;
  recommendations: Track[];
}

export interface TasteProfile {
  user_id: string;
  user_idx: number;
  total_listens: number;
  unique_tracks: number;
  unique_artists: number;
  openness_score: number;
  mainstream_index: number;
  artist_diversity_score: number;
  mean_novelty_bits: number;
  acoustic_radar: {
    energy: number;
    valence: number;
    danceability: number;
    acousticness: number;
  };
  top_artists: Array<{
    artist_id: number;
    artist_name: string;
    plays: number;
  }>;
  history_sample: Track[];
  latency_ms: number;
}

export interface UniversePoint {
  track_id: number;
  title: string;
  artist: string;
  x: number;
  y: number;
  z: number;
  energy: number;
  valence: number;
  novelty_bits: number;
  is_head_decile: boolean;
  is_recommended: boolean;
  cover_art_url: string;
}

export interface LabResponse {
  user_id: string;
  latency_ms: number;
  controls: {
    alpha: number;
    beta: number;
    max_per_artist: number;
  };
  live_deltas: {
    novelty_bits: number;
    delta_novelty: number;
    unique_artists: number;
    delta_artists: number;
    artist_cap_interventions: number;
  };
  recommendations: Track[];
}
