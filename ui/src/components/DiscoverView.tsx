import React, { useState, useRef } from "react";
import { Sparkles, Play, Pause, RotateCcw, Volume2, Shield, Info, ExternalLink, Award, Sliders, Layers, ThumbsUp, ThumbsDown, Eye, Bookmark, Check, MessageSquare } from "lucide-react";
import { DiscoverResponse, Track } from "../types";

interface DiscoverViewProps {
  data: DiscoverResponse | null;
  loading: boolean;
  onPlayTrack: (track: Track) => void;
  onRefresh: () => void;
}

export const DiscoverView: React.FC<DiscoverViewProps> = ({
  data,
  loading,
  onPlayTrack,
  onRefresh,
}) => {
  const [selectedExplanation, setSelectedExplanation] = useState<Track | null>(null);
  const [feedbackMap, setFeedbackMap] = useState<Record<number, string>>({});
  const [activePreviewTrackId, setActivePreviewTrackId] = useState<number | null>(null);
  const [isPlayingPreview, setIsPlayingPreview] = useState<boolean>(false);
  const [qualitativeText, setQualitativeText] = useState<string>("");
  const [qualitativeSubmitted, setQualitativeSubmitted] = useState<boolean>(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handlePreviewPlay = async (track: Track) => {
    if (!track.has_audio_preview || !track.preview_url) {
      onPlayTrack(track); // Fallback tone synthesis
      return;
    }

    if (activePreviewTrackId === track.track_id && isPlayingPreview) {
      audioRef.current?.pause();
      setIsPlayingPreview(false);
      return;
    }

    setActivePreviewTrackId(track.track_id);
    setIsPlayingPreview(true);

    if (audioRef.current) {
      audioRef.current.src = track.preview_url;
      audioRef.current.play().catch((err) => {
        console.warn("Audio preview playback hindered by browser policy:", err);
      });
    }

    // Telemetry: record playback start
    try {
      await fetch("/api/impressions/audio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          impression_id: `imp_${track.track_id}_${data?.user_id || "listener"}`,
          started: true,
          completed: false,
          replay_increment: 0,
          play_duration_sec: 0.0,
        }),
      });
    } catch (_) {}
  };

  const handleAudioEnded = async () => {
    setIsPlayingPreview(false);
    if (!activePreviewTrackId) return;
    try {
      await fetch("/api/impressions/audio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          impression_id: `imp_${activePreviewTrackId}_${data?.user_id || "listener"}`,
          started: false,
          completed: true,
          replay_increment: 0,
          play_duration_sec: 30.0,
        }),
      });
    } catch (_) {}
  };

  const handleAudioReplay = async (trackId: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play();
      setIsPlayingPreview(true);
    }
    try {
      await fetch("/api/impressions/audio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          impression_id: `imp_${trackId}_${data?.user_id || "listener"}`,
          started: true,
          completed: false,
          replay_increment: 1,
          play_duration_sec: 0.0,
        }),
      });
    } catch (_) {}
  };

  const handleFeedback = async (trackId: number, action: string, rank: number) => {
    if (!data) return;
    setFeedbackMap((prev) => ({ ...prev, [trackId]: action }));
    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: data.user_id,
          track_id: trackId,
          action: action,
          slate_rank: rank,
          session_id: "phase7b_active_session",
        }),
      });
    } catch (err) {
      console.error("Failed to submit feedback:", err);
    }
  };

  const handleQualitativeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!qualitativeText.trim() || !data) return;
    try {
      await fetch("/api/feedback/qualitative", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          participant_id: data.user_id,
          session_id: "phase7b_active_session",
          exposure_condition: "blind_engine_active",
          model_version: "euterpe-v1.0.0",
          question: "What did this recommendation experience do particularly well or poorly?",
          response_text: qualitativeText,
        }),
      });
      setQualitativeSubmitted(true);
    } catch (err) {
      console.error("Failed to submit qualitative feedback:", err);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <div className="w-12 h-12 border-4 border-violet-500/20 border-t-violet-500 rounded-full animate-spin" />
        <p className="text-slate-400 font-medium">Computing hybrid discovery slate across 443,375 tracks...</p>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Banner & Slate Metrics */}
      <div className="glass-panel p-6 rounded-2xl relative overflow-hidden border border-violet-500/20">
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-br from-violet-600/10 via-transparent to-transparent pointer-events-none" />
        
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-violet-600/20 text-violet-300 border border-violet-500/30 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-gold-400" /> Hybrid Discovery Model
              </span>
              <span className="text-xs text-slate-400 font-mono">
                P95 Latency: {data.latency_ms} ms
              </span>
            </div>
            <h1 className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              Personalized Discovery Slate
            </h1>
            <p className="text-sm text-slate-400 mt-1">
              Dual-source fusion (ALS collaborative + Two-Tower neural) with novelty regularized ranking and bounded artist variety.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onRefresh}
              className="px-4 py-2 rounded-xl bg-obsidian-800 hover:bg-obsidian-700 text-sm font-semibold text-slate-200 border border-obsidian-600 transition"
            >
              Re-Rank Slate
            </button>
          </div>
        </div>

        {/* Live Slate Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-obsidian-700/60 font-mono text-xs">
          <div className="p-3 rounded-xl bg-obsidian-900/60 border border-obsidian-800">
            <span className="text-slate-400 block mb-1">AVERAGE NOVELTY</span>
            <span className="text-base font-bold text-amber-400">
              {data.slate_metrics.mean_novelty_bits} bits
            </span>
          </div>
          <div className="p-3 rounded-xl bg-obsidian-900/60 border border-obsidian-800">
            <span className="text-slate-400 block mb-1">UNIQUE ARTISTS</span>
            <span className="text-base font-bold text-violet-400">
              {data.slate_metrics.unique_artists} / {data.recommendations.length} slots
            </span>
          </div>
          <div className="p-3 rounded-xl bg-obsidian-900/60 border border-obsidian-800">
            <span className="text-slate-400 block mb-1">HEAD EXPOSURE</span>
            <span className="text-base font-bold text-cyan-400">
              {data.slate_metrics.top_10pct_share}% top decile
            </span>
          </div>
          <div className="p-3 rounded-xl bg-obsidian-900/60 border border-obsidian-800">
            <span className="text-slate-400 block mb-1">BOUNDED ARTIST CAP</span>
            <span className="text-base font-bold text-emerald-400">
              Max {data.parameters.max_per_artist} / artist
            </span>
          </div>
        </div>
      </div>

      {/* Recommended Tracks Grid */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs text-slate-400 font-semibold px-2 uppercase tracking-wider">
          <span>Rank & Track Details</span>
          <span>Model Attribution & Provenance</span>
        </div>

        {data.recommendations.map((track, idx) => {
          const attr = track.attribution;
          const isSelected = selectedExplanation?.track_id === track.track_id;

          return (
            <div
              key={track.track_id}
              className={`p-4 rounded-xl transition duration-200 border ${
                isSelected
                  ? "bg-violet-950/40 border-violet-500/60 shadow-lg shadow-violet-950/50"
                  : "glass-panel hover:bg-obsidian-800/80 border-obsidian-800"
              }`}
            >
              <div className="flex items-center justify-between gap-4">
                {/* Track Info */}
                <div className="flex items-center gap-4 flex-1 min-w-0">
                  <span className="text-base font-bold font-mono text-slate-400 w-6 text-center">
                    {idx + 1}
                  </span>

                  <div className="relative group cursor-pointer" onClick={() => onPlayTrack(track)}>
                    <img
                      src={track.cover_art_url}
                      alt={track.title}
                      className="w-14 h-14 rounded-lg object-cover border border-obsidian-700 shadow-md group-hover:opacity-75 transition"
                      onError={(e) => {
                        (e.target as any).src = `https://api.dicebear.com/7.x/identicon/svg?seed=${track.artist_id}`;
                      }}
                    />
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition">
                      <Play className="w-6 h-6 text-white fill-white drop-shadow-md" />
                    </div>
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <h4 className="font-bold text-white text-base truncate hover:text-violet-300 transition cursor-pointer" onClick={() => onPlayTrack(track)}>
                        {track.title}
                      </h4>
                      {track.is_head_decile ? (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-obsidian-700 text-slate-400">
                          Popular
                        </span>
                      ) : (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          Discovery
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-slate-300 truncate">{track.artist}</p>
                    
                    <div className="flex items-center gap-3 text-xs text-slate-400 font-mono mt-1">
                      <span>Novelty: {track.novelty_bits.toFixed(1)} bits</span>
                      <span>Tempo: {track.tempo_bpm} BPM</span>
                      <span>Energy (Est): {(track.energy * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>

                {/* Attribution Badge & Controls */}
                <div className="flex items-center gap-3">
                  {attr && (
                    <div className="text-right hidden sm:block">
                      <span
                        className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${
                          attr.attribution_tag.includes("Synergy")
                            ? "bg-violet-950/70 text-violet-300 border-violet-500/40"
                            : attr.attribution_tag.includes("Semantic")
                            ? "bg-cyan-950/70 text-cyan-300 border-cyan-500/40"
                            : attr.attribution_tag.includes("Gem")
                            ? "bg-amber-950/70 text-amber-300 border-amber-500/40"
                            : "bg-emerald-950/70 text-emerald-300 border-emerald-500/40"
                        }`}
                      >
                        <Award className="w-3.5 h-3.5" /> {attr.attribution_tag}
                      </span>
                    </div>
                  )}

                  <button
                    onClick={() => setSelectedExplanation(isSelected ? null : track)}
                    className="p-2 rounded-lg bg-obsidian-800 hover:bg-obsidian-700 text-slate-300 hover:text-white border border-obsidian-700 transition"
                    title="Score Attribution Details"
                  >
                    <Info className="w-4 h-4" />
                  </button>
                  
                  {/* 30-Second Audio Preview Player or Unavailable Flag */}
                  {track.has_audio_preview ? (
                    <div className="flex items-center gap-1.5 bg-obsidian-900 border border-obsidian-700 rounded-lg p-1">
                      <button
                        onClick={() => handlePreviewPlay(track)}
                        className={`p-1.5 rounded-md text-white transition ${
                          activePreviewTrackId === track.track_id && isPlayingPreview
                            ? "bg-amber-500 text-obsidian-950 font-bold shadow-md shadow-amber-500/30"
                            : "bg-violet-600 hover:bg-violet-500"
                        }`}
                        title={activePreviewTrackId === track.track_id && isPlayingPreview ? "Pause 30s Preview" : "Play 30s Audio Preview"}
                      >
                        {activePreviewTrackId === track.track_id && isPlayingPreview ? (
                          <Pause className="w-3.5 h-3.5 fill-current" />
                        ) : (
                          <Play className="w-3.5 h-3.5 fill-current" />
                        )}
                      </button>
                      <button
                        onClick={() => handleAudioReplay(track.track_id)}
                        className="p-1.5 rounded-md hover:bg-obsidian-800 text-slate-400 hover:text-white transition"
                        title="Replay 30s Preview"
                      >
                        <RotateCcw className="w-3 h-3" />
                      </button>
                      <span className="text-[10px] font-mono text-emerald-400 px-1 font-semibold">30s</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-1.5 text-[11px] font-mono text-slate-500 px-2 py-1.5 rounded-lg bg-obsidian-900/60 border border-obsidian-800">
                      <span>No audio</span>
                      <button
                        onClick={() => onPlayTrack(track)}
                        className="p-1 rounded bg-obsidian-800 hover:bg-obsidian-700 text-slate-300"
                        title="Synthesize Acoustic Tone Proxy (No Audio Preview Available)"
                      >
                        <Play className="w-3 h-3 fill-current" />
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Traceable Explanation Card (Accordion) */}
              {isSelected && attr && (
                <div className="mt-4 pt-4 border-t border-obsidian-700/80 bg-obsidian-900/60 -mx-4 -mb-4 p-4 rounded-b-xl space-y-3">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <span className="text-xs font-bold text-violet-400 font-mono uppercase tracking-wider block mb-1">
                        Ranking Score Attribution (Mathematical Decomposition, Not Causal Taste)
                      </span>
                      <p className="text-sm text-slate-200 leading-relaxed font-sans">
                        {attr.rationale}
                      </p>
                    </div>

                    <a
                      href={track.lastfm_url}
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center gap-1 text-xs text-slate-400 hover:text-violet-300 whitespace-nowrap"
                    >
                      Last.fm <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>

                  {/* Component Breakdown Bars */}
                  <div className="grid grid-cols-3 gap-3 font-mono text-xs pt-2">
                    <div className="p-2 rounded-lg bg-obsidian-950 border border-obsidian-800">
                      <span className="text-slate-400 block mb-1">ALS COLLABORATIVE</span>
                      <span className="font-bold text-violet-300">{attr.als_component > 0 ? `+${attr.als_component}` : attr.als_component}</span>
                    </div>
                    <div className="p-2 rounded-lg bg-obsidian-950 border border-obsidian-800">
                      <span className="text-slate-400 block mb-1">TWO-TOWER SEMANTIC</span>
                      <span className="font-bold text-cyan-300">{attr.tt_component > 0 ? `+${attr.tt_component}` : attr.tt_component}</span>
                    </div>
                    <div className="p-2 rounded-lg bg-obsidian-950 border border-obsidian-800">
                      <span className="text-slate-400 block mb-1">NOVELTY SURPRISAL</span>
                      <span className="font-bold text-amber-300">+{attr.novelty_component}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Listener Discovery Feedback Action Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-2 pt-3 mt-3 border-t border-obsidian-800/80 text-xs">
                <div className="flex items-center gap-1.5 text-slate-400 font-mono text-[11px]">
                  <span>FEEDBACK:</span>
                  {feedbackMap[track.track_id] ? (
                    <span className="text-amber-400 font-bold uppercase flex items-center gap-1">
                      <Check className="w-3 h-3" /> {feedbackMap[track.track_id].replace(/_/g, " ")}
                    </span>
                  ) : (
                    <span className="text-slate-500 italic">Rate discovery signal</span>
                  )}
                </div>

                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => handleFeedback(track.track_id, "discovered_something_new", idx + 1)}
                    className={`px-2.5 py-1 rounded-lg flex items-center gap-1 text-[11px] font-semibold transition border ${
                      feedbackMap[track.track_id] === "discovered_something_new"
                        ? "bg-amber-500 text-obsidian-950 border-amber-400 shadow-md shadow-amber-500/30"
                        : "bg-obsidian-900 hover:bg-amber-500/20 text-amber-300 border-amber-500/30"
                    }`}
                    title="Discovered Something New (Serendipity Signal)"
                  >
                    <Sparkles className="w-3 h-3" /> Discovered Something New
                  </button>

                  <button
                    onClick={() => handleFeedback(track.track_id, "like", idx + 1)}
                    className={`p-1.5 rounded-lg border transition ${
                      feedbackMap[track.track_id] === "like"
                        ? "bg-violet-600 text-white border-violet-500"
                        : "bg-obsidian-900 hover:bg-obsidian-800 text-slate-400 hover:text-white border-obsidian-700"
                    }`}
                    title="Like this track"
                  >
                    <ThumbsUp className="w-3.5 h-3.5" />
                  </button>

                  <button
                    onClick={() => handleFeedback(track.track_id, "already_know", idx + 1)}
                    className={`px-2 py-1 rounded-lg flex items-center gap-1 text-[11px] border transition ${
                      feedbackMap[track.track_id] === "already_know"
                        ? "bg-cyan-600 text-white border-cyan-500"
                        : "bg-obsidian-900 hover:bg-obsidian-800 text-slate-400 hover:text-cyan-300 border-obsidian-700"
                    }`}
                    title="Already Know This Track (Familiarity Signal)"
                  >
                    <Eye className="w-3 h-3" /> Already Know
                  </button>

                  <button
                    onClick={() => handleFeedback(track.track_id, "not_my_taste", idx + 1)}
                    className={`p-1.5 rounded-lg border transition ${
                      feedbackMap[track.track_id] === "not_my_taste"
                        ? "bg-rose-900 text-rose-200 border-rose-700"
                        : "bg-obsidian-900 hover:bg-obsidian-800 text-slate-400 hover:text-rose-400 border-obsidian-700"
                    }`}
                    title="Not My Taste"
                  >
                    <ThumbsDown className="w-3.5 h-3.5" />
                  </button>

                  <button
                    onClick={() => handleFeedback(track.track_id, "save_for_later", idx + 1)}
                    className={`p-1.5 rounded-lg border transition ${
                      feedbackMap[track.track_id] === "save_for_later"
                        ? "bg-emerald-600 text-white border-emerald-500"
                        : "bg-obsidian-900 hover:bg-obsidian-800 text-slate-400 hover:text-emerald-300 border-obsidian-700"
                    }`}
                    title="Save For Later"
                  >
                    <Bookmark className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Hidden Audio Element for 30s Previews */}
      <audio ref={audioRef} onEnded={handleAudioEnded} className="hidden" preload="none" />

      {/* Phase 7B Qualitative Post-Slate Assessment */}
      <div className="glass-panel p-6 rounded-2xl border border-violet-500/20 space-y-4">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-violet-400" />
          <h3 className="text-lg font-bold text-white">Qualitative Slate Assessment (Phase 7B)</h3>
        </div>
        <p className="text-sm text-slate-400">
          What did this recommendation experience do particularly well or poorly?
        </p>
        {qualitativeSubmitted ? (
          <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-500/40 text-emerald-300 text-sm flex items-center gap-2">
            <Check className="w-4 h-4 text-emerald-400" />
            Your qualitative feedback has been recorded for this experimental slate. Thank you!
          </div>
        ) : (
          <form onSubmit={handleQualitativeSubmit} className="space-y-3">
            <textarea
              value={qualitativeText}
              onChange={(e) => setQualitativeText(e.target.value)}
              placeholder="e.g. Discovered an unfamiliar ambient post-rock artist that fit my taste perfectly, though track #4 felt slightly too abrasive..."
              className="w-full h-24 p-3 rounded-xl bg-obsidian-900 border border-obsidian-700 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-violet-500 resize-none font-sans"
            />
            <button
              type="submit"
              disabled={!qualitativeText.trim()}
              className="px-4 py-2 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white text-xs font-semibold tracking-wide transition shadow-md shadow-violet-600/30"
            >
              Submit Qualitative Response
            </button>
          </form>
        )}
      </div>
    </div>
  );
};
