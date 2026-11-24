import React from "react";
import { Compass, Music, Flame, Award, Shield, Play } from "lucide-react";
import { TasteProfile, Track } from "../types";

interface TasteProfileViewProps {
  profile: TasteProfile | null;
  loading: boolean;
  onPlayTrack: (track: Track) => void;
}

export const TasteProfileView: React.FC<TasteProfileViewProps> = ({
  profile,
  loading,
  onPlayTrack,
}) => {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <div className="w-12 h-12 border-4 border-violet-500/20 border-t-violet-500 rounded-full animate-spin" />
        <p className="text-slate-400 font-medium">Auditing listener historical scrobbles and entropy...</p>
      </div>
    );
  }

  if (!profile) return null;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl relative overflow-hidden border border-violet-500/20">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-violet-600/20 text-violet-300 border border-violet-500/30 flex items-center gap-1.5">
                <Compass className="w-3.5 h-3.5 text-gold-400" /> Transparent Taste Analysis
              </span>
              <span className="text-xs text-slate-400 font-mono">
                {profile.total_listens.toLocaleString()} Scrobbles Audited
              </span>
            </div>
            <h1 className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              Listener Discovery Radar: {profile.user_id}
            </h1>
            <p className="text-sm text-slate-400 mt-1">
              Deterministic taste profiling evaluating long-tail openness, mainstream dependency, Shannon artist diversity, and acoustic preference.
            </p>
          </div>
        </div>

        {/* 3 Core Metric Meters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 border-t border-obsidian-800">
          {/* Openness Meter */}
          <div className="p-5 rounded-2xl bg-obsidian-900 border border-obsidian-800 space-y-3">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-slate-400 uppercase">Long-Tail Openness</span>
              <span className="font-bold text-amber-400 text-sm">{profile.openness_score}%</span>
            </div>
            <div className="w-full bg-obsidian-950 h-3 rounded-full overflow-hidden p-0.5 border border-obsidian-800">
              <div
                className="bg-gradient-to-r from-amber-500 to-amber-300 h-full rounded-full transition-all duration-500"
                style={{ width: `${profile.openness_score}%` }}
              />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Average listening self-information: <strong className="text-slate-200">{profile.mean_novelty_bits} bits</strong>. {profile.openness_score > 60 ? "Deep exploratory listener." : "Balanced mainstream orientation."}
            </p>
          </div>

          {/* Mainstream Index */}
          <div className="p-5 rounded-2xl bg-obsidian-900 border border-obsidian-800 space-y-3">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-slate-400 uppercase">Mainstream Play Share</span>
              <span className="font-bold text-violet-400 text-sm">{profile.mainstream_index}%</span>
            </div>
            <div className="w-full bg-obsidian-950 h-3 rounded-full overflow-hidden p-0.5 border border-obsidian-800">
              <div
                className="bg-gradient-to-r from-violet-600 to-violet-400 h-full rounded-full transition-all duration-500"
                style={{ width: `${profile.mainstream_index}%` }}
              />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Percentage of historical scrobbles in the top 10% most popular catalog decile.
            </p>
          </div>

          {/* Artist Diversity (Shannon Entropy) */}
          <div className="p-5 rounded-2xl bg-obsidian-900 border border-obsidian-800 space-y-3">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-slate-400 uppercase">Artist Diversity (Entropy)</span>
              <span className="font-bold text-cyan-400 text-sm">{profile.artist_diversity_score}%</span>
            </div>
            <div className="w-full bg-obsidian-950 h-3 rounded-full overflow-hidden p-0.5 border border-obsidian-800">
              <div
                className="bg-gradient-to-r from-cyan-500 to-cyan-300 h-full rounded-full transition-all duration-500"
                style={{ width: `${profile.artist_diversity_score}%` }}
              />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Listening spans <strong className="text-slate-200">{profile.unique_artists} unique artists</strong> across the catalogue.
            </p>
          </div>
        </div>
      </div>

      {/* Acoustic Characteristics Radar & Top Artists */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Acoustic Radar Bars */}
        <div className="glass-panel p-6 rounded-2xl border border-obsidian-800 space-y-5">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Music className="w-5 h-5 text-violet-400" /> Acoustic Profile Signature (Artist-Level Estimates)
          </h3>
          <p className="text-xs text-slate-400">
            Calibrated acoustic dimensions derived from artist-level historical profiles (proxy estimates, not direct track audio waveform analysis).
          </p>

          <div className="space-y-4 pt-2">
            <div>
              <div className="flex justify-between text-xs font-mono mb-1.5">
                <span className="text-slate-400">ENERGY (RHYTHM & INTENSITY)</span>
                <span className="text-white font-bold">{(profile.acoustic_radar.energy * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-obsidian-900 h-2.5 rounded-full overflow-hidden">
                <div className="bg-violet-500 h-full rounded-full" style={{ width: `${profile.acoustic_radar.energy * 100}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1.5">
                <span className="text-slate-400">VALENCE (MUSICAL POSITIVENESS)</span>
                <span className="text-white font-bold">{(profile.acoustic_radar.valence * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-obsidian-900 h-2.5 rounded-full overflow-hidden">
                <div className="bg-amber-500 h-full rounded-full" style={{ width: `${profile.acoustic_radar.valence * 100}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1.5">
                <span className="text-slate-400">DANCEABILITY (BEAT CONSISTENCY)</span>
                <span className="text-white font-bold">{(profile.acoustic_radar.danceability * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-obsidian-900 h-2.5 rounded-full overflow-hidden">
                <div className="bg-cyan-500 h-full rounded-full" style={{ width: `${profile.acoustic_radar.danceability * 100}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1.5">
                <span className="text-slate-400">ACOUSTICNESS (NATURAL TIMBRE)</span>
                <span className="text-white font-bold">{(profile.acoustic_radar.acousticness * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-obsidian-900 h-2.5 rounded-full overflow-hidden">
                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${profile.acoustic_radar.acousticness * 100}%` }} />
              </div>
            </div>
          </div>
        </div>

        {/* Top Historical Artists */}
        <div className="glass-panel p-6 rounded-2xl border border-obsidian-800 space-y-4">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Award className="w-5 h-5 text-gold-400" /> Primary Influence Artists
          </h3>
          <p className="text-xs text-slate-400">
            Top artists in historical training scrobbles powering collaborative factor projection.
          </p>

          <div className="space-y-3 pt-2">
            {profile.top_artists.map((artist, idx) => (
              <div
                key={artist.artist_id}
                className="flex items-center justify-between p-3.5 rounded-xl bg-obsidian-900/80 border border-obsidian-800 hover:border-violet-500/30 transition"
              >
                <div className="flex items-center gap-3">
                  <span className="text-sm font-bold font-mono text-slate-500 w-5">#{idx + 1}</span>
                  <span className="text-sm font-bold text-white">{artist.artist_name}</span>
                </div>
                <span className="text-xs font-mono text-amber-400 font-semibold">
                  {artist.plays.toLocaleString()} plays
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Historical Listening Sample */}
      <div className="glass-panel p-6 rounded-2xl border border-obsidian-800 space-y-4">
        <h3 className="text-lg font-bold text-white">Representative Training History</h3>
        <p className="text-xs text-slate-400">
          Tracks consumed prior to evaluation split cutoff (strictly held out from future validation).
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
          {profile.history_sample.map((t) => (
            <div
              key={t.track_id}
              className="p-3.5 rounded-xl bg-obsidian-900 border border-obsidian-800 flex items-center gap-3 group hover:border-violet-500/40 transition cursor-pointer"
              onClick={() => onPlayTrack(t)}
            >
              <img
                src={t.cover_art_url}
                alt={t.title}
                className="w-12 h-12 rounded-lg object-cover border border-obsidian-700"
                onError={(e) => {
                  (e.target as any).src = `https://api.dicebear.com/7.x/identicon/svg?seed=${t.artist_id}`;
                }}
              />
              <div className="min-w-0 flex-1">
                <h5 className="text-sm font-bold text-white truncate group-hover:text-violet-300 transition">
                  {t.title}
                </h5>
                <p className="text-xs text-slate-400 truncate">{t.artist}</p>
              </div>
              <Play className="w-4 h-4 text-violet-400 opacity-0 group-hover:opacity-100 transition" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
