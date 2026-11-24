import React from "react";
import { Sliders, Sparkles, Play, Award, RotateCcw, Info, ArrowUpRight, Shield } from "lucide-react";
import { LabResponse, Track } from "../types";

interface DiscoveryLabViewProps {
  labData: LabResponse | null;
  loading: boolean;
  alpha: number;
  beta: number;
  maxPerArtist: number;
  onAlphaChange: (val: number) => void;
  onBetaChange: (val: number) => void;
  onMaxPerArtistChange: (val: number) => void;
  onResetDefaults: () => void;
  onPlayTrack: (track: Track) => void;
}

export const DiscoveryLabView: React.FC<DiscoveryLabViewProps> = ({
  labData,
  loading,
  alpha,
  beta,
  maxPerArtist,
  onAlphaChange,
  onBetaChange,
  onMaxPerArtistChange,
  onResetDefaults,
  onPlayTrack,
}) => {
  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Banner */}
      <div className="glass-panel p-6 rounded-2xl relative overflow-hidden border border-gold-500/20">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-gold-400" /> Algorithmic Sandbox
              </span>
              <span className="text-xs text-slate-400 font-mono">
                Real-Time Inference: {labData?.latency_ms || 28} ms
              </span>
            </div>
            <h1 className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              Discovery Lab & Parameter Sandbox
            </h1>
            <p className="text-sm text-slate-400 mt-1">
              Directly manipulate the multi-objective objective function in real time and observe live Pareto trade-offs.
            </p>
          </div>

          <button
            onClick={onResetDefaults}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-obsidian-800 hover:bg-obsidian-700 text-sm font-semibold text-slate-300 border border-obsidian-700 transition"
          >
            <RotateCcw className="w-4 h-4" /> Reset Phase 4 Optimal (α=0.7, β=0.2, C=2)
          </button>
        </div>

        {/* 3 Interactive Sliders */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 border-t border-obsidian-700/60">
          {/* Alpha Slider: Collaborative vs Neural */}
          <div className="space-y-2 p-4 rounded-xl bg-obsidian-900 border border-obsidian-800">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-slate-300 font-semibold">ALPHA (MODEL FUSION)</span>
              <span className="text-violet-400 font-bold">{alpha.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={alpha}
              onChange={(e) => onAlphaChange(parseFloat(e.target.value))}
              className="w-full accent-violet-500 bg-obsidian-950 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[11px] text-slate-500 font-mono">
              <span>0.0 (Neural TT)</span>
              <span>0.70 (Optimal)</span>
              <span>1.0 (ALS)</span>
            </div>
          </div>

          {/* Beta Slider: Novelty Weight */}
          <div className="space-y-2 p-4 rounded-xl bg-obsidian-900 border border-obsidian-800">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-slate-300 font-semibold">BETA (NOVELTY SURPRISAL)</span>
              <span className="text-amber-400 font-bold">{beta.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={beta}
              onChange={(e) => onBetaChange(parseFloat(e.target.value))}
              className="w-full accent-amber-500 bg-obsidian-950 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[11px] text-slate-500 font-mono">
              <span>0.0 (Relevance)</span>
              <span>0.20 (Optimal)</span>
              <span>1.0 (Max Novelty)</span>
            </div>
          </div>

          {/* Artist Cap Slider */}
          <div className="space-y-2 p-4 rounded-xl bg-obsidian-900 border border-obsidian-800">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-slate-300 font-semibold">MAX TRACKS PER ARTIST</span>
              <span className="text-emerald-400 font-bold">{maxPerArtist}</span>
            </div>
            <input
              type="range"
              min="1"
              max="5"
              step="1"
              value={maxPerArtist}
              onChange={(e) => onMaxPerArtistChange(parseInt(e.target.value))}
              className="w-full accent-emerald-500 bg-obsidian-950 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[11px] text-slate-500 font-mono">
              <span>1 (Max Variety)</span>
              <span>2 (Optimal)</span>
              <span>5 (Permissive)</span>
            </div>
          </div>
        </div>

        {/* Live Delta Metrics against Raw ALS Baseline */}
        {labData && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-4 border-t border-obsidian-700/60 font-mono text-xs">
            <div className="p-3.5 rounded-xl bg-obsidian-900 border border-obsidian-800">
              <span className="text-slate-400 block mb-1">NOVELTY SURPRISAL</span>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-white">{labData.live_deltas.novelty_bits} bits</span>
                <span className={`text-xs font-semibold ${labData.live_deltas.delta_novelty >= 0 ? "text-amber-400" : "text-rose-400"}`}>
                  {labData.live_deltas.delta_novelty >= 0 ? `+${labData.live_deltas.delta_novelty}` : labData.live_deltas.delta_novelty}
                </span>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-obsidian-900 border border-obsidian-800">
              <span className="text-slate-400 block mb-1">UNIQUE ARTISTS</span>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-white">{labData.live_deltas.unique_artists} artists</span>
                <span className={`text-xs font-semibold ${labData.live_deltas.delta_artists >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {labData.live_deltas.delta_artists >= 0 ? `+${labData.live_deltas.delta_artists}` : labData.live_deltas.delta_artists}
                </span>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-obsidian-900 border border-obsidian-800">
              <span className="text-slate-400 block mb-1">ARTIST CAP INTERVENTIONS</span>
              <span className="text-base font-bold text-cyan-400">
                {labData.live_deltas.artist_cap_interventions} tracks shifted
              </span>
            </div>

            <div className="p-3.5 rounded-xl bg-obsidian-900 border border-obsidian-800">
              <span className="text-slate-400 block mb-1">OBJECTIVE FUNCTION</span>
              <span className="text-[11px] font-mono text-slate-300 truncate block">
                S = {alpha}·ALS + {(1-alpha).toFixed(2)}·TT + {beta}·Nov
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Live Reranked Slates */}
      <div className="space-y-3">
        <h3 className="text-lg font-bold text-white">Live Reranked Output ({labData?.recommendations.length || 0} Tracks)</h3>

        {labData?.recommendations.map((track, idx) => {
          const attr = track.attribution;
          return (
            <div
              key={track.track_id}
              className="glass-panel p-4 rounded-xl border border-obsidian-800 hover:border-gold-500/40 hover:bg-obsidian-800/80 transition flex items-center justify-between gap-4"
            >
              <div className="flex items-center gap-4 min-w-0 flex-1">
                <span className="text-sm font-bold font-mono text-slate-500 w-6 text-center">
                  #{idx + 1}
                </span>

                <img
                  src={track.cover_art_url}
                  alt={track.title}
                  className="w-12 h-12 rounded-lg object-cover border border-obsidian-700"
                  onError={(e) => {
                    (e.target as any).src = `https://api.dicebear.com/7.x/identicon/svg?seed=${track.artist_id}`;
                  }}
                />

                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <h4 className="font-bold text-white text-sm truncate">{track.title}</h4>
                    {attr?.artist_cap_triggered && (
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                        Cap Triggered
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 truncate">{track.artist}</p>
                </div>
              </div>

              {attr && (
                <div className="hidden md:flex items-center gap-6 font-mono text-xs">
                  <div>
                    <span className="text-slate-500 block text-[10px]">TOTAL SCORE</span>
                    <span className="text-white font-bold">{attr.total_score}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">NOVELTY</span>
                    <span className="text-amber-400 font-bold">{attr.novelty_bits.toFixed(1)}b</span>
                  </div>
                  <span className="px-2.5 py-1 rounded-full text-[11px] font-medium bg-obsidian-900 border border-obsidian-700 text-slate-300">
                    {attr.attribution_tag}
                  </span>
                </div>
              )}

              <button
                onClick={() => onPlayTrack(track)}
                className="p-2 rounded-lg bg-obsidian-800 hover:bg-amber-500 text-slate-300 hover:text-obsidian-950 border border-obsidian-700 transition"
                title="Play Preview"
              >
                <Play className="w-4 h-4 fill-current" />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
