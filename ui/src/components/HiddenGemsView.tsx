import React, { useState } from "react";
import { Sparkles, Play, Shield, Sliders, ExternalLink, Flame, Info } from "lucide-react";
import { Track } from "../types";

interface HiddenGemsViewProps {
  gems: Track[];
  loading: boolean;
  minBits: number;
  onMinBitsChange: (bits: number) => void;
  onPlayTrack: (track: Track) => void;
  onRefresh: () => void;
}

export const HiddenGemsView: React.FC<HiddenGemsViewProps> = ({
  gems,
  loading,
  minBits,
  onMinBitsChange,
  onPlayTrack,
  onRefresh,
}) => {
  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl relative overflow-hidden border border-amber-500/20">
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-br from-amber-500/10 via-transparent to-transparent pointer-events-none" />

        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30 flex items-center gap-1.5">
                <Flame className="w-3.5 h-3.5 text-amber-400" /> Long-Tail Surprisal Stream
              </span>
              <span className="text-xs text-slate-400 font-mono">
                Threshold: &gt; {minBits.toFixed(1)} bits
              </span>
            </div>
            <h1 className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              Hidden Gems & Long-Tail Treasures
            </h1>
            <p className="text-sm text-slate-400 mt-1">
              Tracks with high statistical surprisal (-log₂ P(item) &gt; {minBits} bits) backed by personalized collaborative and semantic relevance safeguards.
            </p>
          </div>

          <button
            onClick={onRefresh}
            className="px-4 py-2 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 text-sm font-semibold text-amber-300 border border-amber-500/30 transition"
          >
            Mine New Gems
          </button>
        </div>

        {/* Interactive Surprisal Filter Slider */}
        <div className="pt-4 border-t border-obsidian-700/60 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Sliders className="w-4 h-4 text-amber-400" />
            <span className="text-xs font-mono text-slate-300 font-semibold">
              MINIMUM SURPRISAL: {minBits.toFixed(1)} BITS
            </span>
          </div>

          <div className="flex items-center gap-3 w-72">
            <span className="text-xs text-slate-500 font-mono">14.0</span>
            <input
              type="range"
              min="14.0"
              max="22.0"
              step="0.5"
              value={minBits}
              onChange={(e) => onMinBitsChange(parseFloat(e.target.value))}
              className="w-full accent-amber-500 bg-obsidian-800 rounded-lg cursor-pointer"
            />
            <span className="text-xs text-slate-500 font-mono">22.0</span>
          </div>
        </div>
      </div>

      {/* Gems Grid */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-24 space-y-4">
          <div className="w-12 h-12 border-4 border-amber-500/20 border-t-amber-500 rounded-full animate-spin" />
          <p className="text-slate-400 font-medium">Filtering long-tail catalogue for verified hidden gems...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {gems.map((track) => (
            <div
              key={track.track_id}
              className="glass-panel p-5 rounded-2xl border border-obsidian-700 hover:border-amber-500/40 hover:bg-obsidian-800/80 transition duration-200 flex flex-col justify-between group"
            >
              <div>
                <div className="flex items-start gap-4 mb-4">
                  <div className="relative cursor-pointer" onClick={() => onPlayTrack(track)}>
                    <img
                      src={track.cover_art_url}
                      alt={track.title}
                      className="w-16 h-16 rounded-xl object-cover border border-obsidian-700 shadow-md group-hover:opacity-80 transition"
                      onError={(e) => {
                        (e.target as any).src = `https://api.dicebear.com/7.x/identicon/svg?seed=${track.artist_id}`;
                      }}
                    />
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition">
                      <Play className="w-6 h-6 text-amber-400 fill-amber-400 drop-shadow-md" />
                    </div>
                  </div>

                  <div className="flex-1 min-w-0">
                    <span className="inline-block px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30 mb-1">
                      {track.gem_tag || `${track.novelty_bits.toFixed(1)} Bits Novelty`}
                    </span>
                    <h3 className="font-bold text-white text-base truncate hover:text-amber-300 transition cursor-pointer" onClick={() => onPlayTrack(track)}>
                      {track.title}
                    </h3>
                    <p className="text-sm text-slate-300 truncate font-medium">{track.artist}</p>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-obsidian-900 border border-obsidian-800 space-y-1.5 font-mono text-xs mb-4">
                  <div className="flex justify-between text-slate-400">
                    <span>Novelty Surprisal:</span>
                    <span className="text-amber-400 font-bold">{track.novelty_bits.toFixed(2)} bits</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Training Plays:</span>
                    <span className="text-slate-200">{track.playcount} scrobbles</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Acoustic Energy (Est):</span>
                    <span className="text-cyan-400">{(track.energy * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between pt-3 border-t border-obsidian-800 text-xs">
                <a
                  href={track.lastfm_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-slate-400 hover:text-amber-300 flex items-center gap-1 transition"
                >
                  Last.fm Profile <ExternalLink className="w-3.5 h-3.5" />
                </a>

                <button
                  onClick={() => onPlayTrack(track)}
                  className="px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500 text-amber-300 hover:text-obsidian-950 font-semibold border border-amber-500/30 transition flex items-center gap-1.5"
                  title="Acoustic Sonification"
                >
                  <Play className="w-3.5 h-3.5 fill-current" /> Sonify
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
