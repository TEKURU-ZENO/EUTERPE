import React, { useEffect, useRef, useState } from "react";
import { Play, Pause, X, ExternalLink, Shield, Sparkles, Volume2 } from "lucide-react";
import { Track } from "../types";

interface AudioSynthesizerModalProps {
  track: Track | null;
  onClose: () => void;
}

export const AudioSynthesizerModal: React.FC<AudioSynthesizerModalProps> = ({ track, onClose }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const oscillatorRefs = useRef<OscillatorNode[]>([]);
  const gainNodeRef = useRef<GainNode | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);

  useEffect(() => {
    if (track) {
      startSynth();
    }
    return () => {
      stopSynth();
    };
  }, [track]);

  const startSynth = () => {
    if (!track) return;
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const ctx = new AudioCtx();
      audioCtxRef.current = ctx;

      const masterGain = ctx.createGain();
      masterGain.gain.setValueAtTime(0.18, ctx.currentTime);
      masterGain.connect(ctx.destination);
      gainNodeRef.current = masterGain;

      // Base root frequency from artist_id
      const rootNotes = [130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94]; // C3 to B3
      const root = rootNotes[track.artist_id % rootNotes.length];

      // Major chord if valence >= 0.5, minor if valence < 0.5
      const thirdMult = track.valence >= 0.5 ? 1.2599 : 1.1892;
      const fifthMult = 1.4983;

      const freqs = [root, root * thirdMult, root * fifthMult, root * 2];
      const oscs: OscillatorNode[] = [];

      freqs.forEach((f, idx) => {
        const osc = ctx.createOscillator();
        const oscGain = ctx.createGain();
        
        // Energy influences waveform type & filter
        osc.type = track.energy > 0.6 ? (idx === 0 ? "sawtooth" : "triangle") : "sine";
        osc.frequency.setValueAtTime(f, ctx.currentTime);

        // Lowpass filter
        const filter = ctx.createBiquadFilter();
        filter.type = "lowpass";
        filter.frequency.setValueAtTime(400 + track.energy * 2400, ctx.currentTime);

        oscGain.gain.setValueAtTime(0.2 / freqs.length, ctx.currentTime);
        osc.connect(filter);
        filter.connect(oscGain);
        oscGain.connect(masterGain);

        osc.start();
        oscs.push(osc);
      });

      oscillatorRefs.current = oscs;
      setIsPlaying(true);
      drawWaveform();
    } catch (e) {
      console.error("AudioContext initialization error", e);
    }
  };

  const stopSynth = () => {
    oscillatorRefs.current.forEach((osc) => {
      try {
        osc.stop();
        osc.disconnect();
      } catch (e) {}
    });
    oscillatorRefs.current = [];
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close();
    }
    audioCtxRef.current = null;
    setIsPlaying(false);
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
    }
  };

  const togglePlay = () => {
    if (isPlaying) {
      stopSynth();
    } else {
      startSynth();
    }
  };

  const drawWaveform = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let phase = 0;
    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;

      // Gradient stroke
      const grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
      grad.addColorStop(0, "#8B5CF6");
      grad.addColorStop(0.5, "#06B6D4");
      grad.addColorStop(1, "#F59E0B");
      ctx.strokeStyle = grad;

      ctx.beginPath();
      const sliceWidth = canvas.width / 80;
      let x = 0;

      for (let i = 0; i < 80; i++) {
        const energyAmp = track ? track.energy * 24 : 15;
        const y = canvas.height / 2 + Math.sin(phase + i * 0.2) * energyAmp;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }
      ctx.stroke();

      phase += 0.08 * (track ? track.tempo_bpm / 120 : 1);
      animFrameRef.current = requestAnimationFrame(render);
    };
    render();
  };

  if (!track) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-obsidian-950/80 backdrop-blur-md">
      <div className="relative w-full max-w-lg p-6 rounded-2xl glass-panel border border-violet-500/30 shadow-2xl shadow-violet-950/50">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white rounded-lg hover:bg-obsidian-800 transition"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-start gap-4 mb-6">
          <img
            src={track.cover_art_url}
            alt={track.title}
            className="w-20 h-20 rounded-xl object-cover border border-violet-500/20 shadow-md shadow-violet-950/40"
            onError={(e) => {
              (e.target as any).src = `https://api.dicebear.com/7.x/identicon/svg?seed=${track.artist_id}`;
            }}
          />
          <div className="flex-1 pr-6">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-violet-950/80 text-violet-300 border border-violet-500/30 mb-1.5">
              <Sparkles className="w-3 h-3 text-gold-400" /> Acoustic Sonification (Synthesized Tone)
            </span>
            <h3 className="text-lg font-bold text-white line-clamp-1">{track.title}</h3>
            <p className="text-sm text-slate-300 font-medium">{track.artist}</p>
          </div>
        </div>

        {/* Oscilloscope Canvas */}
        <div className="mb-6 p-4 rounded-xl bg-obsidian-900 border border-obsidian-700/60">
          <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1 font-mono uppercase tracking-wider">
            <span className="text-violet-400 font-semibold">Artist-Level Proxy Profile</span>
            <span>TEMPO: {track.tempo_bpm} BPM</span>
          </div>
          <div className="flex items-center justify-between text-xs text-slate-400 mb-2 font-mono">
            <span>ENERGY (EST): {(track.energy * 100).toFixed(0)}%</span>
            <span>VALENCE (EST): {(track.valence * 100).toFixed(0)}%</span>
            <span>DANCE (EST): {(track.danceability * 100).toFixed(0)}%</span>
          </div>
          <canvas ref={canvasRef} width={400} height={80} className="w-full h-20 rounded-lg bg-obsidian-950" />
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between mb-6">
          <button
            onClick={togglePlay}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold transition ${
              isPlaying
                ? "bg-amber-500 hover:bg-amber-600 text-obsidian-950"
                : "bg-violet-600 hover:bg-violet-500 text-white shadow-lg shadow-violet-600/30"
            }`}
          >
            {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current" />}
            {isPlaying ? "Pause Sonification" : "Resume Sonification"}
          </button>

          <a
            href={track.lastfm_url}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-violet-300 transition"
          >
            Open on Last.fm <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        {/* Provenance & Licensing Disclosure */}
        <div className="pt-4 border-t border-obsidian-800 text-[11px] text-slate-400 space-y-1">
          <div className="flex items-center gap-1.5 text-slate-300 font-medium">
            <Shield className="w-3.5 h-3.5 text-emerald-400" /> Provenance & Sonification Disclosure
          </div>
          <p>
            Catalog: <span className="text-slate-300">{track.provenance}</span>. Temporal cutoff:{" "}
            <span className="text-slate-300 font-mono">2008-10-16 18:26 UTC</span>.
          </p>
          <p className="text-amber-300/80">
            <strong>Disclosure:</strong> Generated tone proxy based on artist-level statistical profile, NOT actual recorded track playback.
          </p>
        </div>
      </div>
    </div>
  );
};
