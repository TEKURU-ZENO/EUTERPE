import React, { useEffect, useRef, useState } from "react";
import { Orbit, ZoomIn, ZoomOut, RotateCcw, Play, Info } from "lucide-react";
import { Track, UniversePoint } from "../types";

interface MusicUniverseViewProps {
  points: UniversePoint[];
  loading: boolean;
  onPlayTrack: (track: Track) => void;
  onRefresh: () => void;
}

export const MusicUniverseView: React.FC<MusicUniverseViewProps> = ({
  points,
  loading,
  onPlayTrack,
  onRefresh,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [hoveredPoint, setHoveredPoint] = useState<UniversePoint | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    drawUniverse();
  }, [points, zoom, pan, hoveredPoint]);

  const drawUniverse = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background cosmic grid
    ctx.strokeStyle = "rgba(139, 92, 246, 0.05)";
    ctx.lineWidth = 1;
    const gridSize = 40 * zoom;
    const startX = (pan.x % gridSize);
    const startY = (pan.y % gridSize);

    for (let x = startX; x < canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = startY; y < canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    if (!points.length) return;

    // Normalization bounds
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);

    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const scale = Math.min(canvas.width, canvas.height) * 0.38 * zoom;

    // Draw tracks
    points.forEach((p) => {
      // Map [-range, range] to canvas coords
      const nx = (p.x - (minX + maxX) / 2) / (Math.max(1e-5, (maxX - minX) / 2));
      const ny = (p.y - (minY + maxY) / 2) / (Math.max(1e-5, (maxY - minY) / 2));

      const screenX = cx + nx * scale + pan.x;
      const screenY = cy + ny * scale + pan.y;

      const isHovered = hoveredPoint?.track_id === p.track_id;

      if (p.is_recommended) {
        // Glowing Gold Halo
        ctx.fillStyle = isHovered ? "#FBBF24" : "#F59E0B";
        ctx.shadowColor = "#F59E0B";
        ctx.shadowBlur = isHovered ? 15 : 8;
        ctx.beginPath();
        ctx.arc(screenX, screenY, isHovered ? 7 : 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.shadowBlur = 0;
      } else if (p.is_head_decile) {
        // Violet head cluster
        ctx.fillStyle = isHovered ? "#C4B5FD" : "#8B5CF6";
        ctx.shadowColor = "#8B5CF6";
        ctx.shadowBlur = isHovered ? 10 : 3;
        ctx.beginPath();
        ctx.arc(screenX, screenY, isHovered ? 5 : 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.shadowBlur = 0;
      } else {
        // Cyan long-tail discovery stars
        ctx.fillStyle = isHovered ? "#67E8F9" : "rgba(6, 182, 212, 0.6)";
        ctx.beginPath();
        ctx.arc(screenX, screenY, isHovered ? 4.5 : 2.5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStartRef.current.x,
        y: e.clientY - dragStartRef.current.y,
      });
      return;
    }

    // Hover detection
    const canvas = canvasRef.current;
    if (!canvas || !points.length) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);

    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const scale = Math.min(canvas.width, canvas.height) * 0.38 * zoom;

    let found: UniversePoint | null = null;
    for (const p of points) {
      const nx = (p.x - (minX + maxX) / 2) / (Math.max(1e-5, (maxX - minX) / 2));
      const ny = (p.y - (minY + maxY) / 2) / (Math.max(1e-5, (maxY - minY) / 2));
      const screenX = cx + nx * scale + pan.x;
      const screenY = cy + ny * scale + pan.y;

      const dist = Math.hypot(mouseX - screenX, mouseY - screenY);
      if (dist < 10) {
        found = p;
        break;
      }
    }
    setHoveredPoint(found);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleCanvasClick = () => {
    if (hoveredPoint) {
      onPlayTrack({
        track_id: hoveredPoint.track_id,
        recording_key: `${hoveredPoint.artist}:::${hoveredPoint.title}`,
        title: hoveredPoint.title,
        artist: hoveredPoint.artist,
        artist_id: 0,
        playcount: 10,
        novelty_bits: hoveredPoint.novelty_bits,
        is_head_decile: hoveredPoint.is_head_decile,
        lastfm_url: `https://www.last.fm/music/${hoveredPoint.artist.replace(/ /g, "+")}/_/${hoveredPoint.title.replace(/ /g, "+")}`,
        cover_art_url: hoveredPoint.cover_art_url,
        energy: hoveredPoint.energy,
        valence: hoveredPoint.valence,
        danceability: 0.5,
        acousticness: 0.5,
        tempo_bpm: 120,
        provenance: "Last.fm 1K + MusicBrainz Open Catalog",
        license: "Academic Research / CC BY-NC-SA 3.0",
        preview_status: "available_simulated",
      });
    }
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Top Banner */}
      <div className="glass-panel p-6 rounded-2xl relative overflow-hidden border border-violet-500/20">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-violet-600/20 text-violet-300 border border-violet-500/30 flex items-center gap-1.5">
                <Orbit className="w-3.5 h-3.5 text-cyan-400" /> 64D Neural Latent Projection
              </span>
              <span className="text-xs text-slate-400 font-mono">
                {points.length} Tracks in Viewport
              </span>
            </div>
            <h1 className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              Music Universe Explorer
            </h1>
            <p className="text-sm text-slate-400 mt-1">
              Low-rank PCA manifold of the Two-Tower neural embedding space. Interactive drag, zoom, and explore semantic clusters.
            </p>
          </div>

          {/* Navigation Controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setZoom((z) => Math.min(4.0, z * 1.3))}
              className="p-2.5 rounded-xl bg-obsidian-800 hover:bg-obsidian-700 text-slate-300 border border-obsidian-700"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={() => setZoom((z) => Math.max(0.4, z / 1.3))}
              className="p-2.5 rounded-xl bg-obsidian-800 hover:bg-obsidian-700 text-slate-300 border border-obsidian-700"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              onClick={() => {
                setZoom(1.0);
                setPan({ x: 0, y: 0 });
              }}
              className="p-2.5 rounded-xl bg-obsidian-800 hover:bg-obsidian-700 text-slate-300 border border-obsidian-700"
              title="Reset View"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-6 mt-4 pt-4 border-t border-obsidian-800 text-xs font-mono">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-amber-500 shadow-sm shadow-amber-500" />
            <span className="text-slate-300">Recommended For You</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-violet-500" />
            <span className="text-slate-300">Head / Popular Tracks</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-cyan-500" />
            <span className="text-slate-300">Long-Tail Discovery</span>
          </div>
        </div>
      </div>

      {/* Canvas Viewport */}
      <div className="relative rounded-2xl overflow-hidden glass-panel border border-obsidian-700 h-[520px]">
        {loading && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-obsidian-950/70 backdrop-blur-sm">
            <div className="w-10 h-10 border-4 border-violet-500/20 border-t-violet-500 rounded-full animate-spin" />
          </div>
        )}

        <canvas
          ref={canvasRef}
          width={900}
          height={520}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onClick={handleCanvasClick}
          className="w-full h-full cursor-grab active:cursor-grabbing"
        />

        {/* Tooltip Card */}
        {hoveredPoint && (
          <div className="absolute bottom-6 left-6 z-10 glass-panel p-4 rounded-xl border border-violet-500/40 shadow-xl max-w-sm pointer-events-none animate-fadeIn">
            <div className="flex items-center gap-3 mb-2">
              <img
                src={hoveredPoint.cover_art_url}
                alt={hoveredPoint.title}
                className="w-12 h-12 rounded-lg object-cover border border-obsidian-700"
                onError={(e) => {
                  (e.target as any).src = "https://api.dicebear.com/7.x/identicon/svg?seed=0";
                }}
              />
              <div className="min-w-0">
                <h4 className="font-bold text-white text-sm truncate">{hoveredPoint.title}</h4>
                <p className="text-xs text-slate-300 truncate">{hoveredPoint.artist}</p>
                <div className="flex items-center gap-2 text-[10px] text-slate-400 font-mono mt-0.5">
                  <span>Novelty: {hoveredPoint.novelty_bits.toFixed(1)} bits</span>
                  <span>Energy (Est): {(hoveredPoint.energy * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
            <p className="text-[11px] text-violet-300 font-medium">Click to synthesize acoustic tone sonification</p>
          </div>
        )}
      </div>
    </div>
  );
};
