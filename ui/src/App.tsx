import React, { useEffect, useState } from "react";
import {
  Sparkles,
  Flame,
  Orbit,
  Compass,
  Sliders,
  Music,
  Users,
  Shield,
  Layers,
  ExternalLink,
} from "lucide-react";
import { DiscoverView } from "./components/DiscoverView";
import { HiddenGemsView } from "./components/HiddenGemsView";
import { MusicUniverseView } from "./components/MusicUniverseView";
import { TasteProfileView } from "./components/TasteProfileView";
import { DiscoveryLabView } from "./components/DiscoveryLabView";
import { AudioSynthesizerModal } from "./components/AudioSynthesizerModal";
import {
  DiscoverResponse,
  LabResponse,
  TasteProfile,
  Track,
  UniversePoint,
} from "./types";

const API_BASE = "http://localhost:8000";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<
    "discover" | "gems" | "universe" | "taste" | "lab"
  >("discover");

  // User state
  const [userList, setUserList] = useState<Array<{ user_id: string; user_idx: number; cohort: string }>>([]);
  const [selectedUser, setSelectedUser] = useState<string>("user_000001");

  // Data states
  const [discoverData, setDiscoverData] = useState<DiscoverResponse | null>(null);
  const [gems, setGems] = useState<Track[]>([]);
  const [minGemsBits, setMinGemsBits] = useState<number>(16.0);
  const [universePoints, setUniversePoints] = useState<UniversePoint[]>([]);
  const [tasteProfile, setTasteProfile] = useState<TasteProfile | null>(null);
  const [labData, setLabData] = useState<LabResponse | null>(null);

  // Lab Slider States
  const [labAlpha, setLabAlpha] = useState<number>(0.70);
  const [labBeta, setLabBeta] = useState<number>(0.20);
  const [labMaxArtist, setLabMaxArtist] = useState<number>(2);

  // Loading states
  const [loading, setLoading] = useState<boolean>(true);

  // Audio preview modal
  const [previewTrack, setPreviewTrack] = useState<Track | null>(null);

  // 1. Load users on initial mount
  useEffect(() => {
    fetch(`${API_BASE}/api/users?limit=25`)
      .then((res) => res.json())
      .then((data) => {
        if (data.users && data.users.length > 0) {
          setUserList(data.users);
          setSelectedUser(data.users[0].user_id);
        }
      })
      .catch((err) => console.error("Failed to load users", err));
  }, []);

  // 2. Fetch data based on active tab and selected user
  useEffect(() => {
    fetchTabData();
  }, [activeTab, selectedUser]);

  const fetchTabData = async () => {
    setLoading(true);
    try {
      if (activeTab === "discover") {
        const res = await fetch(`${API_BASE}/api/discover?user_id=${selectedUser}&limit=20&alpha=0.7&beta=0.2&max_per_artist=2`);
        const data = await res.json();
        setDiscoverData(data);
      } else if (activeTab === "gems") {
        const res = await fetch(`${API_BASE}/api/hidden-gems?user_id=${selectedUser}&min_novelty_bits=${minGemsBits}&limit=18`);
        const data = await res.json();
        setGems(data.gems || []);
      } else if (activeTab === "universe") {
        const res = await fetch(`${API_BASE}/api/universe?limit=350&user_id=${selectedUser}`);
        const data = await res.json();
        setUniversePoints(data.points || []);
      } else if (activeTab === "taste") {
        const res = await fetch(`${API_BASE}/api/taste-profile?user_id=${selectedUser}`);
        const data = await res.json();
        setTasteProfile(data);
      } else if (activeTab === "lab") {
        fetchLabData(labAlpha, labBeta, labMaxArtist);
      }
    } catch (e) {
      console.error("API error", e);
    } finally {
      setLoading(false);
    }
  };

  const fetchLabData = async (a: number, b: number, m: number) => {
    try {
      const res = await fetch(
        `${API_BASE}/api/lab?user_id=${selectedUser}&alpha=${a}&beta=${b}&max_per_artist=${m}&limit=20`
      );
      const data = await res.json();
      setLabData(data);
    } catch (e) {
      console.error("Lab API error", e);
    }
  };

  const handleLabParamChange = (a: number, b: number, m: number) => {
    setLabAlpha(a);
    setLabBeta(b);
    setLabMaxArtist(m);
    fetchLabData(a, b, m);
  };

  const handleGemsBitsChange = (bits: number) => {
    setMinGemsBits(bits);
    fetch(`${API_BASE}/api/hidden-gems?user_id=${selectedUser}&min_novelty_bits=${bits}&limit=18`)
      .then((r) => r.json())
      .then((d) => setGems(d.gems || []));
  };

  return (
    <div className="min-h-screen bg-obsidian-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-40 bg-obsidian-950/80 backdrop-blur-xl border-b border-obsidian-800/80 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          {/* Logo & Identity */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-violet-600 via-violet-500 to-amber-500 flex items-center justify-center shadow-lg shadow-violet-600/30">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-xl tracking-tight text-white">EUTERPE</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-violet-950 text-violet-300 border border-violet-500/30">
                  v1.0 Discovery Intelligence
                </span>
              </div>
              <p className="text-xs text-slate-400 font-medium">Beyond what you listen to. Into what you'll love.</p>
            </div>
          </div>

          {/* User Profile Selector */}
          <div className="flex items-center gap-3 bg-obsidian-900/90 px-3 py-1.5 rounded-xl border border-obsidian-700">
            <Users className="w-4 h-4 text-violet-400" />
            <span className="text-xs text-slate-400 font-mono">LISTENER:</span>
            <select
              value={selectedUser}
              onChange={(e) => setSelectedUser(e.target.value)}
              className="bg-transparent text-sm font-semibold text-white focus:outline-none cursor-pointer font-mono"
            >
              {userList.map((u) => (
                <option key={u.user_id} value={u.user_id} className="bg-obsidian-900 text-white">
                  {u.user_id} ({u.cohort})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Surface Tab Navigation */}
        <div className="max-w-7xl mx-auto flex items-center gap-2 mt-4 pt-3 border-t border-obsidian-800/60 overflow-x-auto text-sm">
          <button
            onClick={() => setActiveTab("discover")}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl font-semibold transition ${
              activeTab === "discover"
                ? "bg-violet-600 text-white shadow-md shadow-violet-600/30"
                : "text-slate-400 hover:text-white hover:bg-obsidian-900"
            }`}
          >
            <Sparkles className="w-4 h-4 text-gold-400" /> Discover Slate
          </button>

          <button
            onClick={() => setActiveTab("gems")}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl font-semibold transition ${
              activeTab === "gems"
                ? "bg-amber-500 text-obsidian-950 font-bold shadow-md shadow-amber-500/30"
                : "text-slate-400 hover:text-white hover:bg-obsidian-900"
            }`}
          >
            <Flame className="w-4 h-4 text-amber-400" /> Hidden Gems
          </button>

          <button
            onClick={() => setActiveTab("universe")}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl font-semibold transition ${
              activeTab === "universe"
                ? "bg-cyan-600 text-white shadow-md shadow-cyan-600/30"
                : "text-slate-400 hover:text-white hover:bg-obsidian-900"
            }`}
          >
            <Orbit className="w-4 h-4 text-cyan-300" /> Music Universe
          </button>

          <button
            onClick={() => setActiveTab("taste")}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl font-semibold transition ${
              activeTab === "taste"
                ? "bg-violet-700 text-white shadow-md shadow-violet-700/30"
                : "text-slate-400 hover:text-white hover:bg-obsidian-900"
            }`}
          >
            <Compass className="w-4 h-4 text-violet-300" /> Taste Radar
          </button>

          <button
            onClick={() => setActiveTab("lab")}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl font-semibold transition ${
              activeTab === "lab"
                ? "bg-gold-500 text-obsidian-950 font-bold shadow-md shadow-gold-500/30"
                : "text-slate-400 hover:text-white hover:bg-obsidian-900"
            }`}
          >
            <Sliders className="w-4 h-4 text-gold-400" /> Discovery Lab
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6">
        {activeTab === "discover" && (
          <DiscoverView
            data={discoverData}
            loading={loading}
            onPlayTrack={(t) => setPreviewTrack(t)}
            onRefresh={fetchTabData}
          />
        )}

        {activeTab === "gems" && (
          <HiddenGemsView
            gems={gems}
            loading={loading}
            minBits={minGemsBits}
            onMinBitsChange={handleGemsBitsChange}
            onPlayTrack={(t) => setPreviewTrack(t)}
            onRefresh={fetchTabData}
          />
        )}

        {activeTab === "universe" && (
          <MusicUniverseView
            points={universePoints}
            loading={loading}
            onPlayTrack={(t) => setPreviewTrack(t)}
            onRefresh={fetchTabData}
          />
        )}

        {activeTab === "taste" && (
          <TasteProfileView
            profile={tasteProfile}
            loading={loading}
            onPlayTrack={(t) => setPreviewTrack(t)}
          />
        )}

        {activeTab === "lab" && (
          <DiscoveryLabView
            labData={labData}
            loading={loading}
            alpha={labAlpha}
            beta={labBeta}
            maxPerArtist={labMaxArtist}
            onAlphaChange={(a) => handleLabParamChange(a, labBeta, labMaxArtist)}
            onBetaChange={(b) => handleLabParamChange(labAlpha, b, labMaxArtist)}
            onMaxPerArtistChange={(m) => handleLabParamChange(labAlpha, labBeta, m)}
            onResetDefaults={() => handleLabParamChange(0.70, 0.20, 2)}
            onPlayTrack={(t) => setPreviewTrack(t)}
          />
        )}
      </main>

      {/* Audio Synthesizer Preview Modal */}
      <AudioSynthesizerModal track={previewTrack} onClose={() => setPreviewTrack(null)} />

      {/* Global Footer */}
      <footer className="mt-auto border-t border-obsidian-800 py-6 px-6 bg-obsidian-950 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-emerald-400" />
            <span>
              EUTERPE Scientific Milestone: Phase 4 Certified (103.4% ALS NDCG@20 Retention, p=0.0037). Phase 5 Product Architecture.
            </span>
          </div>

          <div className="flex items-center gap-4 text-slate-400">
            <span>Catalogue: 443,375 Tracks</span>
            <span>•</span>
            <span>Embeddings: 64D Normalized</span>
            <span>•</span>
            <a href="https://last.fm" target="_blank" rel="noreferrer" className="hover:text-white transition">
              Last.fm 1K Dataset
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
