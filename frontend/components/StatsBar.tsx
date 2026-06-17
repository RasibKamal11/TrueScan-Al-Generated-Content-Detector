"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, FileText, Image as ImageIcon, Video, Link as LinkIcon, Clock, Zap } from "lucide-react";

interface Stats {
  scans: {
    text: number;
    image: number;
    video: number;
    url: number;
    total: number;
  };
  uptime: string;
  models_loaded: {
    text: boolean;
    image: boolean;
    video: boolean;
  };
}

export default function StatsBar() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [online, setOnline] = useState(false);
  const [pulse, setPulse] = useState(false);

  const fetchStats = async () => {
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${apiBase}/detect/stats`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const data = await res.json();
        setStats(data);
        setOnline(true);
        setPulse(true);
        setTimeout(() => setPulse(false), 600);
      } else {
        setOnline(false);
      }
    } catch {
      setOnline(false);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 15000); // refresh every 15s
    return () => clearInterval(interval);
  }, []);

  if (!online || !stats) {
    return (
      <div className="flex items-center space-x-2 text-xs text-slate-600 py-2 px-4 rounded-xl bg-slate-900/40 border border-white/5">
        <span className="w-1.5 h-1.5 rounded-full bg-slate-600 animate-pulse" />
        <span>Backend offline</span>
      </div>
    );
  }

  const statItems = [
    { icon: FileText, label: "Text", value: stats.scans.text, color: "text-blue-400" },
    { icon: ImageIcon, label: "Image", value: stats.scans.image, color: "text-violet-400" },
    { icon: Video, label: "Video", value: stats.scans.video, color: "text-cyan-400" },
    { icon: LinkIcon, label: "URL", value: stats.scans.url, color: "text-orange-400" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-wrap items-center gap-3 text-xs py-2.5 px-5 rounded-2xl bg-slate-900/60 border border-white/5 backdrop-blur-md"
    >
      {/* Online indicator */}
      <div className="flex items-center space-x-1.5">
        <motion.span
          animate={{ scale: pulse ? [1, 1.4, 1] : 1 }}
          transition={{ duration: 0.4 }}
          className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_#34d399]"
        />
        <span className="text-emerald-400 font-semibold">Live</span>
      </div>

      <div className="w-px h-3 bg-white/10" />

      {/* Total scans */}
      <div className="flex items-center space-x-1.5">
        <Zap size={11} className="text-amber-400" />
        <span className="text-slate-400">
          <span className="text-white font-bold font-mono">{stats.scans.total}</span>
          {" "}scans this session
        </span>
      </div>

      <div className="w-px h-3 bg-white/10" />

      {/* Per-type breakdown */}
      {statItems.map(({ icon: Icon, label, value, color }) => (
        <div key={label} className="flex items-center space-x-1">
          <Icon size={11} className={color} />
          <span className={`font-bold font-mono ${color}`}>{value}</span>
          <span className="text-slate-600">{label}</span>
        </div>
      ))}

      <div className="w-px h-3 bg-white/10" />

      {/* Uptime */}
      <div className="flex items-center space-x-1.5">
        <Clock size={11} className="text-slate-500" />
        <span className="text-slate-500 font-mono">{stats.uptime}</span>
      </div>

      {/* Models */}
      <div className="flex items-center space-x-1.5 ml-auto">
        <Activity size={11} className="text-slate-500" />
        <span className="text-slate-500">
          {[
            stats.models_loaded.text && "Text",
            stats.models_loaded.image && "Image",
            stats.models_loaded.video && "Video",
          ].filter(Boolean).join(" · ")} loaded
        </span>
      </div>
    </motion.div>
  );
}
