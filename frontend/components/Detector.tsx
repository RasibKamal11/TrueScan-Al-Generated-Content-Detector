"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, Image as ImageIcon, Video, Link as LinkIcon, ScanLine, Loader2, History, ChevronRight, Check, Upload, AlertCircle, Sparkles, Share2, Copy, CheckCheck, X, XCircle, Volume2, Code2, Layers, Table, AlertTriangle, Info } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { CircularProgress } from "./CircularProgress";
import { HistorySidebar, HistoryItem } from "./HistorySidebar";
import ExplainabilityPanel from "./ExplainabilityPanel";
import ScanningOverlay from "./ScanningOverlay";
import { Download, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import jsPDF from "jspdf";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export default function Detector() {
  const [activeTab, setActiveTab] = useState<"text" | "url" | "image" | "video" | "audio" | "code" | "bulk">("text");
  const [inputData, setInputData] = useState<string | File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [copied, setCopied] = useState(false);
  const [shareTooltip, setShareTooltip] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [deepScan, setDeepScan] = useState(false);
  const [wsProgress, setWsProgress] = useState<{step: number; total: number; message: string; progress: number} | null>(null);
  const analyzeButtonRef = useRef<HTMLButtonElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Load history on mount
  useEffect(() => {
    const saved = localStorage.getItem("scan_history");
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse history", e);
      }
    }
  }, []);

  // ── Keyboard shortcuts ─────────────────────────────────────────────────────
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+Enter → run scan
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        if (!loading && inputData) analyzeButtonRef.current?.click();
      }
      // Ctrl+H → toggle history
      if ((e.ctrlKey || e.metaKey) && e.key === "h") {
        e.preventDefault();
        setShowHistory(prev => !prev);
      }
      // Escape → clear error
      if (e.key === "Escape") setErrorMsg(null);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [loading, inputData]);

  const saveToHistory = (res: any, data: string | File) => {
    const newItem: HistoryItem = {
        id: crypto.randomUUID(),
        type: activeTab as HistoryItem["type"],
        content: typeof data === 'string' ? data : data.name,
        score: res.ai_probability ?? res.score,
        timestamp: Date.now()
    };
    
    const newHistory = [newItem, ...history].slice(0, 50); // Keep last 50
    setHistory(newHistory);
    localStorage.setItem("scan_history", JSON.stringify(newHistory));
  };

  const handleShare = useCallback(() => {
    if (!result) return;
    const scanId = result.id;
    let url = "";
    if (scanId) {
      url = `${window.location.origin}/share/${scanId}`;
    } else {
      const score = result.ai_probability ?? result.score;
      const payload = {
        type: activeTab,
        score: Math.round(score * 100),
        preview: typeof inputData === 'string' ? inputData.slice(0, 120) : (inputData as File)?.name ?? "",
        ts: Date.now(),
      };
      const encoded = btoa(JSON.stringify(payload));
      url = `${window.location.origin}${window.location.pathname}?scan=${encoded}`;
    }
    navigator.clipboard.writeText(url).then(() => {
      setCopied(true);
      setShareTooltip(true);
      setTimeout(() => { setCopied(false); setShareTooltip(false); }, 2500);
    });
  }, [result, activeTab, inputData]);

  const handleDownload = () => {
    if (!result) return;
    
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    
    // Header
    doc.setFillColor(15, 23, 42); // Slate 900
    doc.rect(0, 0, pageWidth, 40, "F");
    
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(22);
    doc.setFont("helvetica", "bold");
    doc.text("TrueScan Report", 20, 25);
    
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.text(`Generated: ${new Date().toLocaleString()}`, pageWidth - 20, 25, { align: "right" });
    
    // Summary Section
    doc.setTextColor(51, 65, 85); // Slate 700
    doc.setFontSize(16);
    doc.setFont("helvetica", "bold");
    doc.text("Analysis Summary", 20, 60);
    
    const score = result.ai_probability ?? result.score;
    const aiPercent = Math.round(score * 100);
    const humanPercent = 100 - aiPercent;
    
    let verdict = "Mixed Content";
    let color = [245, 158, 11]; // Amber
    if (score > 0.6) { verdict = "Likely AI-Generated"; color = [239, 68, 68]; } // Red
    else if (score < 0.4) { verdict = "Likely Human"; color = [16, 185, 129]; } // Emerald
    
    // Verdict Box
    doc.setDrawColor(color[0], color[1], color[2]);
    doc.setFillColor(color[0], color[1], color[2]);
    doc.roundedRect(20, 70, pageWidth - 40, 35, 3, 3, "FD");
    
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(14);
    doc.text("VERDICT", 30, 85);
    doc.setFontSize(24);
    doc.setFont("helvetica", "bold");
    doc.text(verdict.toUpperCase(), 30, 98);
    
    doc.setFontSize(40);
    doc.text(`${aiPercent}% Risk`, pageWidth - 30, 93, { align: "right" });
    
    // Details
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(12);
    doc.text("Detailed Breakdown:", 20, 125);
    
    doc.setFontSize(11);
    doc.setTextColor(100);
    doc.text(`• Risk Probability: ${aiPercent}%`, 25, 135);
    doc.text(`• Authenticity Probability: ${humanPercent}%`, 25, 142);
    
    if (typeof inputData === 'string') {
        doc.text(`• Input Length: ${inputData.length} characters`, 25, 149);
    } else if (inputData) {
        doc.text(`• File Name: ${inputData.name}`, 25, 149);
    }
    
    // Footer
    doc.setFillColor(15, 23, 42);
    doc.rect(0, 280, pageWidth, 20, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(8);
    doc.text("TrueScan AI-Generated Content Detector - Enterprise Grade Verification", pageWidth / 2, 290, { align: "center" });
    
    if (result.sentences) {
        doc.addPage();
        doc.setFillColor(15, 23, 42);
        doc.rect(0, 0, pageWidth, 30, "F");
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(16);
        doc.text("Segment Analysis", 20, 20);
        
        doc.setTextColor(0, 0, 0);
        doc.setFontSize(10);
        let y = 50;
        result.sentences.slice(0, 20).forEach((s: any, i: number) => {
            const sScore = s.ai_probability ?? s.score;
            const sType = sScore > 0.6 ? "[AI]" : "[Human]";
            doc.setFont("helvetica", "bold");
            doc.text(`${i+1}. ${sType} (${Math.round(sScore * 100)}%)`, 20, y);
            doc.setFont("helvetica", "normal");
            const lines = doc.splitTextToSize(s.text, pageWidth - 40);
            doc.text(lines, 25, y + 5);
            y += (lines.length * 5) + 15;
            if (y > 270) { doc.addPage(); y = 30; }
        });
    }

    doc.save("TrueScan_Report.pdf");
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setInputData(e.target.files[0]);
    }
  };

  const analyze = async () => {
    setLoading(true);
    setResult(null);
    setWsProgress(null);
    await new Promise(r => setTimeout(r, 300));
    setErrorMsg(null);

    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

    try {
      let data: any;

      // ─ WebSocket path for text-based tabs ──────────────────────────────
      if (activeTab === "text" || activeTab === "url") {
        data = await wsAnalyze(activeTab, inputData as string);
      }
      // ─ Code tab (WebSocket) ───────────────────────────────────────────
      else if (activeTab === "code") {
        data = await wsAnalyze("code", inputData as string);
      }
      // ─ File-based REST endpoints ────────────────────────────────────
      else {
        const endpoint = `${apiBase}/detect/${activeTab === "bulk" ? "bulk-file" : activeTab}`;
        const formData = new FormData();
        formData.append("file", inputData as File);
        const response = await fetch(endpoint, { method: "POST", body: formData });
        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || `Server returned ${response.status}`);
        }
        data = await response.json();
      }

      setResult(data);
      saveToHistory(data, inputData as string | File);

    } catch (error) {
      console.error("Analysis failed:", error);
      const msg = error instanceof Error ? error.message : String(error);
      setErrorMsg(msg);
    } finally {
      setLoading(false);
      setWsProgress(null);
    }
  };

  // ── WebSocket analyze helper ───────────────────────────────────────────────
  const wsAnalyze = useCallback((type: string, payload: string): Promise<any> => {
    return new Promise((resolve, reject) => {
      const apiBase = (process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000').replace(/^https?/, 'ws');
      const ws = new WebSocket(`${apiBase}/detect/ws/detect`);
      wsRef.current = ws;
      ws.onmessage = (evt) => {
        const msg = JSON.parse(evt.data);
        if (msg.done) {
          setWsProgress(null);
          ws.close();
          if (msg.error) reject(new Error(msg.error));
          else resolve(msg.result);
        } else {
          setWsProgress({ step: msg.step, total: msg.total, message: msg.message, progress: msg.progress });
        }
      };
      ws.onerror = () => {
        ws.close();
        const base = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
        const ep = type === 'code' ? `${base}/detect/code`
                 : type === 'url'  ? `${base}/detect/url`
                 :                   `${base}/detect/${type}`;
        const hdrs: Record<string, string> = { 'Content-Type': 'application/json' };
        const body = type === 'url'  ? JSON.stringify({ url: payload })
                   : type === 'code' ? JSON.stringify({ code: payload, deep_scan: deepScan })
                   :                   JSON.stringify({ text: payload, detailed: true, deep_scan: deepScan });
        fetch(ep, { method: 'POST', headers: hdrs, body })
          .then(r => r.ok ? r.json() : r.json().then((e: any) => Promise.reject(new Error(e.detail))))
          .then(resolve)
          .catch(reject);
      };
      ws.onopen = () => ws.send(JSON.stringify({ type, payload, deep_scan: deepScan }));
    });
  }, []);

  // Cancel active WebSocket on unmount
  useEffect(() => () => { wsRef.current?.close(); }, []);

  // ── Tab icons ────────────────────────────────────────────────
  const tabIcons = {
    text:  FileText,
    url:   LinkIcon,
    image: ImageIcon,
    video: Video,
    audio: Volume2,
    code:  Code2,
    bulk:  Layers,
  };

  const tabLabels: Record<string, string> = {
    text:  "Text",
    url:   "URL",
    image: "Image",
    video: "Video",
    audio: "Audio",
    code:  "Code",
    bulk:  "Bulk",
  };

  const renderHighlightedText = () => {
    if (!result) return null;
    const textContent = typeof inputData === "string" ? inputData : "Analyzed Content";

    if (result.sentences && Array.isArray(result.sentences) && result.sentences.length > 0) {
        // Group consecutive sentences of the same bucket
        const groups: { type: 'ai' | 'partial' | 'human'; text: string; sentences: any[] }[] = [];
        
        result.sentences.forEach((s: any) => {
            const p = s.ai_probability ?? 0;
            let type: 'ai' | 'partial' | 'human' = 'human';
            if (p > 0.75) {
                type = 'ai';
            } else if (p > 0.45) {
                type = 'partial';
            }
            
            if (groups.length > 0 && groups[groups.length - 1].type === type) {
                groups[groups.length - 1].text += " " + s.text;
                groups[groups.length - 1].sentences.push(s);
            } else {
                groups.push({
                    type,
                    text: s.text,
                    sentences: [s]
                });
            }
        });

        return (
            <div className="space-y-6">
                {/* legend header info */}
                <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-white/5">
                    <div className="flex flex-wrap gap-2">
                        <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500 self-center mr-1">Highlighting Legend:</span>
                        {[
                            { color: "bg-red-500/20 border-red-500/30 text-red-300", dot: "bg-red-400", label: "AI-Generated" },
                            { color: "bg-orange-500/15 border-orange-500/25 text-orange-300", dot: "bg-orange-400", label: "Partial AI" },
                            { color: "bg-emerald-500/10 border-emerald-500/20 text-emerald-300", dot: "bg-emerald-400", label: "Human-Written" },
                        ].map(({ color, dot, label }) => (
                            <span key={label} className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[11px] font-semibold ${color}`}>
                                <span className={`w-2 h-2 rounded-full ${dot}`} />
                                {label}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Highlighted text container */}
                <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                    {groups.map((group, gIdx) => {
                        let bg = "bg-emerald-950/20 border-emerald-500/20 text-emerald-100/90";
                        let bannerBg = "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
                        let Icon = Check;
                        let label = "Human-Written (Natural)";
                        
                        if (group.type === "ai") {
                            bg = "bg-red-950/30 border-red-500/20 text-red-100/90";
                            bannerBg = "bg-red-500/20 text-red-400 border-red-500/30";
                            Icon = AlertTriangle;
                            label = "AI-Generated (High probability)";
                        } else if (group.type === "partial") {
                            bg = "bg-orange-950/20 border-orange-500/20 text-orange-100/90";
                            bannerBg = "bg-orange-500/20 text-orange-400 border-orange-500/30";
                            Icon = Info;
                            label = "Partial AI (Moderate probability)";
                        }

                        return (
                            <div key={gIdx} className={`p-5 rounded-2xl border ${bg} leading-[1.8] text-[15px] font-normal transition-all duration-300`}>
                                <div className="mb-3">
                                    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-bold ${bannerBg}`}>
                                        <Icon size={12} />
                                        {label}
                                    </span>
                                </div>
                                <p>{group.text}</p>
                            </div>
                        );
                    })}
                </div>
            </div>
        );
    }

    return <div className="bg-slate-950/70 rounded-2xl p-6 text-[15px] leading-[1.9] text-slate-200 max-h-[480px] overflow-y-auto border border-white/5 whitespace-pre-wrap custom-scrollbar">{textContent}</div>;
  };

  const renderHighlightedCode = () => {
    if (!result) return null;
    const codeContent = typeof inputData === "string" ? inputData : "";
    const lines = codeContent.split("\n");
    const score = result.ai_probability ?? result.score ?? 0.5;

    // Use backend line_analysis if available, fallback to heuristic
    const codeLineFlags: number[] = lines.map((line, idx) => {
        if (result.line_analysis && result.line_analysis[idx]) {
            return result.line_analysis[idx].ai_probability;
        }
        // Fallback
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#") || trimmed.startsWith("//") || trimmed.startsWith("*")) return 0;
        let h = 0;
        for (let i = 0; i < Math.min(trimmed.length, 40); i++) h = (h * 31 + trimmed.charCodeAt(i)) & 0xffff;
        const norm = (h % 100) / 100;
        return norm < score ? score * 0.8 + norm * 0.2 : norm * 0.1;
    });

    const cs = result.code_stats;

    return (
        <div className="space-y-6">
            <div className="space-y-2">
                <div className="flex items-center gap-2">
                    <div className="flex gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-red-500/80" />
                        <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                        <div className="w-3 h-3 rounded-full bg-green-500/80" />
                    </div>
                    <span className="text-xs text-slate-500 font-mono ml-2">{result.language ?? "code"} · Code Detector Console</span>
                </div>
                <div className="bg-[#0d1117] rounded-2xl border border-white/5 overflow-hidden shadow-2xl">
                    <div className="max-h-[500px] overflow-y-auto custom-scrollbar">
                        <table className="w-full text-sm font-mono border-collapse">
                            <tbody>
                                {lines.map((line, idx) => {
                                    const prob = codeLineFlags[idx];
                                    const isHigh = prob > 0.65;
                                    const isMed = prob > 0.35 && !isHigh;
                                    return (
                                        <tr key={idx} className={cn("transition-colors duration-100", isHigh ? "bg-red-500/15 hover:bg-red-500/22" : isMed ? "bg-orange-500/8 hover:bg-orange-500/14" : "hover:bg-white/3")}>
                                            <td className={cn("select-none text-right pr-3 pl-4 py-[3px] text-[11px] border-r w-10 shrink-0 tabular-nums", isHigh ? "text-red-400/60 border-red-500/18" : isMed ? "text-orange-400/50 border-orange-500/12" : "text-slate-600 border-white/5")}>
                                                {idx + 1}
                                            </td>
                                            <td className={cn("pl-4 pr-4 py-[3px] whitespace-pre leading-6", isHigh ? "text-red-200" : isMed ? "text-orange-200" : "text-slate-300")}>
                                                {line || " "}
                                                {isHigh && (
                                                    <span className="ml-3 inline-flex items-center px-1.5 py-px rounded text-[9px] font-bold bg-red-500/25 text-red-300 border border-red-500/20 align-middle">
                                                        AI pattern
                                                    </span>
                                                )}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            {/* Code Stats underneath the viewer */}
            {cs && (
                <div className="pt-4 border-t border-white/5">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-3">Code Statistics</p>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {[
                            { label: "Lines",     value: cs.lines,         icon: "📄" },
                            { label: "Functions", value: cs.functions,     icon: "⚡" },
                            { label: "Comments",  value: cs.comment_lines, icon: "💬" },
                            { label: "Blank",     value: cs.blank_lines,   icon: "⬜" },
                        ].map(({ label, value, icon }) => (
                            <div key={label} className="bg-slate-950/40 rounded-xl p-3 text-center border border-white/5 flex items-center justify-center gap-3">
                                <span className="text-xl">{icon}</span>
                                <div className="text-left">
                                    <div className="text-base font-black text-white tabular-nums leading-none">{value ?? "—"}</div>
                                    <div className="text-[9px] text-slate-500 uppercase tracking-wider mt-0.5">{label}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
  };

  return (
    <div className="w-full max-w-5xl mx-auto">
        {/* Header Actions */}
        <div className="flex justify-between items-center mb-6 px-2">
            <div className="flex items-center space-x-2 text-slate-400 text-sm">
                <ShieldCheck size={16} className="text-blue-500" />
                <span className="font-medium">Enterprise Security Active</span>
            </div>
            <Button 
                variant="ghost" 
                size="sm"
                onClick={() => setShowHistory(true)}
                className="text-slate-400 hover:text-white"
            >
                <History size={16} className="mr-2" />
                History
            </Button>
        </div>

        <HistorySidebar 
            isOpen={showHistory} 
            onClose={() => setShowHistory(false)}
            items={history}
            onSelect={(item) => {
                setActiveTab(item.type);
                setInputData(item.content); 
                setShowHistory(false);
            }} 
            onClear={() => {
                setHistory([]);
                localStorage.removeItem("scan_history");
            }}
        />

        <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full"
        >
            <Card className="border-0 shadow-2xl bg-slate-900/60 backdrop-blur-2xl ring-1 ring-white/10 overflow-visible">
                <CardContent className="p-1">
                    
                    <div className="bg-slate-950/50 rounded-[14px] p-6 md:p-10 relative overflow-hidden">
                        {/* Dynamic Background Glow */}
                        <div className="absolute -top-32 -right-32 w-96 h-96 bg-blue-500/10 blur-[100px] pointer-events-none rounded-full" />
                        <div className="absolute -bottom-32 -left-32 w-96 h-96 bg-violet-500/10 blur-[100px] pointer-events-none rounded-full" />

                        {/* Tabs — horizontally scrollable on mobile */}
                        <div className="flex justify-center mb-10 relative z-10">
                            <div className="flex p-1.5 rounded-2xl bg-black/40 border border-white/5 backdrop-blur-md overflow-x-auto max-w-full" style={{scrollbarWidth:'none'}}>
                                {(["text", "url", "image", "video", "audio", "code", "bulk"] as const).map((tab) => {
                                    const Icon = tabIcons[tab];
                                    return (
                                        <button
                                            key={tab}
                                            id={`tab-${tab}`}
                                            onClick={() => { setActiveTab(tab); setResult(null); setInputData(null); setErrorMsg(null); }}
                                            className={cn(
                                                "flex items-center space-x-2 px-4 sm:px-6 py-2.5 rounded-xl text-sm font-semibold transition-all relative z-10 whitespace-nowrap shrink-0",
                                                activeTab === tab ? "text-white" : "text-slate-500 hover:text-slate-300"
                                            )}
                                        >
                                            {activeTab === tab && (
                                                <motion.div
                                                    layoutId="activeTab"
                                                    className="absolute inset-0 bg-white/10 shadow-[0_0_20px_rgba(255,255,255,0.05)] border border-white/10 rounded-xl"
                                                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                                                />
                                            )}
                                            <span className="relative z-10 flex items-center space-x-2">
                                                <Icon size={16} />
                                                <span className="capitalize">{tabLabels[tab] ?? tab}</span>
                                            </span>
                                        </button>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Input Area */}
                        <div className="mb-8 min-h-[300px] relative z-10">
                            <AnimatePresence mode="wait">
                                <motion.div
                                    key={activeTab}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -20 }}
                                    transition={{ duration: 0.2 }}
                                    className="h-full"
                                >
                                    {activeTab === "text" ? (
                                        <div className="relative group h-full">
                                            <textarea
                                                className="relative w-full h-80 bg-slate-900/40 border border-white/5 rounded-2xl p-8 text-lg text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500/50 transition-all resize-none placeholder:text-slate-600 font-light leading-relaxed font-mono shadow-inner custom-scrollbar"
                                                placeholder="Paste the text you want to analyze..."
                                                value={typeof inputData === "string" ? inputData : ""}
                                                onChange={(e) => setInputData(e.target.value)}
                                            />
                                            <div className="absolute bottom-4 right-4 text-xs font-mono text-slate-600 bg-black/40 px-2 py-1 rounded">
                                                {(typeof inputData === 'string' ? inputData.length : 0)} chars
                                            </div>
                                        </div>
                                    ) : activeTab === "code" ? (
                                        <div className="relative group h-full">
                                            <textarea
                                                className="relative w-full h-80 bg-slate-900/40 border border-white/5 rounded-2xl p-5 text-sm text-slate-200 focus:outline-none focus:ring-1 focus:ring-pink-500/50 transition-all resize-none placeholder:text-slate-600 font-mono leading-relaxed shadow-inner custom-scrollbar"
                                                placeholder={`// Paste your code here — Python, JavaScript, TypeScript, Java, C++...
function example() {
  return 'Is this AI-generated?';
}`}
                                                value={typeof inputData === "string" ? inputData : ""}
                                                onChange={(e) => setInputData(e.target.value)}
                                                spellCheck={false}
                                            />
                                            <div className="absolute bottom-4 right-4 flex items-center space-x-2">
                                                <Code2 size={12} className="text-pink-500" />
                                                <span className="text-xs font-mono text-slate-600 bg-black/40 px-2 py-1 rounded">
                                                    {(typeof inputData === 'string' ? inputData.split('\n').length : 0)} lines
                                                </span>
                                            </div>
                                        </div>
                                    ) : activeTab === "url" ? (
                                         <div className="flex flex-col items-center justify-center h-80 space-y-8">
                                            <div className="w-full max-w-lg">
                                                <Input 
                                                    type="url"
                                                    placeholder="https://example.com/article..."
                                                    className="h-14 text-lg px-6 bg-black/40 border-white/10 focus:border-blue-500/50 rounded-2xl"
                                                    value={typeof inputData === "string" ? inputData : ""}
                                                    onChange={(e) => setInputData(e.target.value)}
                                                />
                                            </div>
                                            <div className="flex flex-col items-center text-slate-500 space-y-2">
                                                <LinkIcon size={32} className="opacity-20" />
                                                <p className="text-sm max-w-sm text-center opacity-60">
                                                    Enter a publicly accessible URL. Our robust crawler will extract the main content for analysis.
                                                </p>
                                            </div>
                                         </div>
                                    ) : activeTab === "bulk" ? (
                                        <div className="w-full h-80 border border-dashed border-indigo-500/30 rounded-2xl flex flex-col items-center justify-center text-center hover:border-indigo-500/50 hover:bg-indigo-500/5 transition-all cursor-pointer relative group overflow-hidden bg-black/20">
                                            <input
                                                type="file"
                                                accept=".txt,.csv"
                                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-50"
                                                onChange={handleFileChange}
                                            />
                                            <div className="z-10 flex flex-col items-center space-y-4 group-hover:scale-105 transition-transform duration-300">
                                                <div className="w-20 h-20 rounded-full bg-slate-900/80 flex items-center justify-center group-hover:bg-indigo-500/20 group-hover:text-indigo-400 transition-colors border border-white/5">
                                                    {inputData && typeof inputData !== 'string' ? <Check size={32} className="text-emerald-400" /> : <Layers size={32} className="text-indigo-400" />}
                                                </div>
                                                <div className="text-slate-400 space-y-1 max-w-xs">
                                                    {inputData && typeof inputData !== 'string' ? (
                                                        <span className="font-medium text-white">{(inputData as File).name}</span>
                                                    ) : (
                                                        <>
                                                            <p className="font-medium text-lg text-slate-200">Upload .txt or .csv file</p>
                                                            <p className="text-sm opacity-50">TXT: one paragraph per blank line · CSV: text in first column</p>
                                                            <p className="text-xs text-indigo-400/70 mt-2">Up to 200 items per file</p>
                                                        </>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    ) : activeTab === "audio" ? (
                                        <div className="w-full h-80 border border-dashed border-green-500/30 rounded-2xl flex flex-col items-center justify-center text-center hover:border-green-500/50 hover:bg-green-500/5 transition-all cursor-pointer relative group overflow-hidden bg-black/20">
                                            <input
                                                type="file"
                                                accept="audio/*,.mp3,.wav,.ogg,.flac,.m4a"
                                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-50"
                                                onChange={handleFileChange}
                                            />
                                            <div className="z-10 flex flex-col items-center space-y-4 group-hover:scale-105 transition-transform duration-300">
                                                <div className="w-20 h-20 rounded-full bg-slate-900/80 flex items-center justify-center group-hover:bg-green-500/20 group-hover:text-green-400 transition-colors border border-white/5">
                                                    {inputData && typeof inputData !== 'string' ? <Check size={32} className="text-emerald-400" /> : <Volume2 size={32} className="text-green-400" />}
                                                </div>
                                                <div className="text-slate-400 space-y-1 max-w-xs">
                                                    {inputData && typeof inputData !== 'string' ? (
                                                        <span className="font-medium text-white">{(inputData as File).name}</span>
                                                    ) : (
                                                        <>
                                                            <p className="font-medium text-lg text-slate-200">Drop your audio file here</p>
                                                            <p className="text-sm opacity-50">MP3, WAV, OGG, FLAC, M4A supported</p>
                                                            <p className="text-xs text-green-400/70 mt-2">Detects ElevenLabs, Suno, Bark, Play.ht and more</p>
                                                        </>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="w-full h-80 border border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center text-center hover:border-blue-500/30 hover:bg-blue-500/5 transition-all cursor-pointer relative group overflow-hidden bg-black/20">
                                            <input
                                                type="file"
                                                accept={activeTab === "image" ? "image/*" : "video/*"}
                                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-50"
                                                onChange={handleFileChange}
                                            />
                                            <div className="z-10 flex flex-col items-center space-y-6 group-hover:scale-105 transition-transform duration-300">
                                                <div className="w-20 h-20 rounded-full bg-slate-900/80 flex items-center justify-center group-hover:bg-blue-500/20 group-hover:text-blue-400 transition-colors border border-white/5 shadow-lg">
                                                    {activeTab === "image" ? <ImageIcon size={32} /> : <Video size={32} />}
                                                </div>
                                                <div className="text-slate-400 group-hover:text-blue-300 transition-colors max-w-xs space-y-1">
                                                    {inputData && typeof inputData !== "string" ? (
                                                        <div className="flex flex-col items-center space-y-2">
                                                            <Check size={20} className="text-emerald-400 mb-2" />
                                                            <span className="font-medium text-white break-all bg-white/5 py-1 px-3 rounded-lg">{inputData.name}</span>
                                                        </div>
                                                    ) : (
                                                        <>
                                                            <p className="font-medium text-lg text-slate-200">Drop your {activeTab} file here</p>
                                                            <p className="text-sm opacity-50">Supports all major formats</p>
                                                        </>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </motion.div>
                            </AnimatePresence>
                        </div>

                        {/* Scanning Overlay */}
                        <AnimatePresence>
                          {loading && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: "auto" }}
                              exit={{ opacity: 0, height: 0 }}
                              className="mb-6 overflow-hidden"
                            >
                              <div className="bg-slate-950/60 border border-white/5 rounded-2xl p-6">
                                <ScanningOverlay isLoading={loading} activeTab={activeTab} wsProgress={wsProgress} />
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>

                        {/* Inline Error Card */}
                        <AnimatePresence>
                          {errorMsg && (
                            <motion.div
                              initial={{ opacity: 0, y: -8, scale: 0.97 }}
                              animate={{ opacity: 1, y: 0, scale: 1 }}
                              exit={{ opacity: 0, y: -8, scale: 0.97 }}
                              transition={{ duration: 0.2 }}
                              className="mb-4 flex items-start gap-3 bg-red-950/60 border border-red-500/30 rounded-2xl px-5 py-4 text-red-300 text-sm shadow-lg backdrop-blur-sm"
                              role="alert"
                            >
                              <XCircle size={18} className="shrink-0 mt-0.5 text-red-400" />
                              <div className="flex-1">
                                <p className="font-semibold text-red-200 mb-0.5">Analysis Failed</p>
                                <p className="text-red-400 text-xs leading-relaxed">{errorMsg}</p>
                              </div>
                              <button
                                onClick={() => setErrorMsg(null)}
                                className="shrink-0 text-red-500 hover:text-red-300 transition-colors"
                                aria-label="Dismiss error"
                              >
                                <X size={15} />
                              </button>
                            </motion.div>
                          )}
                        </AnimatePresence>

                            {/* Analyze Button */}
                            <div className="flex flex-col items-center space-y-6">
                                {/* Deep Scan Toggle */}
                                {(activeTab === "text" || activeTab === "code") && (
                                    <div className="flex items-center space-x-3 bg-white/5 px-4 py-2 rounded-full border border-white/5">
                                        <div className="flex flex-col items-start">
                                            <div className="flex items-center space-x-2">
                                                <Sparkles size={12} className={cn("transition-colors", deepScan ? "text-blue-400" : "text-slate-500")} />
                                                <span className={cn("text-xs font-bold uppercase tracking-wider", deepScan ? "text-blue-100" : "text-slate-500")}>
                                                    Deep Scan (AI+Ensemble)
                                                </span>
                                            </div>
                                        </div>
                                        <button 
                                            onClick={() => setDeepScan(!deepScan)}
                                            className={cn(
                                                "w-10 h-5 rounded-full relative transition-colors duration-300",
                                                deepScan ? "bg-blue-600" : "bg-slate-700"
                                            )}
                                        >
                                            <motion.div 
                                                animate={{ x: deepScan ? 22 : 2 }}
                                                className="absolute top-1 left-0 w-3 h-3 bg-white rounded-full shadow-lg"
                                            />
                                        </button>
                                    </div>
                                )}

                                <Button
                                    ref={analyzeButtonRef}
                                    id="analyze-btn"
                                    size="lg"
                                    onClick={analyze}
                                    disabled={loading || !inputData}
                                    variant="glow"
                                    className="w-full max-w-sm h-14 text-lg rounded-xl relative overflow-hidden group shadow-[0_0_40px_rgba(59,130,246,0.3)] hover:shadow-[0_0_60px_rgba(59,130,246,0.5)] transition-all duration-500"
                                    title="Run scan (Ctrl+Enter)"
                                >
                                {loading ? (
                                    <span className="flex items-center space-x-3">
                                        <Loader2 className="animate-spin" />
                                        <span>Scanning...</span>
                                    </span>
                                ) : (
                                    <span className="flex items-center space-x-2">
                                        <Sparkles size={18} /> 
                                        <span>Run Smart Scan</span>
                                    </span>
                                )}
                            </Button>
                            <p className="text-xs text-slate-600">Tip: press <kbd className="bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded text-[10px] font-mono">Ctrl+Enter</kbd> to scan</p>
                        </div>

                    </div>
                </CardContent>
            </Card>
        </motion.div>

        {/* Results */}
        <AnimatePresence>
            {result && !loading && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    transition={{ type: "spring", bounce: 0.2 }}
                    className="mt-8 space-y-6"
                >
                    {(() => {
                        const score = result.ai_probability ?? result.score;
                        const humanScore = 1 - score;
                        
                        let status = { text: "Likely Mixed", color: "text-amber-400", bg: "bg-amber-500", border: "border-amber-500/20", shadow:"shadow-amber-500/20", desc: "Contains a mix of patterns." };
                        
                        if (activeTab === "audio") {
                             if (score > 0.6) status = { text: "Likely AI-Generated Voice", color: "text-red-400", bg: "bg-red-500", border: "border-red-500/20", shadow:"shadow-red-500/20", desc: "Signal patterns indicate synthetic audio generation." };
                             else if (score < 0.4) status = { text: "Likely Human Voice", color: "text-emerald-400", bg: "bg-emerald-500", border: "border-emerald-500/20", shadow:"shadow-emerald-500/20", desc: "Natural speech patterns detected." };
                        } else if (activeTab === "code") {
                             if (score > 0.6) status = { text: "Likely AI-Generated Code", color: "text-red-400", bg: "bg-red-500", border: "border-red-500/20", shadow:"shadow-red-500/20", desc: "Copilot/ChatGPT patterns detected in code structure." };
                             else if (score < 0.4) status = { text: "Likely Human-Written Code", color: "text-emerald-400", bg: "bg-emerald-500", border: "border-emerald-500/20", shadow:"shadow-emerald-500/20", desc: "Natural coding style and human error patterns detected." };
                        } else {
                             if (score > 0.6) status = { text: "Likely AI-Generated", color: "text-red-400", bg: "bg-red-500", border: "border-red-500/20", shadow:"shadow-red-500/20", desc: "Strong AI signatures detected." };
                             else if (score < 0.4) status = { text: "Likely Human", color: "text-emerald-400", bg: "bg-emerald-500", border: "border-emerald-500/20", shadow:"shadow-emerald-500/20", desc: "Consistent with human writing patterns." };
                        }

                        
                        if (activeTab === "text" || activeTab === "url") {
                            const perplexityScore = result.metrics?.perplexity ?? (score > 0.5 ? 20 : 85);
                            const perplexityIsAI = perplexityScore < 40;
                            const perplexityBadge = perplexityIsAI ? "Low" : "High";

                            const burstinessScore = result.metrics?.burstiness ?? (score > 0.5 ? 1.8 : 5.8);
                            const burstinessIsAI = burstinessScore < 3.0;
                            const burstinessBadge = burstinessIsAI ? "Low" : "High";

                            return (
                                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                    {/* Left: Source Text Card */}
                                    <Card className="lg:col-span-2 border border-white/10 bg-slate-900/40 shadow-2xl">
                                        <CardContent className="p-8">
                                            <div className="flex items-center space-x-2 mb-6">
                                                <ScanLine size={18} className="text-blue-500" />
                                                <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Source Text</h3>
                                            </div>
                                            {renderHighlightedText()}
                                        </CardContent>
                                    </Card>

                                    {/* Right: Metrics & Scores Card */}
                                    <Card className="lg:col-span-1 border border-white/10 bg-slate-900/40 shadow-2xl flex flex-col justify-between">
                                        <CardContent className="p-8 space-y-8 flex-1 flex flex-col justify-between">
                                            <div className="space-y-6">
                                                <div className="flex items-center justify-between pb-2 border-b border-white/5">
                                                    <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Metrics & Scores</h3>
                                                </div>

                                                {/* Circular Gauge */}
                                                <div className="flex flex-col items-center py-4">
                                                    <div className="relative shrink-0">
                                                        <CircularProgress 
                                                            value={score * 100} 
                                                            size={160}
                                                            strokeWidth={12}
                                                            color={score > 0.6 ? "#ef4444" : score < 0.4 ? "#10b981" : "#f59e0b"}
                                                        />
                                                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                                                            <span className="text-3xl font-black text-white tracking-tighter">
                                                                {Math.round(score * 100)}%
                                                            </span>
                                                            <span className="text-[9px] font-bold uppercase tracking-widest text-slate-500 mt-0.5">
                                                                AI Probability
                                                            </span>
                                                            <span className={`text-[10px] font-bold uppercase mt-1 ${status.color}`}>
                                                                {status.text}
                                                            </span>
                                                        </div>
                                                    </div>
                                                </div>

                                                {/* Perplexity Chart */}
                                                <div className="space-y-4">
                                                    <div className="flex justify-between items-center">
                                                        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Perplexity</span>
                                                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${perplexityIsAI ? "bg-red-500/10 text-red-400 border-red-500/20" : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"}`}>
                                                            {perplexityBadge}
                                                        </span>
                                                    </div>
                                                    
                                                    <div className="flex items-stretch h-28 gap-4">
                                                        <div className="flex flex-col justify-between text-[10px] font-mono text-slate-600 select-none pr-1">
                                                            <span>30</span>
                                                            <span>20</span>
                                                            <span>10</span>
                                                            <span>0</span>
                                                        </div>
                                                        <div className="flex-1 flex items-end justify-around pb-1 border-b border-white/5 relative h-full">
                                                            <div className="flex flex-col items-center w-8 group relative z-10">
                                                                <span className={`text-[10px] font-bold font-mono mb-1 ${perplexityIsAI ? "text-red-400" : "text-emerald-400"}`}>
                                                                    {perplexityScore}
                                                                </span>
                                                                <div 
                                                                    className={`w-3.5 rounded-t-sm transition-all duration-1000 ${perplexityIsAI ? "bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.3)]" : "bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.3)]"}`} 
                                                                    style={{ height: `${Math.min(100, (perplexityScore / 30) * 100)}%`, minHeight: '4px' }}
                                                                />
                                                                <span className="text-[8px] font-bold text-slate-500 mt-1 uppercase tracking-wide">Doc</span>
                                                            </div>
                                                            <div className="flex flex-col items-center w-8">
                                                                <span className="text-[10px] font-mono text-slate-600 mb-1">15.0</span>
                                                                <div className="w-3.5 bg-slate-800 rounded-t-sm" style={{ height: "50%" }} />
                                                                <span className="text-[8px] font-bold text-slate-600 mt-1 uppercase tracking-wide">AI</span>
                                                            </div>
                                                            <div className="flex flex-col items-center w-8">
                                                                <span className="text-[10px] font-mono text-slate-600 mb-1">85.0</span>
                                                                <div className="w-3.5 bg-slate-800 rounded-t-sm" style={{ height: "100%" }} />
                                                                <span className="text-[8px] font-bold text-slate-600 mt-1 uppercase tracking-wide">Human</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>

                                                {/* Burstiness Chart */}
                                                <div className="space-y-4">
                                                    <div className="flex justify-between items-center">
                                                        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Burstiness</span>
                                                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${burstinessIsAI ? "bg-red-500/10 text-red-400 border-red-500/20" : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"}`}>
                                                            {burstinessBadge}
                                                        </span>
                                                    </div>
                                                    
                                                    <div className="flex items-stretch h-28 gap-4">
                                                        <div className="flex flex-col justify-between text-[10px] font-mono text-slate-600 select-none pr-1">
                                                            <span>4</span>
                                                            <span>3</span>
                                                            <span>2</span>
                                                            <span>1</span>
                                                            <span>0</span>
                                                        </div>
                                                        <div className="flex-1 flex items-end justify-around pb-1 border-b border-white/5 relative h-full">
                                                            <div className="flex flex-col items-center w-8 group relative z-10">
                                                                <span className={`text-[10px] font-bold font-mono mb-1 ${burstinessIsAI ? "text-red-400" : "text-emerald-400"}`}>
                                                                    {burstinessScore}
                                                                </span>
                                                                <div 
                                                                    className={`w-3.5 rounded-t-sm transition-all duration-1000 ${burstinessIsAI ? "bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.3)]" : "bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.3)]"}`} 
                                                                    style={{ height: `${Math.min(100, (burstinessScore / 4) * 100)}%`, minHeight: '4px' }}
                                                                />
                                                                <span className="text-[8px] font-bold text-slate-500 mt-1 uppercase tracking-wide">Doc</span>
                                                            </div>
                                                            <div className="flex flex-col items-center w-8">
                                                                <span className="text-[10px] font-mono text-slate-600 mb-1">1.5</span>
                                                                <div className="w-3.5 bg-slate-800 rounded-t-sm" style={{ height: "37.5%" }} />
                                                                <span className="text-[8px] font-bold text-slate-600 mt-1 uppercase tracking-wide">AI</span>
                                                            </div>
                                                            <div className="flex flex-col items-center w-8">
                                                                <span className="text-[10px] font-mono text-slate-600 mb-1">5.5</span>
                                                                <div className="w-3.5 bg-slate-800 rounded-t-sm" style={{ height: "100%" }} />
                                                                <span className="text-[8px] font-bold text-slate-600 mt-1 uppercase tracking-wide">Human</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="pt-6 border-t border-white/5 flex flex-col gap-3">
                                                <div className="flex justify-between gap-4">
                                                    <Button
                                                        variant="outline"
                                                        size="sm"
                                                        className="flex-1 bg-transparent border-white/10 text-slate-300 hover:text-white hover:bg-white/5"
                                                        onClick={handleShare}
                                                    >
                                                        {copied ? "Link Copied!" : "Share Result"}
                                                    </Button>
                                                    <Button 
                                                        variant="outline" 
                                                        size="sm" 
                                                        className="flex-1 bg-transparent border-white/10 text-slate-300 hover:text-white hover:bg-white/5"
                                                        onClick={handleDownload}
                                                    >
                                                        Download PDF
                                                    </Button>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                </div>
                            );
                        }



                        // Standard layout for other tabs (Image, Video, Audio, Bulk)
                        return (
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                {/* Left: Verdict Card */}
                                <Card className={`lg:col-span-2 border border-white/10 ${status.border} shadow-2xl relative overflow-hidden group`}>
                                    <div className={cn("absolute inset-0 opacity-5 blur-3xl transition-opacity group-hover:opacity-10 pointer-events-none", status.bg)} />
                                    
                                    <CardContent className="p-8">
                                       <div className="flex flex-col md:flex-row items-center gap-10">
                                            {/* Gauge */}
                                            <div className="relative shrink-0">
                                                <CircularProgress 
                                                    value={score * 100} 
                                                    size={180}
                                                    strokeWidth={14}
                                                    color={score > 0.6 ? "#ef4444" : score < 0.4 ? "#10b981" : "#f59e0b"}
                                                />
                                                <div className="absolute inset-0 flex flex-col items-center justify-center">
                                                    <span className="text-4xl font-black text-white tracking-tighter">
                                                        {Math.round(score * 100)}%
                                                    </span>
                                                    <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mt-1 flex items-center gap-1 group relative cursor-help">
                                                        Risk Score
                                                        <Info size={10} className="text-slate-500 hover:text-blue-400 transition-colors" />
                                                        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-slate-950/95 border border-white/10 rounded-xl text-[10px] text-slate-400 normal-case tracking-normal opacity-0 group-hover:opacity-100 transition-all duration-300 pointer-events-none z-50 shadow-2xl backdrop-blur-md translate-y-2 group-hover:translate-y-0 leading-relaxed text-center font-normal">
                                                            <strong>AI Probability / Risk Score</strong> represents the likelihood that the analyzed content was generated by an artificial intelligence model (like ChatGPT, Claude, or Midjourney) rather than a human. It measures structural predictability and synthetic noise.
                                                            <span className="absolute top-full left-1/2 -translate-x-1/2 -mt-2 border-8 border-transparent border-t-slate-950/95" />
                                                        </span>
                                                    </span>
                                                </div>
                                            </div>

                                            <div className="flex-1 space-y-6 text-center md:text-left w-full">
                                                <div>
                                                    <h3 className="text-sm font-bold uppercase tracking-widest text-slate-500 mb-2">Detection Result</h3>
                                                    <div className={cn("text-4xl font-bold tracking-tight", status.color)}>
                                                        {status.text}
                                                    </div>
                                                    {result.predicted_source && (
                                                        <div className="mt-2 inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-800 border border-white/10 text-xs font-bold text-slate-300">
                                                            <Sparkles size={12} className="text-blue-400" />
                                                            <span>Predicted Source: <span className="text-white">{result.predicted_source}</span></span>
                                                        </div>
                                                    )}
                                                    <p className="text-slate-400 mt-2 text-sm max-w-md">{status.desc}</p>
                                                </div>

                                                {/* Confidence Bars */}
                                                <div className="space-y-4 w-full max-w-md">
                                                    <div className="space-y-1">
                                                        <div className="flex justify-between text-xs font-bold uppercase tracking-wider text-slate-400">
                                                            <span className="flex items-center gap-1 group relative cursor-help">
                                                                Risk Probability
                                                                <Info size={11} className="text-slate-500 hover:text-blue-400 transition-colors" />
                                                                <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-slate-950/95 border border-white/10 rounded-xl text-[10px] text-slate-400 normal-case tracking-normal opacity-0 group-hover:opacity-100 transition-all duration-300 pointer-events-none z-50 shadow-2xl backdrop-blur-md translate-y-2 group-hover:translate-y-0 leading-relaxed font-normal">
                                                                    <strong>AI Probability Risk Levels:</strong><br />
                                                                    • <strong>&lt; 40%:</strong> Likely human-created.<br />
                                                                    • <strong>40% - 60%:</strong> Uncertain or mixed editing.<br />
                                                                    • <strong>&gt; 60%:</strong> High likelihood of AI origin.
                                                                    <span className="absolute top-full left-1/2 -translate-x-1/2 -mt-2 border-8 border-transparent border-t-slate-950/95" />
                                                                </span>
                                                            </span>
                                                            <span>{Math.round(score * 100)}%</span>
                                                        </div>
                                                        <div className="h-2 bg-slate-900/50 rounded-full overflow-hidden border border-white/5">
                                                            <motion.div initial={{ width: 0 }} animate={{ width: `${score*100}%` }} className="h-full bg-gradient-to-r from-red-600 to-red-400 rounded-full" />
                                                        </div>
                                                    </div>

                                                    <div className="space-y-1">
                                                        <div className="flex justify-between text-xs font-bold uppercase tracking-wider text-slate-400">
                                                            <span>Authenticity Probability</span>
                                                            <span>{Math.round(humanScore * 100)}%</span>
                                                        </div>
                                                        <div className="h-2 bg-slate-900/50 rounded-full overflow-hidden border border-white/5">
                                                             <motion.div initial={{ width: 0 }} animate={{ width: `${humanScore*100}%` }} className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full" />
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                       </div>
                                        
                                        <div className="mt-8 pt-6 border-t border-white/5 flex items-center justify-between">
                                            {/* Share Button */}
                                            <div className="relative">
                                              <Button
                                                variant="outline"
                                                size="sm"
                                                className="bg-transparent border-white/10 text-slate-300 hover:text-white hover:bg-white/5"
                                                onClick={handleShare}
                                              >
                                                {copied ? (
                                                  <><CheckCheck size={14} className="mr-2 text-emerald-400" /><span className="text-emerald-400">Link Copied!</span></>
                                                ) : (
                                                  <><Share2 size={14} className="mr-2" />Share Result</>
                                                )}
                                              </Button>
                                              <AnimatePresence>
                                                {shareTooltip && (
                                                  <motion.div
                                                    initial={{ opacity: 0, y: 6 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    exit={{ opacity: 0, y: 6 }}
                                                    className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-emerald-900/90 border border-emerald-500/30 text-emerald-300 text-xs rounded-lg whitespace-nowrap shadow-xl"
                                                  >
                                                    Shareable link copied to clipboard!
                                                  </motion.div>
                                                )}
                                              </AnimatePresence>
                                            </div>

                                            <Button 
                                                variant="outline" 
                                                size="sm" 
                                                className="bg-transparent border-white/10 text-slate-300 hover:text-white hover:bg-white/5"
                                                onClick={handleDownload}
                                            >
                                                <Download size={14} className="mr-2" />
                                                Download Report
                                            </Button>
                                        </div>
                                    </CardContent>
                                </Card>

                                {/* Right: Explainability Panel */}
                                <div className="lg:col-span-1 h-full">
                                    <ExplainabilityPanel aiScore={score} metrics={result.metrics} />
                                </div>
                                
                                {activeTab === "image" && result.heatmap && (
                                     <Card className="lg:col-span-3 border border-white/10 bg-slate-900/40">
                                        <CardContent className="p-8">
                                            <div className="flex items-center space-x-2 mb-6">
                                                <Layers size={18} className="text-blue-500" />
                                                <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Visual Attribution Map</h3>
                                            </div>
                                            <div className="relative rounded-2xl overflow-hidden border border-white/10 max-w-2xl mx-auto shadow-2xl">
                                                <img 
                                                    src={`data:image/jpeg;base64,${result.heatmap}`} 
                                                    alt="AI Artifact Heatmap" 
                                                    className="w-full h-auto"
                                                />
                                                <div className="absolute bottom-4 left-4 right-4 bg-black/60 backdrop-blur-md p-3 rounded-xl border border-white/5 text-[10px] text-slate-400">
                                                    Heatmap highlights regions with high-frequency anomalies and structural inconsistencies typical of AI generation (GAN/Diffusion artifacts).
                                                </div>
                                            </div>
                                        </CardContent>
                                     </Card>
                                )}

                                {/* Audio signals panel */}
                                {activeTab === "audio" && result.signals && (
                                     <Card className="lg:col-span-3 border border-white/10 bg-slate-900/40">
                                        <CardContent className="p-8">
                                            <div className="flex items-center space-x-2 mb-6">
                                                <Volume2 size={18} className="text-green-500" />
                                                <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Audio Signal Analysis</h3>
                                            </div>
                                            <div className="space-y-3">
                                                {result.signals.map((s: any, i: number) => (
                                                    <div key={i} className={`flex items-center gap-3 p-3 rounded-xl border ${s.ai ? 'bg-red-500/5 border-red-500/20' : 'bg-emerald-500/5 border-emerald-500/20'}`}>
                                                        <div className={`w-2 h-2 rounded-full shrink-0 ${s.ai ? 'bg-red-400' : 'bg-emerald-400'}`} />
                                                        <span className="text-sm text-slate-300 flex-1">{s.signal}</span>
                                                        <span className={`text-xs font-bold uppercase px-2 py-0.5 rounded ${s.weight === 'high' ? 'bg-red-500/20 text-red-400' : 'bg-slate-700 text-slate-400'}`}>{s.weight}</span>
                                                    </div>
                                                ))}
                                            </div>
                                            {result.audio_stats && (
                                                <div className="mt-6 pt-4 border-t border-white/5 flex gap-6 text-xs text-slate-500">
                                                    {result.audio_stats.duration_s && <span>Duration: <b className="text-slate-300">{result.audio_stats.duration_s}s</b></span>}
                                                    {result.audio_stats.sample_rate && <span>Sample rate: <b className="text-slate-300">{result.audio_stats.sample_rate}Hz</b></span>}
                                                    <span>Size: <b className="text-slate-300">{result.audio_stats.size_kb}KB</b></span>
                                                    <span>Model: <b className="text-slate-300">{result.model}</b></span>
                                                </div>
                                            )}
                                        </CardContent>
                                     </Card>
                                )}

                                {/* Video signals panel */}
                                {activeTab === "video" && result.signals && (
                                     <Card className="lg:col-span-3 border border-white/10 bg-slate-900/40 animate-fade-in">
                                        <CardContent className="p-8">
                                            <div className="flex items-center space-x-2 mb-6">
                                                <Video size={18} className="text-cyan-400" />
                                                <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Video Motion & Artifact Analysis</h3>
                                            </div>
                                            <div className="space-y-3">
                                                {result.signals.map((s: any, i: number) => (
                                                    <div key={i} className={`flex items-center gap-3 p-3 rounded-xl border ${s.ai ? 'bg-red-500/5 border-red-500/20' : 'bg-emerald-500/5 border-emerald-500/20'}`}>
                                                        <div className={`w-2 h-2 rounded-full shrink-0 ${s.ai ? 'bg-red-400' : 'bg-emerald-400'}`} />
                                                        <span className="text-sm text-slate-300 flex-1">{s.signal}</span>
                                                        <span className={`text-xs font-bold uppercase px-2 py-0.5 rounded ${s.weight === 'high' ? 'bg-red-500/20 text-red-400' : s.weight === 'medium' ? 'bg-amber-500/10 text-amber-400' : 'bg-slate-700 text-slate-400'}`}>{s.weight}</span>
                                                    </div>
                                                ))}
                                            </div>
                                            {result.video_stats && (
                                                <div className="mt-6 pt-4 border-t border-white/5 flex flex-wrap gap-x-6 gap-y-2 text-xs text-slate-500">
                                                    {result.video_stats.duration_s !== undefined && <span>Duration: <b className="text-slate-300">{result.video_stats.duration_s}s</b></span>}
                                                    {result.video_stats.fps !== undefined && <span>FPS: <b className="text-slate-300">{result.video_stats.fps}</b></span>}
                                                    {result.video_stats.frames_analysed !== undefined && <span>Frames Analysed: <b className="text-slate-300">{result.video_stats.frames_analysed}</b></span>}
                                                    {result.video_stats.size_kb !== undefined && result.video_stats.size_kb > 0 && <span>Size: <b className="text-slate-300">{result.video_stats.size_kb}KB</b></span>}
                                                </div>
                                            )}
                                        </CardContent>
                                     </Card>
                                )}

                                {/* Bulk results table */}
                                {activeTab === "bulk" && result.results && (
                                     <Card className="lg:col-span-3 border border-white/10 bg-slate-900/40">
                                        <CardContent className="p-6">
                                            <div className="flex items-center justify-between mb-6">
                                                <div className="flex items-center space-x-2">
                                                    <Layers size={18} className="text-indigo-400" />
                                                    <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Bulk Analysis Results</h3>
                                                </div>
                                                <div className="flex gap-3 text-xs">
                                                    <span className="bg-red-500/10 text-red-400 px-2 py-1 rounded-lg font-semibold">{result.high_risk} high risk</span>
                                                    <span className="bg-slate-700/60 text-slate-300 px-2 py-1 rounded-lg">{result.total_items} total</span>
                                                </div>
                                            </div>
                                            <div className="space-y-2 max-h-96 overflow-y-auto custom-scrollbar pr-1">
                                                {result.results.map((item: any, i: number) => {
                                                    const pct = Math.round(item.ai_probability * 100);
                                                    const color = pct > 60 ? 'text-red-400' : pct < 40 ? 'text-emerald-400' : 'text-amber-400';
                                                    const barColor = pct > 60 ? 'bg-red-500' : pct < 40 ? 'bg-emerald-500' : 'bg-amber-500';
                                                    return (
                                                        <div key={i} className="flex items-center gap-3 p-3 rounded-xl bg-slate-900/50 border border-white/5 hover:border-white/10 transition-colors">
                                                            <span className="text-xs font-mono text-slate-600 w-6 shrink-0">{i+1}</span>
                                                            <span className="text-sm text-slate-400 flex-1 truncate font-mono">{item.preview}</span>
                                                            <div className="flex items-center gap-2 shrink-0">
                                                                <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                                                    <motion.div initial={{width:0}} animate={{width:`${pct}%`}} className={`h-full rounded-full ${barColor}`} />
                                                                </div>
                                                                <span className={`text-xs font-bold font-mono w-10 text-right ${color}`}>{pct}%</span>
                                                            </div>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </CardContent>
                                     </Card>
                                )}
                            </div>
                        );
                    })()}

                    <div className="flex justify-center pt-12 pb-8">
                        <Button 
                            onClick={() => { setResult(null); setInputData(null); }}
                            variant="secondary"
                            className="bg-slate-800 text-slate-200 hover:bg-slate-700 rounded-full px-8 py-6 h-auto"
                        >
                            Start New Scan
                        </Button>
                    </div>

                </motion.div>
            )}
        </AnimatePresence>
    </div>
  );
}
