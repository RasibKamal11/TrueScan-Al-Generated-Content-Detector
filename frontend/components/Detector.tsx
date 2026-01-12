"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, Image as ImageIcon, Video, Link as LinkIcon, ScanLine, Loader2, History, ChevronRight, Check, Upload, AlertCircle, Sparkles, Newspaper } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { CircularProgress } from "./CircularProgress";
import { HistorySidebar, HistoryItem } from "./HistorySidebar";
import ExplainabilityPanel from "./ExplainabilityPanel";
import { Download, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import jsPDF from "jspdf";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export default function Detector() {
  const [activeTab, setActiveTab] = useState<"text" | "url" | "image" | "video" | "fake_news">("text");
  const [inputData, setInputData] = useState<string | File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);

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

  const saveToHistory = (res: any, data: string | File) => {
    const newItem: HistoryItem = {
        id: crypto.randomUUID(),
        type: activeTab,
        content: typeof data === 'string' ? data : data.name,
        score: res.ai_probability ?? res.score,
        timestamp: Date.now()
    };
    
    const newHistory = [newItem, ...history].slice(0, 50); // Keep last 50
    setHistory(newHistory);
    localStorage.setItem("scan_history", JSON.stringify(newHistory));
  };

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
    if (score > 0.6) { verdict = activeTab === "fake_news" ? "Likely Fake/Misleading" : "Likely AI-Generated"; color = [239, 68, 68]; } // Red
    else if (score < 0.4) { verdict = activeTab === "fake_news" ? "Likely Reliable" : "Likely Human"; color = [16, 185, 129]; } // Emerald
    
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
    doc.setFontSize(8);
    doc.setTextColor(150);
    doc.text("TrueScan AI-Generated Content Detector - Enterprise Grade Verification", pageWidth / 2, doc.internal.pageSize.getHeight() - 10, { align: "center" });
    
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
    
    // Simulate deeper scan visual delay
    await new Promise(r => setTimeout(r, 800)); 
    
    // Determine Endpoint
    // Use 'text' endpoint for fake_news if backend doesn't support it directly
    const endpointType = activeTab === "fake_news" ? "text" : activeTab;
    const endpoint = `http://127.0.0.1:8000/detect/${endpointType}`;
    
    try {
        let response;
        let body;
        let headers = {};

        if (activeTab === "text" || activeTab === "fake_news") {
            headers = { "Content-Type": "application/json" };
            body = JSON.stringify({ text: inputData, detailed: true });
        } else if (activeTab === "url") {
            headers = { "Content-Type": "application/json" };
            body = JSON.stringify({ url: inputData });
        } else {
            const formData = new FormData();
            formData.append("file", inputData as File);
            body = formData;
        }

        response = await fetch(endpoint, {
            method: "POST",
            headers: (activeTab === "image" || activeTab === "video") ? undefined : headers,
            body: body,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || `Server returned ${response.status}`);
        }

        const data = await response.json();
        setResult(data);
        saveToHistory(data, inputData as string | File);
        
    } catch (error) {
        console.error("Analysis failed:", error);
        alert(`Analysis failed: ${error}`);
    } finally {
        setLoading(false);
    }
  };

  const tabIcons = {
    text: FileText,
    url: LinkIcon,
    image: ImageIcon,
    video: Video,
    fake_news: Newspaper,
  };

  const renderHighlightedText = () => {
    if (!result) return null;
    const textContent = typeof inputData === "string" ? inputData : "Analyzed Content";
    
    if (result.sentences && Array.isArray(result.sentences)) {
        return (
            <div className="bg-slate-950/50 rounded-xl p-6 font-mono text-sm leading-relaxed text-slate-300 max-h-96 overflow-y-auto border border-white/5 whitespace-pre-wrap custom-scrollbar">
                {result.sentences.map((item: any, idx: number) => {
                    const probability = item.ai_probability ?? item.score ?? 0;
                    let className = "";
                    if (probability > 0.8) className = "bg-red-500/20 text-red-100 px-1 rounded mx-0.5 border-b border-red-500/30";
                    else if (probability > 0.6) className = "bg-yellow-500/20 text-yellow-100 px-1 rounded mx-0.5 border-b border-yellow-500/30";
                    
                    return (
                        <span key={idx} className={className}>
                            {item.text}
                        </span>
                    );
                })}
            </div>
        );
    }

    return (
        <div className="bg-slate-950/50 rounded-xl p-6 font-mono text-sm leading-relaxed text-slate-300 max-h-96 overflow-y-auto border border-white/5 whitespace-pre-wrap">
            {textContent}
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

                        {/* Tabs */}
                        <div className="flex justify-center mb-10 relative z-10">
                            <div className="flex p-1.5 rounded-2xl bg-black/40 border border-white/5 backdrop-blur-md">
                                {(["text", "url", "image", "video", "fake_news"] as const).map((tab) => {
                                    const Icon = tabIcons[tab];
                                    return (
                                        <button
                                            key={tab}
                                            onClick={() => { setActiveTab(tab); setResult(null); setInputData(null); }}
                                            className={cn(
                                                "flex items-center space-x-2 px-6 py-2.5 rounded-xl text-sm font-semibold transition-all relative z-10",
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
                                                <span className="capitalize">{tab === "fake_news" ? "Fake News" : tab}</span>
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
                                    {activeTab === "text" || activeTab === "fake_news" ? (
                                        <div className="relative group h-full">
                                            <textarea
                                                className="relative w-full h-80 bg-slate-900/40 border border-white/5 rounded-2xl p-8 text-lg text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500/50 transition-all resize-none placeholder:text-slate-600 font-light leading-relaxed font-mono shadow-inner custom-scrollbar"
                                                placeholder={activeTab === "fake_news" ? "Paste the article or news text here..." : "Paste the text you want to analyze..."}
                                                value={typeof inputData === "string" ? inputData : ""}
                                                onChange={(e) => setInputData(e.target.value)}
                                            />
                                            <div className="absolute bottom-4 right-4 text-xs font-mono text-slate-600 bg-black/40 px-2 py-1 rounded">
                                                {(typeof inputData === 'string' ? inputData.length : 0)} chars
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

                        {/* Analyze Button */}
                        <div className="flex flex-col items-center space-y-4">
                            <Button
                                size="lg"
                                onClick={analyze}
                                disabled={loading || !inputData}
                                variant="glow"
                                className="w-full max-w-sm h-14 text-lg rounded-xl relative overflow-hidden group shadow-[0_0_40px_rgba(59,130,246,0.3)] hover:shadow-[0_0_60px_rgba(59,130,246,0.5)] transition-all duration-500"
                            >
                                {loading ? (
                                    <span className="flex items-center space-x-3">
                                        <Loader2 className="animate-spin" />
                                        <span>Analyzing Content...</span>
                                    </span>
                                ) : (
                                    <span className="flex items-center space-x-2">
                                        <Sparkles size={18} /> 
                                        <span>Run Smart Scan</span>
                                    </span>
                                )}
                            </Button>
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
                        
                        // Customize Verdict for Fake News vs AI
                        if (activeTab === "fake_news") {
                             if (score > 0.6) status = { text: "Likely Misinformation", color: "text-red-400", bg: "bg-red-500", border: "border-red-500/20", shadow:"shadow-red-500/20", desc: "Our models detected clear signs of fabricated content." };
                             else if (score < 0.4) status = { text: "Likely Authentic", color: "text-emerald-400", bg: "bg-emerald-500", border: "border-emerald-500/20", shadow:"shadow-emerald-500/20", desc: "Content appears consistent with verified reporting." };
                        } else {
                             if (score > 0.6) status = { text: "Likely AI-Generated", color: "text-red-400", bg: "bg-red-500", border: "border-red-500/20", shadow:"shadow-red-500/20", desc: "Strong AI signatures detected." };
                             else if (score < 0.4) status = { text: "Likely Human", color: "text-emerald-400", bg: "bg-emerald-500", border: "border-emerald-500/20", shadow:"shadow-emerald-500/20", desc: "Consistent with human writing patterns." };
                        }

                        
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
                                                    <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mt-1">Risk Score</span>
                                                </div>
                                            </div>

                                            <div className="flex-1 space-y-6 text-center md:text-left w-full">
                                                <div>
                                                    <h3 className="text-sm font-bold uppercase tracking-widest text-slate-500 mb-2">Detection Result</h3>
                                                    <div className={cn("text-4xl font-bold tracking-tight", status.color)}>
                                                        {status.text}
                                                    </div>
                                                    <p className="text-slate-400 mt-2 text-sm max-w-md">{status.desc}</p>
                                                </div>

                                                {/* Confidence Bars */}
                                                <div className="space-y-4 w-full max-w-md">
                                                    <div className="space-y-1">
                                                        <div className="flex justify-between text-xs font-bold uppercase tracking-wider text-slate-400">
                                                            <span>Risk Probability</span>
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
                                        
                                        <div className="mt-8 pt-6 border-t border-white/5 flex justify-end">
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
                                    <ExplainabilityPanel aiScore={score} />
                                </div>
                                
                                {/* Bottom: Detailed Text Analysis */}
                                {activeTab !== "image" && activeTab !== "video" && (
                                     <Card className="lg:col-span-3 border border-white/10 bg-slate-900/40">
                                        <CardContent className="p-8">
                                            <div className="flex items-center space-x-2 mb-6">
                                                <ScanLine size={18} className="text-blue-500" />
                                                <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Detailed Segment Analysis</h3>
                                            </div>
                                            {renderHighlightedText()}
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
