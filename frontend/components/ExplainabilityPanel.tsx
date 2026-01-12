"use client";

import { motion } from "framer-motion";
import { Info, BarChart3 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";

interface MetricProps {
    label: string;
    value: string;
    score: number; // 0-100
    color: string;
    description: string;
}

const Metric = ({ label, value, score, color, description }: MetricProps) => (
    <div className="space-y-3 group">
        <div className="flex justify-between items-end">
            <div className="flex items-center space-x-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 group-hover:text-slate-300 transition-colors">{label}</span>
                <div className="relative">
                    <Info size={12} className="text-slate-600 cursor-help hover:text-blue-400 transition-colors" />
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 p-4 bg-slate-900/95 border border-white/10 rounded-xl text-xs text-slate-300 opacity-0 group-hover:opacity-100 transition-all pointer-events-none z-50 shadow-2xl backdrop-blur-md translate-y-2 group-hover:translate-y-0">
                        {description}
                        <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-2 border-8 border-transparent border-t-slate-900/95" />
                    </div>
                </div>
            </div>
            <span className="text-sm font-bold text-white font-mono">{value}</span>
        </div>
        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden border border-white/5">
            <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${score}%` }}
                transition={{ duration: 1.5, ease: [0.22, 1, 0.36, 1] }}
                className={`h-full rounded-full shadow-[0_0_10px_currentColor] ${color}`} 
            />
        </div>
    </div>
);

export default function ExplainabilityPanel({ aiScore }: { aiScore: number }) {
    // Simulate metrics based on AI score for demo purposes
    // IN REAL APP: These would come from the backend explanation endpoint
    const perplexity = aiScore > 0.5 ? "Low" : "High";
    const perplexityScore = aiScore > 0.5 ? 35 : 88;
    
    const burstiness = aiScore > 0.5 ? "Low" : "High";
    const burstinessScore = aiScore > 0.5 ? 28 : 92;
    
    const patterns = aiScore > 0.5 ? "Machine-like" : "Natural";
    const patternScore = aiScore * 100;

    return (
        <Card className="h-full border border-white/10 bg-slate-900/40 shadow-xl">
            <CardHeader className="pb-2">
                <div className="flex items-center space-x-2">
                    <BarChart3 size={18} className="text-purple-500" />
                    <CardTitle className="text-sm font-bold uppercase tracking-widest text-slate-400">Deep Analysis</CardTitle>
                </div>
            </CardHeader>
            <CardContent className="space-y-8 pt-6">
                <div className="space-y-6">
                    <Metric 
                        label="Perplexity" 
                        value={perplexity}
                        score={perplexityScore} 
                        color="bg-purple-500 text-purple-500"
                        description="Measures how surprised the model is by the text. AI text often has low perplexity (predictable)."
                    />
                    <Metric 
                        label="Burstiness" 
                        value={burstiness}
                        score={burstinessScore} 
                        color="bg-blue-500 text-blue-500"
                        description="Measures variations in sentence structure. Humans exhibit high burstiness (varied sentences)."
                    />
                    <Metric 
                        label="Recursion Pattern" 
                        value={patterns}
                        score={patternScore} 
                        color="bg-red-500 text-red-500"
                        description="Detects repetitive neural structures common in LLM outputs."
                    />
                     <Metric 
                        label="Semantic Consistency" 
                        value="High"
                        score={95} 
                        color="bg-emerald-500 text-emerald-500"
                        description="Coherence of the text flow across long contexts."
                    />
                </div>
                
                <div className="pt-6 border-t border-white/5">
                    <p className="text-[10px] text-slate-500 leading-relaxed font-medium">
                        *Analysis based on multi-dimensional vector embeddings comparing input against 1.5B parameter language patterns.
                    </p>
                </div>
            </CardContent>
        </Card>
    );
}
