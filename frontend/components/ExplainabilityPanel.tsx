"use client";

import { motion } from "framer-motion";
import { Info, BarChart3, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";

interface MetricProps {
    label: string;
    value: string;
    score: number; // 0-100
    color: string;
    description: string;
    trend?: "human" | "ai" | "neutral";
    delay?: number;
}

const Metric = ({ label, value, score, color, description, trend = "neutral", delay = 0 }: MetricProps) => {
    const TrendIcon = trend === "human" ? TrendingUp : trend === "ai" ? TrendingDown : Minus;
    const trendColor = trend === "human" ? "text-emerald-400" : trend === "ai" ? "text-red-400" : "text-slate-500";

    return (
        <motion.div
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay }}
            className="space-y-3 group"
        >
            <div className="flex justify-between items-end">
                <div className="flex items-center space-x-2">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 group-hover:text-slate-300 transition-colors">{label}</span>
                    <div className="relative">
                        <Info size={12} className="text-slate-600 cursor-help hover:text-blue-400 transition-colors" />
                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-60 p-4 bg-slate-900/95 border border-white/10 rounded-xl text-xs text-slate-300 opacity-0 group-hover:opacity-100 transition-all pointer-events-none z-50 shadow-2xl backdrop-blur-md translate-y-2 group-hover:translate-y-0">
                            {description}
                            <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-2 border-8 border-transparent border-t-slate-900/95" />
                        </div>
                    </div>
                </div>
                <div className="flex items-center space-x-2">
                    <TrendIcon size={11} className={trendColor} />
                    <span className="text-sm font-bold text-white font-mono">{value}</span>
                </div>
            </div>
            <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden border border-white/5">
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${score}%` }}
                    transition={{ duration: 1.5, ease: [0.22, 1, 0.36, 1], delay }}
                    className={`h-full rounded-full shadow-[0_0_10px_currentColor] ${color}`}
                />
            </div>
        </motion.div>
    );
};

interface ExplainabilityPanelProps {
    aiScore: number;
    metrics?: {
        perplexity?: number;
        burstiness?: number;
        vocabulary_richness?: number;
        neural_repetition?: number;
        sentence_count?: number;
        avg_sentence_length?: number;
        word_count?: number;
    };
}

export default function ExplainabilityPanel({ aiScore, metrics }: ExplainabilityPanelProps) {
    // Perplexity: LOW = predictable = AI, HIGH = unpredictable = human
    // Backend returns 0-100 where 100 = very high perplexity (human-like)
    const perplexityScore = metrics?.perplexity ?? (aiScore > 0.5 ? 20 : 85);
    const perplexityLabel = perplexityScore > 60 ? "High (Human)" : perplexityScore > 35 ? "Medium" : "Low (AI)";
    const perplexityTrend: "human" | "ai" | "neutral" = perplexityScore > 60 ? "human" : perplexityScore > 35 ? "neutral" : "ai";

    // Burstiness: HIGH = varied = human, LOW = uniform = AI
    // Backend returns 0-100 where 100 = very bursty (human-like)
    const burstinessScore = metrics?.burstiness ?? (aiScore > 0.5 ? 28 : 92);
    const burstinessLabel = burstinessScore > 60 ? "High (Human)" : burstinessScore > 35 ? "Medium" : "Low (AI)";
    const burstinessTrend: "human" | "ai" | "neutral" = burstinessScore > 60 ? "human" : burstinessScore > 35 ? "neutral" : "ai";

    // Vocabulary Richness: HIGH = diverse = human, LOW = repetitive = AI
    // Backend returns 0-100 (TTR * 100)
    const vocabScore = metrics?.vocabulary_richness ?? (aiScore > 0.5 ? 30 : 75);
    const vocabLabel = vocabScore > 60 ? "Rich (Human)" : vocabScore > 35 ? "Moderate" : "Limited (AI)";
    const vocabTrend: "human" | "ai" | "neutral" = vocabScore > 60 ? "human" : vocabScore > 35 ? "neutral" : "ai";

    // Neural Repetition: from backend bigram analysis, or fallback from AI score
    const neuralScore = metrics?.neural_repetition ?? (aiScore * 100);
    const neuralLabel = neuralScore > 60 ? "Machine-like" : neuralScore > 35 ? "Uncertain" : "Natural";
    const neuralTrend: "human" | "ai" | "neutral" = neuralScore > 60 ? "ai" : neuralScore > 35 ? "neutral" : "human";

    const statsItems = [
        metrics?.word_count ? `${metrics.word_count} words` : null,
        metrics?.sentence_count ? `${metrics.sentence_count} sentences` : null,
        metrics?.avg_sentence_length ? `avg ${metrics.avg_sentence_length} words/sentence` : null,
    ].filter(Boolean);

    return (
        <Card className="h-full border border-white/10 bg-slate-900/40 shadow-xl">
            <CardHeader className="pb-2">
                <div className="flex items-center space-x-2">
                    <BarChart3 size={18} className="text-purple-500" />
                    <CardTitle className="text-sm font-bold uppercase tracking-widest text-slate-400">Deep Analysis</CardTitle>
                </div>
            </CardHeader>
            <CardContent className="space-y-7 pt-4">
                <div className="space-y-6">
                    <Metric
                        label="Perplexity"
                        value={perplexityLabel}
                        score={perplexityScore}
                        color="bg-purple-500 text-purple-500"
                        description="How 'surprised' the model is by the text. AI-generated text is predictable (low perplexity). Human text is more surprising (high perplexity)."
                        trend={perplexityTrend}
                        delay={0.1}
                    />
                    <Metric
                        label="Burstiness"
                        value={burstinessLabel}
                        score={burstinessScore}
                        color="bg-blue-500 text-blue-500"
                        description="Measures variation in sentence lengths. Humans write in varied rhythms (high burstiness). AI tends to produce uniformly structured sentences."
                        trend={burstinessTrend}
                        delay={0.2}
                    />
                    <Metric
                        label="Vocab Richness"
                        value={vocabLabel}
                        score={vocabScore}
                        color="bg-cyan-500 text-cyan-500"
                        description="Type-Token Ratio (TTR): ratio of unique words to total words. AI often reuses the same vocabulary. Humans tend to use a wider range of words."
                        trend={vocabTrend}
                        delay={0.3}
                    />
                    <Metric
                        label="Neural Pattern"
                        value={neuralLabel}
                        score={neuralScore}
                        color="bg-red-500 text-red-500"
                        description="Detects repetitive neural network patterns common in LLM outputs — overly smooth transitions, formulaic structure, and uniform tone."
                        trend={neuralTrend}
                        delay={0.4}
                    />
                </div>

                {statsItems.length > 0 && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.6 }}
                        className="pt-4 border-t border-white/5 space-y-1"
                    >
                        <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-600 mb-2">Text Stats</p>
                        {statsItems.map((stat, i) => (
                            <div key={i} className="flex items-center space-x-2 text-xs text-slate-500">
                                <div className="w-1 h-1 rounded-full bg-slate-600" />
                                <span>{stat}</span>
                            </div>
                        ))}
                    </motion.div>
                )}

                <div className="pt-2 border-t border-white/5">
                    <p className="text-[10px] text-slate-600 leading-relaxed font-medium">
                        *Analysis uses multi-dimensional linguistic features and transformer-based embeddings against 1.5B parameter patterns.
                    </p>
                </div>
            </CardContent>
        </Card>
    );
}
