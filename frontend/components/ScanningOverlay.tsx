"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Brain, Cpu, BarChart3, CheckCircle2, Volume2, Code2 } from "lucide-react";

const SCAN_STEPS: Record<string, { messages: string[]; icon: any }> = {
  text: {
    icon: Brain,
    messages: [
      "Tokenizing input text...",
      "Analyzing syntax patterns...",
      "Running RoBERTa inference...",
      "Computing perplexity score...",
      "Measuring burstiness coefficient...",
      "Evaluating vocabulary richness...",
      "Aggregating sentence-level signals...",
      "Finalizing detection result...",
    ],
  },

  image: {
    icon: Cpu,
    messages: [
      "Preprocessing image tensor...",
      "Applying ResNet feature extraction...",
      "Detecting GAN artifacts...",
      "Analyzing pixel coherence...",
      "Checking for SDXL signatures...",
      "Evaluating edge consistency...",
      "Running classifier head...",
      "Finalizing detection result...",
    ],
  },
  video: {
    icon: Cpu,
    messages: [
      "Extracting video keyframes...",
      "Processing frame sequences...",
      "Detecting temporal inconsistencies...",
      "Analyzing motion artifacts...",
      "Checking compression patterns...",
      "Running frame-by-frame analysis...",
      "Aggregating temporal scores...",
      "Finalizing detection result...",
    ],
  },
  url: {
    icon: Brain,
    messages: [
      "Fetching URL content...",
      "Parsing HTML structure...",
      "Extracting text content...",
      "Tokenizing extracted text...",
      "Running NLP analysis...",
      "Computing credibility metrics...",
      "Evaluating writing patterns...",
      "Finalizing detection result...",
    ],
  },
  audio: {
    icon: Volume2,
    messages: [
      "Decoding audio stream...",
      "Extracting pitch (F0) contour...",
      "Analysing spectral flatness...",
      "Computing zero-crossing rate variance...",
      "Running MFCC uniformity analysis...",
      "Detecting prosody patterns...",
      "Blending feature scores...",
      "Finalizing detection result...",
    ],
  },
  code: {
    icon: Code2,
    messages: [
      "Parsing code structure...",
      "Detecting programming language...",
      "Analysing comment verbosity...",
      "Checking naming conventions...",
      "Detecting boilerplate patterns...",
      "Scanning for human signals...",
      "Computing cognitive complexity...",
      "Finalizing detection result...",
    ],
  },
  bulk: {
    icon: BarChart3,
    messages: [
      "Parsing uploaded file...",
      "Splitting into text segments...",
      "Running batch inference...",
      "Scoring each segment...",
      "Computing aggregate risk score...",
      "Generating full report...",
      "Finalizing results...",
    ],
  },
};

interface ScanningOverlayProps {
  isLoading: boolean;
  activeTab: string;
  wsProgress?: { step: number; total: number; message: string; progress: number } | null;
}

export default function ScanningOverlay({ isLoading, activeTab, wsProgress }: ScanningOverlayProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const steps = SCAN_STEPS[activeTab] ?? SCAN_STEPS.text;
  const Icon = steps.icon;

  // Use simulated timer only when no real WS progress is available
  useEffect(() => {
    if (wsProgress !== undefined) return; // WS is active, don't simulate
    if (!isLoading) {
      setCurrentStep(0);
      return;
    }
    setCurrentStep(0);
    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= steps.messages.length - 1) return prev;
        return prev + 1;
      });
    }, 700);
    return () => clearInterval(interval);
  }, [isLoading, activeTab, steps.messages.length, wsProgress]);

  // Sync step from real WS progress
  const displayStep = wsProgress ? wsProgress.step : currentStep;
  const displayMessage = wsProgress ? wsProgress.message : steps.messages[displayStep];
  const displayProgress = wsProgress ? wsProgress.progress : ((displayStep + 1) / steps.messages.length) * 100;

  if (!isLoading) return null;

  const progress = displayProgress;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="flex flex-col items-center space-y-6 py-4"
    >
      {/* Animated Icon */}
      <div className="relative">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className="w-16 h-16 rounded-full border-2 border-blue-500/30 border-t-blue-500 flex items-center justify-center"
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            <Icon size={22} className="text-blue-400" />
          </motion.div>
        </div>
      </div>

      {/* Step Message */}
      <div className="text-center space-y-2 min-h-[56px]">
        <AnimatePresence mode="wait">
          <motion.p
            key={displayStep}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3 }}
            className="text-sm font-medium text-slate-300 font-mono"
          >
            {displayMessage}
          </motion.p>
        </AnimatePresence>
        <p className="text-xs text-slate-500">
          Step {displayStep + 1} of {wsProgress ? wsProgress.total : steps.messages.length}
        </p>
      </div>

      {/* Progress Bar */}
      <div className="w-64 space-y-2">
        <div className="h-1 bg-slate-800 rounded-full overflow-hidden border border-white/5">
          <motion.div
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="h-full bg-gradient-to-r from-blue-600 to-violet-500 rounded-full shadow-[0_0_8px_rgba(59,130,246,0.6)]"
          />
        </div>
        <div className="flex justify-between text-[10px] text-slate-600 font-mono">
          <span>Analyzing...</span>
          <span>{Math.round(progress)}%</span>
        </div>
      </div>

      {/* Steps Dots */}
      <div className="flex space-x-1.5">
        {steps.messages.map((_, i) => (
          <motion.div
            key={i}
            animate={{
              backgroundColor:
                i < currentStep
                  ? "#10b981"
                  : i === currentStep
                  ? "#3b82f6"
                  : "#1e293b",
              scale: i === currentStep ? 1.3 : 1,
            }}
            transition={{ duration: 0.3 }}
            className="w-1.5 h-1.5 rounded-full"
          />
        ))}
      </div>
    </motion.div>
  );
}
