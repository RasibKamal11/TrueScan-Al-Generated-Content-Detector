"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Clock, FileText, Link as LinkIcon, X, ChevronRight, Trash2 } from "lucide-react";
import { useState, useEffect } from "react";

export type HistoryItem = {
  id: string;
  type: "text" | "url" | "image" | "video";
  content: string; // Preview text or filename or URL
  score: number;
  timestamp: number;
};

interface HistorySidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (item: HistoryItem) => void;
  items: HistoryItem[];
  onClear: () => void;
}

export function HistorySidebar({ isOpen, onClose, onSelect, items, onClear }: HistorySidebarProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          />

          {/* Sidebar Panel */}
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 bottom-0 w-full max-w-sm bg-slate-900 border-l border-white/10 shadow-2xl z-50 p-6 overflow-hidden flex flex-col"
          >
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center space-x-2 text-slate-100 font-bold text-xl">
                <Clock className="text-blue-500" />
                <span>Scan History</span>
              </div>
              <button
                onClick={onClose}
                className="p-2 hover:bg-white/10 rounded-full transition-colors"
              >
                <X size={20} className="text-slate-400" />
              </button>
            </div>

            {items.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center text-slate-500 space-y-4">
                <Clock size={48} className="opacity-20" />
                <p>No recent scans found.</p>
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
                {items.map((item) => (
                  <motion.div
                    key={item.id}
                    layoutId={item.id}
                    onClick={() => onSelect(item)}
                    className="group p-4 rounded-xl bg-white/5 border border-white/5 hover:border-blue-500/30 hover:bg-blue-500/5 transition-all cursor-pointer relative overflow-hidden"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-3">
                        <div className={`p-2 rounded-lg ${
                            item.type === 'url' ? 'bg-orange-500/10 text-orange-400' : 'bg-blue-500/10 text-blue-400'
                        }`}>
                            {item.type === 'url' ? <LinkIcon size={16} /> : <FileText size={16} />}
                        </div>
                        <div>
                            <div className="font-medium text-slate-200 line-clamp-1 text-sm">
                                {item.type === 'url' ? item.content : `Text Scan`}
                            </div>
                            <div className="text-xs text-slate-500 flex items-center space-x-2 mt-1">
                                <span>{new Date(item.timestamp).toLocaleDateString()}</span>
                                <span>•</span>
                                <span className={item.score > 0.5 ? "text-red-400" : "text-emerald-400"}>
                                    {Math.round(item.score * 100)}% AI
                                </span>
                            </div>
                        </div>
                      </div>
                      <ChevronRight size={16} className="text-slate-600 group-hover:text-blue-400 transition-colors" />
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
            
            {items.length > 0 && (
                <div className="pt-6 mt-2 border-t border-white/10">
                    <button 
                        onClick={onClear}
                        className="w-full py-3 rounded-xl border border-red-500/20 text-red-400 hover:bg-red-500/10 transition-colors flex items-center justify-center space-x-2 text-sm font-medium"
                    >
                        <Trash2 size={16} />
                        <span>Clear History</span>
                    </button>
                </div>
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
