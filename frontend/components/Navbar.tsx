"use client";

import { ShieldCheck, Github, Menu, Sparkles } from "lucide-react";
import { Button } from "./ui/Button";
import { motion } from "framer-motion";

export default function Navbar() {
  return (
    <motion.nav 
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      className="fixed top-6 left-0 right-0 z-50 px-6 flex justify-center pointer-events-none"
    >
      <div className="bg-slate-950/80 backdrop-blur-xl border border-white/10 shadow-2xl shadow-black/50 rounded-full px-6 py-3 pointer-events-auto flex items-center gap-8 min-w-[320px] md:min-w-[600px] justify-between group hover:border-white/20 transition-colors">
        
        {/* Logo */}
        <div className="flex items-center space-x-2">
          <div className="relative">
            <div className="absolute inset-0 bg-blue-500 blur-lg opacity-20 group-hover:opacity-40 transition-opacity" />
            <ShieldCheck className="text-blue-500 relative z-10" size={24} />
          </div>
          <span className="font-bold text-slate-100 tracking-tight text-lg">
            TrueScan <span className="text-blue-500">AI-Generated Content Detector</span>
          </span>
        </div>
        
        {/* Desktop Links */}
        <div className="hidden md:flex items-center space-x-1">
          {["Features", "Enterprise", "API"].map((item) => (
            <Button 
                key={item} 
                variant="ghost" 
                size="sm" 
                className="text-slate-400 hover:text-white hover:bg-white/5 rounded-full px-4 font-medium"
            >
              {item}
            </Button>
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-3">
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="hidden md:block text-slate-400 hover:text-white transition-colors">
            <Github size={20} />
          </a>
          <div className="w-px h-6 bg-white/10 hidden md:block" />
          <Button variant="glow" size="sm" className="hidden md:flex rounded-full px-5 text-xs font-bold uppercase tracking-wide">
            <Sparkles size={14} className="mr-2" />
            Get Started
          </Button>

          {/* Mobile Menu */}
          <Button variant="ghost" size="icon" className="md:hidden rounded-full text-white">
            <Menu size={20} />
          </Button>
        </div>
      </div>
    </motion.nav>
  );
}
