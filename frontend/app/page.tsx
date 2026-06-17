"use client";

import { motion } from "framer-motion";
import Detector from "@/components/Detector";
import Background from "@/components/Background";
import Navbar from "@/components/Navbar";
import StatsBar from "@/components/StatsBar";
import { Button } from "@/components/ui/Button";
import { ArrowRight, Sparkles, CheckCircle2 } from "lucide-react";

export default function Home() {
  const scrollToDetector = () => {
    const element = document.getElementById('detector-section');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <>
      <Background />
      <Navbar />
      
      <main className="relative z-10 min-h-screen flex flex-col items-center justify-center p-4 pt-32 pb-20">
        
        <div className="w-full max-w-7xl flex flex-col items-center space-y-24">
          
          {/* Hero Section */}
          <div className="text-center space-y-8 max-w-5xl relative flex flex-col items-center">
            {/* Badge */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass px-4 py-1.5 rounded-full inline-flex items-center space-x-2 border-blue-500/20 shadow-[0_0_20px_rgba(59,130,246,0.15)] mb-6"
            >
              <Sparkles size={14} className="text-blue-400" />
              <span className="text-xs font-semibold tracking-widest text-blue-100 uppercase">New Generation AI Detection</span>
            </motion.div>
            
            <h1 className="text-5xl md:text-7xl font-black tracking-tighter leading-tight text-white mb-6">
               TrueScan
               <span className="text-blue-500 block text-3xl md:text-5xl mt-2 tracking-normal">AI-Generated Content Detector</span>
            </h1>
            
            {/* Description */}
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed font-light"
            >
              The enterprise standard for distinguishing human creativity from artificial generation. Analyze text, images, and video with military-grade precision.
            </motion.p>

            {/* CTA Group */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="flex flex-col sm:flex-row items-center gap-4 pt-4"
            >
              <Button 
                size="lg" 
                variant="glow" 
                className="rounded-full px-8 text-base"
                onClick={scrollToDetector}
              >
                Start Detection <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
              <Button size="lg" variant="ghost" className="rounded-full px-8 text-base text-slate-300 hover:text-white">
                View Documentation
              </Button>
            </motion.div>

            {/* Stats/Trust */}
            <div className="pt-12 grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-16 opacity-50 grayscale hover:grayscale-0 transition-all duration-700">
               {["99.9% Accuracy", "Enterprise Ready", "SOC2 Compliant", "Real-time API"].map((stat) => (
                  <div key={stat} className="flex items-center space-x-2">
                     <CheckCircle2 size={16} className="text-blue-500" />
                     <span className="text-sm font-semibold text-slate-300">{stat}</span>
                  </div>
               ))}
            </div>
          </div>
          
          {/* Main App Block */}
          <motion.div 
            id="detector-section"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.8 }}
            className="w-full relative z-20 space-y-4"
          >
            <div className="absolute inset-x-0 -top-40 -bottom-40 bg-gradient-to-b from-blue-500/5 to-transparent blur-3xl -z-10 rounded-full opacity-30 pointer-events-none" />
            {/* Live Stats */}
            <div className="flex justify-center">
              <StatsBar />
            </div>
            <Detector />
          </motion.div>

        </div>

        <footer className="mt-32 text-slate-500 text-sm py-12 flex flex-col md:flex-row items-center justify-between border-t border-white/5 w-full max-w-7xl px-8">
          <div className="flex items-center space-x-2 mb-4 md:mb-0">
             <span className="font-bold text-slate-200">TrueScan.ai</span>
             <span>&copy; 2026</span>
          </div>
          <div className="flex items-center space-x-8">
            <a href="#" className="hover:text-blue-400 transition-colors">Privacy</a>
            <a href="#" className="hover:text-blue-400 transition-colors">Terms</a>
            <a href="#" className="hover:text-blue-400 transition-colors">Enterprise</a>
          </div>
        </footer>
      </main>
    </>
  );
}
