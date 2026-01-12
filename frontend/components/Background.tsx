"use client";

import { motion } from "framer-motion";
import { useEffect, useState } from "react";

export default function Background() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      // Smooth out the values slightly by dividing by window dimensions
      setMousePosition({ 
        x: e.clientX / window.innerWidth, 
        y: e.clientY / window.innerHeight 
      });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  return (
    <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none bg-[#020617]">
      <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center opacity-[0.15] [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))]" />
      
      {/* Primary Orb - Blue */}
      <motion.div 
        animate={{
          x: mousePosition.x * 50,
          y: mousePosition.y * 50,
        }}
        transition={{ type: "spring", damping: 50, stiffness: 50 }}
        className="absolute -top-[20%] -left-[10%] w-[70vw] h-[70vw] bg-blue-600/20 rounded-full blur-[120px] mix-blend-screen animate-pulse-soft"
      />
      
      {/* Secondary Orb - Purple */}
      <motion.div 
        animate={{
          x: mousePosition.x * -50,
          y: mousePosition.y * -50,
        }}
        transition={{ type: "spring", damping: 50, stiffness: 50 }}
        className="absolute top-[20%] -right-[10%] w-[60vw] h-[60vw] bg-purple-600/15 rounded-full blur-[120px] mix-blend-screen animate-pulse-soft"
        style={{ animationDelay: "1s" }}
      />

       {/* Accent Orb - Indigo */}
       <motion.div 
        animate={{
          x: mousePosition.x * 20,
          y: mousePosition.y * 20,
        }}
        transition={{ type: "spring", damping: 100, stiffness: 20 }}
        className="absolute bottom-[-10%] left-[20%] w-[50vw] h-[50vw] bg-indigo-500/10 rounded-full blur-[100px] mix-blend-screen"
      />
    </div>
  );
}
