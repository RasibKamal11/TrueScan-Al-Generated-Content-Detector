import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "TrueScan AI-Generated Content Detector",
  description: "Verify content authenticity with TrueScan's enterprise-grade AI detection models.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.className} antialiased min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-black selection:bg-blue-500 selection:text-white`}
      >
        <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))]" />
        <main className="relative z-10 flex flex-col items-center justify-center min-h-screen p-4">
            {children}
        </main>
        <div className="fixed bottom-4 right-4 z-50 text-slate-300 text-sm font-semibold opacity-90 hover:opacity-100 transition-opacity pointer-events-none select-none">
          Developed by Rasib, Lisan, Kaif
        </div>
      </body>
    </html>
  );
}
