/**
 * TrueScan Browser Extension — Content Script
 * ==============================================
 * Injected into every page. Responsibilities:
 *   - Listen for messages from background.js to show/hide overlay
 *   - Inject floating result overlay into the DOM
 *   - Auto-scan on text selection (optional, user-configurable)
 */

(function () {
  "use strict";

  // ── Overlay styles ─────────────────────────────────────────────────────────
  const STYLE_ID = "truescan-style";
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      #truescan-overlay {
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 2147483647;
        width: 320px;
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.05);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #e2e8f0;
        backdrop-filter: blur(12px);
        animation: truescan-slide-in 0.3s ease;
      }
      @keyframes truescan-slide-in {
        from { transform: translateY(20px); opacity: 0; }
        to   { transform: translateY(0);    opacity: 1; }
      }
      #truescan-overlay .ts-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
      }
      #truescan-overlay .ts-logo {
        font-size: 14px;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.5px;
      }
      #truescan-overlay .ts-close {
        cursor: pointer;
        font-size: 18px;
        color: #64748b;
        background: none;
        border: none;
        line-height: 1;
        padding: 0;
      }
      #truescan-overlay .ts-close:hover { color: #94a3b8; }
      #truescan-overlay .ts-preview {
        font-size: 12px;
        color: #64748b;
        margin-bottom: 12px;
        line-height: 1.4;
        max-height: 48px;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      #truescan-overlay .ts-loading {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 13px;
        color: #94a3b8;
      }
      #truescan-overlay .ts-spinner {
        width: 18px;
        height: 18px;
        border: 2px solid rgba(99,102,241,0.3);
        border-top-color: #6366f1;
        border-radius: 50%;
        animation: truescan-spin 0.8s linear infinite;
      }
      @keyframes truescan-spin { to { transform: rotate(360deg); } }
      #truescan-overlay .ts-score {
        font-size: 36px;
        font-weight: 800;
        text-align: center;
        margin: 8px 0 4px;
      }
      #truescan-overlay .ts-score.ai    { color: #f87171; }
      #truescan-overlay .ts-score.mixed { color: #fbbf24; }
      #truescan-overlay .ts-score.human { color: #34d399; }
      #truescan-overlay .ts-label {
        text-align: center;
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 12px;
      }
      #truescan-overlay .ts-bar-bg {
        height: 6px;
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 12px;
      }
      #truescan-overlay .ts-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.8s ease;
      }
      #truescan-overlay .ts-cta {
        display: block;
        text-align: center;
        font-size: 12px;
        color: #6366f1;
        text-decoration: none;
        margin-top: 4px;
      }
      #truescan-overlay .ts-cta:hover { color: #8b5cf6; }
      #truescan-overlay .ts-error {
        font-size: 12px;
        color: #f87171;
        text-align: center;
      }
    `;
    document.head.appendChild(style);
  }

  // ── Overlay element ────────────────────────────────────────────────────────
  let _overlay = null;

  function getOverlay() {
    if (!_overlay) {
      _overlay = document.createElement("div");
      _overlay.id = "truescan-overlay";
      document.body.appendChild(_overlay);
    }
    return _overlay;
  }

  function removeOverlay() {
    if (_overlay) {
      _overlay.remove();
      _overlay = null;
    }
  }

  function showLoading(text) {
    const el = getOverlay();
    el.innerHTML = `
      <div class="ts-header">
        <span class="ts-logo">⚡ TrueScan</span>
        <button class="ts-close" id="ts-close-btn">✕</button>
      </div>
      <div class="ts-preview">${text.slice(0, 120)}${text.length > 120 ? "…" : ""}</div>
      <div class="ts-loading">
        <div class="ts-spinner"></div>
        Analysing with AI…
      </div>
    `;
    document.getElementById("ts-close-btn")?.addEventListener("click", removeOverlay);
  }

  function showResult(result, text) {
    const el = getOverlay();

    if (result.error) {
      el.innerHTML = `
        <div class="ts-header">
          <span class="ts-logo">⚡ TrueScan</span>
          <button class="ts-close" id="ts-close-btn">✕</button>
        </div>
        <div class="ts-error">⚠️ ${result.error}</div>
      `;
      document.getElementById("ts-close-btn")?.addEventListener("click", removeOverlay);
      return;
    }

    const prob    = result.ai_probability ?? 0;
    const pct     = Math.round(prob * 100);
    const cls     = prob > 0.65 ? "ai" : prob > 0.4 ? "mixed" : "human";
    const label   = prob > 0.65 ? "Likely AI-Generated" : prob > 0.4 ? "Mixed / Uncertain" : "Likely Human";
    const barColor = prob > 0.65 ? "#f87171" : prob > 0.4 ? "#fbbf24" : "#34d399";

    el.innerHTML = `
      <div class="ts-header">
        <span class="ts-logo">⚡ TrueScan</span>
        <button class="ts-close" id="ts-close-btn">✕</button>
      </div>
      <div class="ts-preview">${text.slice(0, 120)}${text.length > 120 ? "…" : ""}</div>
      <div class="ts-score ${cls}">${pct}%</div>
      <div class="ts-label">${label}</div>
      <div class="ts-bar-bg">
        <div class="ts-bar-fill" style="width:${pct}%; background:${barColor}"></div>
      </div>
      <a class="ts-cta" href="http://localhost:3000" target="_blank">Open full analysis →</a>
    `;
    document.getElementById("ts-close-btn")?.addEventListener("click", removeOverlay);

    // Auto-dismiss after 12s
    setTimeout(removeOverlay, 12000);
  }

  // ── Message listener ───────────────────────────────────────────────────────
  chrome.runtime.onMessage.addListener((message) => {
    if (message.action === "showOverlay") {
      showLoading(message.text);
    } else if (message.action === "showResult") {
      showResult(message.result, message.text);
    }
  });

})();
