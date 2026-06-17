/**
 * TrueScan Browser Extension — Popup Script
 * ==========================================
 */

const inputText  = document.getElementById("inputText");
const scanBtn    = document.getElementById("scanBtn");
const spinner    = document.getElementById("spinner");
const btnLabel   = document.getElementById("btnLabel");
const resultArea = document.getElementById("resultArea");
const statusDot  = document.getElementById("statusDot");

// ── Check API health on open ───────────────────────────────────────────────────
(async () => {
  const { apiBase } = await getSettings();
  try {
    const resp = await fetch(`${apiBase}/health`, { signal: AbortSignal.timeout(3000) });
    if (resp.ok) {
      statusDot.style.background = "#34d399";
      statusDot.title = "API Connected";
    } else {
      throw new Error("unhealthy");
    }
  } catch {
    statusDot.style.background = "#f87171";
    statusDot.style.boxShadow  = "0 0 6px #f87171";
    statusDot.title = "API Offline — is the backend running?";
  }
})();

// ── Pre-fill with selected text if available ──────────────────────────────────
chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
  chrome.scripting?.executeScript({
    target: { tabId: tab.id },
    func:   () => window.getSelection().toString(),
  }).then(results => {
    const sel = results?.[0]?.result?.trim();
    if (sel && sel.length > 5) inputText.value = sel;
  }).catch(() => {});
});

// ── Scan button ───────────────────────────────────────────────────────────────
scanBtn.addEventListener("click", async () => {
  const text = inputText.value.trim();
  if (!text || text.length < 10) {
    showError("Please enter at least 10 characters.");
    return;
  }

  setLoading(true);
  resultArea.classList.remove("visible");

  const response = await chrome.runtime.sendMessage({ action: "scan", text });
  const result   = response?.result;

  setLoading(false);

  if (!result || result.error) {
    showError(result?.error || "Unknown error. Is the backend running?");
    return;
  }

  showResult(result, text);
});

// ── Settings link ─────────────────────────────────────────────────────────────
document.getElementById("settingsLink").addEventListener("click", (e) => {
  e.preventDefault();
  const apiBase = prompt("API Base URL:", localStorage.getItem("apiBase") || "http://localhost:8000");
  if (apiBase) {
    chrome.storage.sync.set({ apiBase });
    localStorage.setItem("apiBase", apiBase);
  }
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function setLoading(loading) {
  scanBtn.disabled    = loading;
  spinner.style.display = loading ? "block" : "none";
  btnLabel.textContent  = loading ? "Scanning…" : "Scan Text";
  document.getElementById("btnIcon").style.display = loading ? "none" : "inline";
}

function showResult(result, text) {
  const prob  = result.ai_probability ?? 0;
  const pct   = Math.round(prob * 100);
  const cls   = prob > 0.65 ? "ai" : prob > 0.4 ? "mixed" : "human";
  const label = prob > 0.65 ? "Likely AI-Generated" : prob > 0.4 ? "Mixed / Uncertain" : "Likely Human";
  const barColor = prob > 0.65 ? "#f87171" : prob > 0.4 ? "#fbbf24" : "#34d399";

  const perp  = result.perplexity  ? `${result.perplexity.toFixed(1)}`  : "–";
  const burst = result.burstiness  ? `${result.burstiness.toFixed(2)}`  : "–";
  const voc   = result.vocab_richness != null ? `${Math.round(result.vocab_richness * 100)}%` : "–";
  const len   = text.length > 999 ? `${(text.length / 1000).toFixed(1)}k` : `${text.length}`;

  resultArea.innerHTML = `
    <div class="score-ring">
      <div class="score-number ${cls}">${pct}%</div>
      <div class="score-label">${label}</div>
    </div>
    <div class="progress-bar">
      <div class="progress-fill" style="width:${pct}%; background:${barColor}"></div>
    </div>
    <div class="metrics">
      <div class="metric">
        <div class="metric-name">Perplexity</div>
        <div class="metric-value">${perp}</div>
      </div>
      <div class="metric">
        <div class="metric-name">Burstiness</div>
        <div class="metric-value">${burst}</div>
      </div>
      <div class="metric">
        <div class="metric-name">Vocab Richness</div>
        <div class="metric-value">${voc}</div>
      </div>
      <div class="metric">
        <div class="metric-name">Length</div>
        <div class="metric-value">${len} chars</div>
      </div>
    </div>
  `;
  resultArea.classList.add("visible");
}

function showError(msg) {
  resultArea.innerHTML = `<div class="error-msg">⚠️ ${msg}</div>`;
  resultArea.classList.add("visible");
}

async function getSettings() {
  return new Promise(resolve => {
    chrome.storage.sync.get(["apiBase", "authToken"], data => {
      resolve({
        apiBase:   data.apiBase   || "http://localhost:8000",
        authToken: data.authToken || null,
      });
    });
  });
}
