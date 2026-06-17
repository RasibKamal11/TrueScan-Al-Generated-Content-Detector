/**
 * TrueScan Browser Extension — Background Service Worker
 * ========================================================
 * Manifest V3 service worker. Handles:
 *   - Context menu creation (right-click "Scan with TrueScan")
 *   - Message routing between content.js ↔ popup.js
 *   - API calls (avoids CORS issues from content scripts)
 */

const DEFAULT_API_BASE = "http://localhost:8000";

// ── Context menu ──────────────────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id:       "truescan-scan-selection",
    title:    "🔍 Scan with TrueScan",
    contexts: ["selection"],
  });
  console.log("[TrueScan] Extension installed. Context menu registered.");
});


// ── Context menu click ────────────────────────────────────────────────────────

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "truescan-scan-selection") return;

  const selectedText = info.selectionText?.trim();
  if (!selectedText || selectedText.length < 10) return;

  // Show loading overlay in content script
  chrome.tabs.sendMessage(tab.id, { action: "showOverlay", text: selectedText });

  const result = await scanText(selectedText);

  // Send result to content script overlay
  chrome.tabs.sendMessage(tab.id, { action: "showResult", result, text: selectedText });
});


// ── Message handler (from popup.js and content.js) ────────────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "scan") {
    scanText(message.text).then(result => sendResponse({ result }));
    return true; // keep channel open for async
  }

  if (message.action === "getSettings") {
    chrome.storage.sync.get(["apiBase", "authToken"], data => {
      sendResponse({
        apiBase:   data.apiBase   || DEFAULT_API_BASE,
        authToken: data.authToken || null,
      });
    });
    return true;
  }
});


// ── API call ──────────────────────────────────────────────────────────────────

async function scanText(text) {
  const { apiBase, authToken } = await chrome.storage.sync.get(["apiBase", "authToken"]);
  const base  = apiBase || DEFAULT_API_BASE;

  const headers = { "Content-Type": "application/json" };
  if (authToken) headers["Authorization"] = `Bearer ${authToken}`;

  try {
    const resp = await fetch(`${base}/detect/text`, {
      method:  "POST",
      headers,
      body:    JSON.stringify({ text, detailed: true }),
    });

    if (!resp.ok) throw new Error(`API error ${resp.status}`);
    return await resp.json();
  } catch (err) {
    console.error("[TrueScan] API error:", err);
    return { error: err.message, ai_probability: null };
  }
}
