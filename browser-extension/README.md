# TrueScan Browser Extension

A Manifest V3 Chrome/Edge/Brave extension that lets you scan any selected text or web content for AI generation directly from the browser.

## Features

- **Right-click scan**: Select text on any page → right-click → "🔍 Scan with TrueScan"
- **Popup scan**: Click the extension icon and paste/type text
- **Live API health indicator**: Green dot = backend running, red = offline
- **Floating result overlay**: Score appears near your selection with auto-dismiss
- **Configurable API endpoint**: Works with local dev (`localhost:8000`) or production

## Installation (Developer Mode)

> Icons are placeholders — replace `icons/icon*.png` with actual 16/32/48/128px PNGs before publishing.

### Chrome / Edge / Brave

1. Open `chrome://extensions` (or `edge://extensions`)
2. Enable **Developer mode** (toggle top-right)
3. Click **"Load unpacked"**
4. Select the `browser-extension/` directory
5. The TrueScan icon will appear in your toolbar

## Usage

### Right-click Scan
1. Select any text on a webpage
2. Right-click → **"🔍 Scan with TrueScan"**
3. A floating overlay appears with the AI probability score

### Popup Scan
1. Click the TrueScan icon in the toolbar
2. Paste or type text in the textarea
3. Click **"Scan Text"**

### Settings
- Click **"⚙ Settings"** in the popup footer to change the API base URL
- Default: `http://localhost:8000` (local dev)
- Production: `https://yourdomain.com`

## File Structure

```
browser-extension/
├── manifest.json     # Manifest V3 configuration
├── background.js     # Service worker: context menu, API calls, message routing
├── content.js        # Content script: floating result overlay injection
├── popup.html        # Extension popup UI
├── popup.js          # Popup logic: health check, scan, result display
└── icons/
    ├── icon16.png    # ← Replace with actual icons
    ├── icon32.png
    ├── icon48.png
    └── icon128.png
```

## Development

The extension calls `POST /detect/text` on your TrueScan backend.

Make sure the backend is running:
```bash
cd backend && python main.py
```

If your backend uses JWT auth, paste your token in popup Settings → it will be included as `Authorization: Bearer <token>`.

## Packaging for Chrome Web Store

1. Replace `icons/` with real PNG icons
2. Zip the `browser-extension/` directory contents (not the folder itself)
3. Upload to [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole)

```bash
# From project root:
cd browser-extension
zip -r ../truescan-extension.zip . --exclude "*.DS_Store"
```
