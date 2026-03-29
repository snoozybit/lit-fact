# FactCheck AI 🔍

> Paste any YouTube link → get every factual claim verified against live sources in seconds.

Built as a community POC to fight misinformation. Open source, MIT licensed, bring your own API keys.

![Status](https://img.shields.io/badge/status-POC-orange) ![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg) ![Go](https://img.shields.io/badge/go-1.22+-00ADD8?logo=go)

---

## How it works

```
YouTube URL
   │
   ▼
yt-dlp → YouTube captions (instant, no download)
   │ fallback
   ▼
yt-dlp download + OpenAI Whisper transcription
   │
   ▼
Claude extracts factual claims from transcript
   │
   ▼
Goroutines: all claims fact-checked in parallel
   │  ├─ Tavily web search
   │  └─ Claude evaluates verdict + sources
   ▼
Overall video verdict + per-claim results
```

### Verdicts
| Verdict | Meaning |
|---------|---------|
| ✅ **Verified** | Accurate and supported by reliable sources |
| ❌ **Incorrect** | Factually wrong with clear contradicting evidence |
| ⚠️ **Misleading** | Has truth but framed deceptively or missing context |
| 🔍 **Unverified** | Insufficient evidence to confirm or deny |

---

## Tech Stack

- **Backend** — Go 1.22 + [Gin](https://github.com/gin-gonic/gin)
- **Transcription** — yt-dlp (YouTube captions) + OpenAI Whisper fallback
- **Claim extraction & fact-checking** — Anthropic Claude (claude-sonnet)
- **Live web search** — Tavily
- **Frontend** — Vanilla HTML/CSS/JS (zero dependencies)
- **Parallel processing** — Go goroutines for concurrent claim verification

---

## Quick Start

### Prerequisites
- [Go 1.22+](https://go.dev/dl/)
- `ffmpeg` — `brew install ffmpeg` (macOS) / `apt install ffmpeg` (Linux)
- `yt-dlp` — `pip install yt-dlp` or `brew install yt-dlp`

### Run locally

```bash
git clone https://github.com/snoozybit/factcheck-ai.git
cd factcheck-ai

# Optional: set server-side default keys
cp .env.example .env

go run .
# → http://localhost:8000
```

Click **API Keys** in the top-right corner to enter your keys in the browser.

### API Keys

| Key | Purpose | Where to get |
|-----|---------|-------------|
| `ANTHROPIC_API_KEY` | Claim extraction + fact-checking | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | Whisper audio transcription | [platform.openai.com](https://platform.openai.com/api-keys) |
| `TAVILY_API_KEY` | Live web search | [app.tavily.com](https://app.tavily.com/) — 1,000 free/month |

Keys entered in the browser are stored in `localStorage` only and sent directly per request.

---

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| YouTube (Shorts + regular) | ✅ Full | Instant via captions API |
| Instagram Reels | ⚠️ Restricted | Requires login cookies |
| TikTok | ⚠️ Restricted | IP-level blocks |
| Twitter / X | ⚠️ Restricted | Auth required for video |

---

## Deploy to Render (free)

1. Fork this repo
2. [render.com](https://render.com) → New → Web Service → connect fork
3. Render auto-reads `render.yaml`
4. Deploy → get your public URL

---

## Project Structure

```
factcheck-ai/
├── main.go          # Go backend — all server logic
├── go.mod / go.sum  # Go modules
├── static/
│   └── index.html   # Single-file frontend
├── Dockerfile       # Multi-stage Go build
├── render.yaml      # Render free hosting config
└── .env.example
```

---

## Contributing

Ideas welcome:
- [ ] Batch processing (multiple URLs)
- [ ] Share results via permalink
- [ ] Chrome extension
- [ ] Support uploaded video files
- [ ] Multilingual support

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.

---

Built by [@snoozybit](https://github.com/snoozybit) · Share it if you find it useful 🙏
