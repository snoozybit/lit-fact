# FactCheck AI 🔍

**Paste any Instagram, YouTube, or TikTok link → get every factual claim verified against live sources in seconds.**

Built as a community POC to fight misinformation on social media. Open source, MIT licensed, bring your own API keys.

![FactCheck AI UI](https://img.shields.io/badge/status-POC-orange) ![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.9+-brightgreen)

---

## How it works

```
Video URL
   │
   ▼
[YouTube?] ──yes──▶ YouTube Transcript API (instant, no download)
   │ no
   ▼
yt-dlp download + OpenAI Whisper transcription
   │
   ▼
Claude extracts factual claims from transcript
   │
   ▼
For each claim: Tavily web search → Claude evaluates verdict
   │
   ▼
Overall video verdict + per-claim results with sources
```

### Verdicts
| Verdict | Meaning |
|---------|---------|
| ✅ **Verified** | Claim is accurate and supported by reliable sources |
| ❌ **Incorrect** | Claim is factually wrong with clear contradicting evidence |
| ⚠️ **Misleading** | Has some truth but omits context or is framed deceptively |
| 🔍 **Unverified** | Insufficient evidence to confirm or deny |

Claims are also categorised as **Key Claims** (central to the video's narrative) or **General Facts** (background statements).

---

## Quick start

### Prerequisites
- Python 3.9+
- `ffmpeg` — `brew install ffmpeg` (macOS) / `apt install ffmpeg` (Linux)

### 1. Clone & install
```bash
git clone https://github.com/lalitnankani/factcheck-ai.git
cd factcheck-ai

python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API keys (optional for local dev)
You can enter keys directly in the UI on every visit — they're saved to your browser's localStorage and never sent to any server except the respective API provider.

For running a shared server, create a `.env`:
```bash
cp .env.example .env
# Fill in your keys
```

| Key | Where to get it | Cost |
|-----|----------------|------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) | ~$0.05–0.10/video |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) | $0.006/min of audio |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com/) | 1,000 free/month |

### 3. Run
```bash
python main.py
```
Open **http://localhost:8000**

---

## Supported platforms
- ✅ YouTube (Shorts, regular videos, embeds)
- ✅ Instagram Reels & posts (public only)
- ✅ TikTok
- ✅ Twitter / X
- ✅ Any platform supported by [yt-dlp](https://github.com/yt-dlp/yt-dlp)

---

## Project structure
```
factcheck-ai/
├── main.py              # FastAPI backend
├── static/
│   └── index.html       # Single-file frontend
├── requirements.txt
├── .env.example
└── README.md
```

---

## Extending this

Ideas for contributions:
- [ ] Batch processing (multiple URLs at once)
- [ ] Share results via link
- [ ] Chrome extension
- [ ] Support for uploaded video files
- [ ] Add more claim categories (legal, financial, etc.)
- [ ] Multilingual support
- [ ] Export results as PDF

---

## Contributing
PRs welcome! Open an issue first for large changes.

---

## License
[MIT](LICENSE) — free to use, modify, and distribute.

---

Built by [@lalitnankani](https://linkedin.com/in/lalitnankani) · Share it if you find it useful 🙏
