# FactCheck AI — Setup Guide

## Prerequisites
- Python 3.11+
- `ffmpeg` installed (required by yt-dlp for audio extraction)

### Install ffmpeg (macOS)
```bash
brew install ffmpeg
```

---

## 1. Get your API keys

| Service | Purpose | Get it at |
|---------|---------|-----------|
| Anthropic | Claim extraction + fact-checking | https://console.anthropic.com/ |
| OpenAI | Whisper audio transcription | https://platform.openai.com/api-keys |
| Tavily | Live web search for sources | https://app.tavily.com/ |

---

## 2. Install & configure

```bash
cd fact-checker

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
# Edit .env and add your keys
```

---

## 3. Run

```bash
python main.py
```

Open http://localhost:8000 in your browser.

---

## Notes
- Instagram: Only **public** posts work. Private accounts will fail.
- Claims are capped at 6 per video for the POC.
- Each analysis costs roughly $0.05–0.15 in API credits depending on video length.
