from __future__ import annotations

import os
import re
import json
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import yt_dlp
import anthropic
from openai import OpenAI
from tavily import TavilyClient
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

app = FastAPI(title="FactCheck AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request model ─────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    url: str
    # API keys — supplied by the user via the UI.
    # Falls back to server .env values if not provided (useful for local dev).
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None


# ── Client factory ────────────────────────────────────────────────────────────

def make_clients(req: AnalyzeRequest):
    """Build API clients, preferring keys from the request over .env."""
    anthropic_key = req.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
    openai_key    = req.openai_api_key    or os.getenv("OPENAI_API_KEY",    "")
    tavily_key    = req.tavily_api_key    or os.getenv("TAVILY_API_KEY",    "")

    missing = []
    if not anthropic_key: missing.append("Anthropic")
    if not openai_key:    missing.append("OpenAI")
    if not tavily_key:    missing.append("Tavily")
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing API key(s): {', '.join(missing)}. Please enter them in the settings panel."
        )

    return (
        anthropic.Anthropic(api_key=anthropic_key),
        OpenAI(api_key=openai_key),
        TavilyClient(api_key=tavily_key),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("static/index.html")


# ── Transcript helpers ────────────────────────────────────────────────────────

def extract_youtube_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def get_youtube_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    except Exception:
        fetched = api.fetch(video_id)
    return " ".join(s.text for s in fetched)


def download_audio(url: str, output_path: str) -> str:
    ydl_opts = {
        "format": "bestaudio/best[ext=mp4]/best",
        "outtmpl": output_path + "/%(id)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
    for f in Path(output_path).glob("*.mp3"):
        return str(f)
    raise FileNotFoundError("Audio file not found after download")


def transcribe_audio(audio_path: str, oai: OpenAI) -> str:
    with open(audio_path, "rb") as f:
        return oai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )


def get_transcript(url: str, oai: OpenAI) -> tuple[str, str]:
    """
    Smart transcript fetcher:
      - YouTube → YouTube Transcript API (instant, no download)
      - Others  → yt-dlp download + Whisper
    Returns (transcript_text, method_used)
    """
    yt_id = extract_youtube_id(url)
    if yt_id:
        try:
            return get_youtube_transcript(yt_id), "youtube_captions"
        except (TranscriptsDisabled, NoTranscriptFound):
            pass
        except Exception:
            pass

    # Non-YouTube or YouTube without captions → download + Whisper
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            audio_path = download_audio(url, tmpdir)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Could not download video. Make sure it's a public post. Error: {str(e)}"
            )
        try:
            transcript = transcribe_audio(audio_path, oai)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    return transcript, "whisper"


# ── AI helpers ────────────────────────────────────────────────────────────────

def strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def extract_claims(transcript: str, claude: anthropic.Anthropic) -> list[dict]:
    prompt = f"""You are a fact-checking assistant. Given the following transcript from a video, \
extract all distinct factual claims that can be verified.

Focus on:
- Specific statistics or numbers
- Historical facts
- Scientific claims
- Claims about current events or people
- Health/medical claims
- Economic or financial claims

Ignore opinions, predictions, and obvious generalizations.

Return a JSON array of objects, each with:
- "claim": the exact claim (1-2 sentences)
- "context": brief context of why this claim matters

Transcript:
{transcript}

Return ONLY valid JSON, no other text."""

    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(strip_json_fences(response.content[0].text))


def compute_overall_verdict(transcript: str, claims: list[dict], claude: anthropic.Anthropic) -> dict:
    """
    Given all checked claims, ask Claude for:
    - An overall verdict on the video's central narrative
    - Whether each claim is 'core' (central to the story) or 'general' (background fact)
    """
    claims_summary = "\n".join([
        f"- [{c['verdict'].upper()}] {c['claim']}"
        for c in claims
    ])

    prompt = f"""You are a senior fact-checker reviewing a completed fact-check of a video.

TRANSCRIPT SUMMARY:
{transcript[:800]}

CLAIM VERDICTS:
{claims_summary}

Your job:
1. Give an OVERALL verdict on the central narrative/story of this video.
2. Classify each claim as either "core" (directly supports or is central to the main story/argument)
   or "general" (a background fact, unrelated tangent, or general statement not core to the video's main claim).

Respond with a JSON object:
{{
  "overall_verdict": "<true | false | misleading | unverified>",
  "overall_summary": "<2-3 sentence plain-English verdict on the video's central story>",
  "claim_types": {{
    "<claim text, first 80 chars>": "<core | general>"
  }}
}}

Overall verdict definitions:
- true:       The central story/claim of this video is accurate
- false:      The central story/claim is factually wrong
- misleading: The story has truth but is framed in a deceptive or incomplete way
- unverified: Not enough evidence to make a definitive call

Return ONLY valid JSON."""

    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(strip_json_fences(response.content[0].text))


def fact_check_claim(claim: str, context: str, claude: anthropic.Anthropic, tavily: TavilyClient) -> dict:
    search_results = tavily.search(
        query=f"fact check: {claim}",
        search_depth="advanced",
        max_results=5,
        include_answer=True,
    )
    sources = search_results.get("results", [])
    search_summary = search_results.get("answer", "")
    sources_text = "\n".join([
        f"- [{s.get('title', 'No title')}]({s.get('url', '')}) — {s.get('content', '')[:300]}"
        for s in sources
    ])

    prompt = f"""You are a professional fact-checker. Evaluate the following claim using the \
provided search results.

CLAIM: {claim}
CONTEXT: {context}

SEARCH RESULTS:
{sources_text}

SEARCH SUMMARY: {search_summary}

Respond with a JSON object:
{{
  "verdict": "<one of: verified | unverified | misleading | incorrect>",
  "explanation": "<2-3 sentence explanation of the verdict, citing evidence>",
  "confidence": "<high | medium | low>",
  "sources": [
    {{"title": "...", "url": "...", "relevance": "..."}}
  ]
}}

Verdict definitions:
- verified:    Claim is accurate and supported by reliable sources
- incorrect:   Claim is factually wrong with clear contradicting evidence
- misleading:  Claim has some truth but omits important context or is framed deceptively
- unverified:  Insufficient evidence to confirm or deny the claim

Return ONLY valid JSON."""

    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    result = json.loads(strip_json_fences(response.content[0].text))

    # Supplement sources if Claude returned fewer than 2
    if len(result.get("sources", [])) < 2 and sources:
        result["sources"] = [
            {"title": s.get("title", "Source"), "url": s.get("url", ""), "relevance": "Supporting evidence"}
            for s in sources[:3]
        ]
    return result


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    # Build per-request clients from user-supplied keys
    claude, oai, tavily = make_clients(request)

    # Step 1 & 2: Transcript
    transcript, method = get_transcript(url, oai)

    if not transcript or len(transcript.strip()) < 20:
        raise HTTPException(
            status_code=422,
            detail="Video has little or no speech to analyze. Try a video with spoken content."
        )

    # Step 3: Extract claims
    try:
        claims = extract_claims(transcript, claude)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claim extraction failed: {str(e)}")

    if not claims:
        return {
            "transcript": transcript,
            "transcript_method": method,
            "claims": [],
            "message": "No verifiable factual claims were found in this video."
        }

    # Step 4: Fact-check each claim (cap at 6 for POC)
    results = []
    for item in claims[:6]:
        try:
            verdict = fact_check_claim(item["claim"], item.get("context", ""), claude, tavily)
            results.append({
                "claim": item["claim"],
                "context": item.get("context", ""),
                **verdict
            })
        except Exception as e:
            results.append({
                "claim": item["claim"],
                "context": item.get("context", ""),
                "verdict": "unverified",
                "explanation": f"Could not verify this claim: {str(e)}",
                "confidence": "low",
                "sources": []
            })

    # Step 5: Overall verdict on the video's central narrative
    overall = {"overall_verdict": "unverified", "overall_summary": "", "claim_types": {}}
    try:
        overall = compute_overall_verdict(transcript, results, claude)
    except Exception:
        pass

    # Tag each claim as core or general
    for r in results:
        key = r["claim"][:80]
        r["claim_type"] = overall.get("claim_types", {}).get(key, "core")

    return {
        "transcript": transcript,
        "transcript_method": method,
        "overall_verdict": overall.get("overall_verdict", "unverified"),
        "overall_summary": overall.get("overall_summary", ""),
        "claims": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
