package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

// ── Models ────────────────────────────────────────────────────────────────────

type AnalyzeRequest struct {
	URL             string `json:"url"`
	AnthropicAPIKey string `json:"anthropic_api_key"`
	OpenAIAPIKey    string `json:"openai_api_key"`
	TavilyAPIKey    string `json:"tavily_api_key"`
}

type Claim struct {
	Claim   string `json:"claim"`
	Context string `json:"context"`
}

type Source struct {
	Title     string `json:"title"`
	URL       string `json:"url"`
	Relevance string `json:"relevance"`
}

type CheckedClaim struct {
	Claim       string   `json:"claim"`
	Context     string   `json:"context"`
	Verdict     string   `json:"verdict"`
	Explanation string   `json:"explanation"`
	Confidence  string   `json:"confidence"`
	Sources     []Source `json:"sources"`
	ClaimType   string   `json:"claim_type"`
}

type OverallResult struct {
	OverallVerdict string            `json:"overall_verdict"`
	OverallSummary string            `json:"overall_summary"`
	ClaimTypes     map[string]string `json:"claim_types"`
}

type AnalyzeResponse struct {
	Transcript       string         `json:"transcript"`
	TranscriptMethod string         `json:"transcript_method"`
	OverallVerdict   string         `json:"overall_verdict"`
	OverallSummary   string         `json:"overall_summary"`
	Claims           []CheckedClaim `json:"claims"`
}

// ── Claude API ────────────────────────────────────────────────────────────────

type claudeMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type claudeReq struct {
	Model     string      `json:"model"`
	MaxTokens int         `json:"max_tokens"`
	Messages  []claudeMsg `json:"messages"`
}

type claudeResp struct {
	Content []struct {
		Text string `json:"text"`
	} `json:"content"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func callClaude(ctx context.Context, apiKey, prompt string, maxTokens int) (string, error) {
	body, _ := json.Marshal(claudeReq{
		Model:     "claude-sonnet-4-5",
		MaxTokens: maxTokens,
		Messages:  []claudeMsg{{Role: "user", Content: prompt}},
	})

	req, _ := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("claude http: %w", err)
	}
	defer resp.Body.Close()

	var cr claudeResp
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return "", fmt.Errorf("claude decode: %w", err)
	}
	if cr.Error != nil {
		return "", fmt.Errorf("claude api: %s", cr.Error.Message)
	}
	if len(cr.Content) == 0 {
		return "", fmt.Errorf("claude: empty response")
	}
	return cr.Content[0].Text, nil
}

// ── Tavily API ────────────────────────────────────────────────────────────────

type tavilyReq struct {
	APIKey        string `json:"api_key"`
	Query         string `json:"query"`
	SearchDepth   string `json:"search_depth"`
	MaxResults    int    `json:"max_results"`
	IncludeAnswer bool   `json:"include_answer"`
}

type tavilyResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Content string `json:"content"`
}

type tavilyResp struct {
	Answer  string         `json:"answer"`
	Results []tavilyResult `json:"results"`
}

func searchTavily(ctx context.Context, apiKey, query string) (string, []tavilyResult, error) {
	body, _ := json.Marshal(tavilyReq{
		APIKey:        apiKey,
		Query:         query,
		SearchDepth:   "advanced",
		MaxResults:    5,
		IncludeAnswer: true,
	})

	req, _ := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://api.tavily.com/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", nil, fmt.Errorf("tavily http: %w", err)
	}
	defer resp.Body.Close()

	var tr tavilyResp
	if err := json.NewDecoder(resp.Body).Decode(&tr); err != nil {
		return "", nil, fmt.Errorf("tavily decode: %w", err)
	}
	return tr.Answer, tr.Results, nil
}

// ── OpenAI Whisper ────────────────────────────────────────────────────────────

func transcribeWhisper(ctx context.Context, apiKey, audioPath string) (string, error) {
	f, err := os.Open(audioPath)
	if err != nil {
		return "", fmt.Errorf("open audio: %w", err)
	}
	defer f.Close()

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	fw, _ := w.CreateFormFile("file", filepath.Base(audioPath))
	if _, err := io.Copy(fw, f); err != nil {
		return "", fmt.Errorf("copy audio: %w", err)
	}
	_ = w.WriteField("model", "whisper-1")
	_ = w.WriteField("response_format", "text")
	w.Close()

	req, _ := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://api.openai.com/v1/audio/transcriptions", &buf)
	req.Header.Set("Content-Type", w.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("whisper http: %w", err)
	}
	defer resp.Body.Close()

	text, _ := io.ReadAll(resp.Body)
	return strings.TrimSpace(string(text)), nil
}

// ── Transcript helpers ────────────────────────────────────────────────────────

var ytIDRe = regexp.MustCompile(`(?:youtube\.com/(?:shorts/|watch\?v=|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})`)

func extractYouTubeID(rawURL string) string {
	if m := ytIDRe.FindStringSubmatch(rawURL); len(m) > 1 {
		return m[1]
	}
	return ""
}

func getYouTubeCaptions(videoID string) (string, error) {
	tmpDir, err := os.MkdirTemp("", "subs-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tmpDir)

	cmd := exec.Command(ytDLP(),
		"--write-auto-sub",
		"--sub-lang", "en",
		"--sub-format", "vtt",
		"--skip-download",
		"-o", filepath.Join(tmpDir, "%(id)s"),
		"--quiet",
		"https://www.youtube.com/watch?v="+videoID,
	)
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("yt-dlp subs: %w", err)
	}

	var vttPath string
	_ = filepath.Walk(tmpDir, func(p string, _ os.FileInfo, _ error) error {
		if strings.HasSuffix(p, ".vtt") {
			vttPath = p
		}
		return nil
	})
	if vttPath == "" {
		return "", fmt.Errorf("no vtt file found")
	}

	raw, err := os.ReadFile(vttPath)
	if err != nil {
		return "", err
	}
	return parseVTT(string(raw)), nil
}

var (
	tsRe  = regexp.MustCompile(`^\d{2}:\d{2}`)
	tagRe = regexp.MustCompile(`<[^>]+>`)
)

func parseVTT(vtt string) string {
	seen := map[string]bool{}
	var parts []string
	for _, line := range strings.Split(vtt, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "WEBVTT") ||
			strings.HasPrefix(line, "NOTE") || tsRe.MatchString(line) ||
			strings.HasPrefix(line, "Kind:") || strings.HasPrefix(line, "Language:") {
			continue
		}
		clean := strings.TrimSpace(tagRe.ReplaceAllString(line, ""))
		if clean != "" && !seen[clean] {
			seen[clean] = true
			parts = append(parts, clean)
		}
	}
	return strings.Join(parts, " ")
}

func downloadAudio(rawURL string) (string, error) {
	tmpDir, err := os.MkdirTemp("", "audio-*")
	if err != nil {
		return "", err
	}

	cmd := exec.Command(ytDLP(),
		"-f", "bestaudio/best[ext=mp4]/best",
		"--extract-audio",
		"--audio-format", "mp3",
		"--audio-quality", "128K",
		"-o", filepath.Join(tmpDir, "%(id)s.%(ext)s"),
		"--quiet",
		rawURL,
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		os.RemoveAll(tmpDir)
		return "", fmt.Errorf("yt-dlp: %s", strings.TrimSpace(string(out)))
	}

	var mp3Path string
	_ = filepath.Walk(tmpDir, func(p string, _ os.FileInfo, _ error) error {
		if strings.HasSuffix(p, ".mp3") {
			mp3Path = p
		}
		return nil
	})
	if mp3Path == "" {
		os.RemoveAll(tmpDir)
		return "", fmt.Errorf("no mp3 found after download")
	}
	return mp3Path, nil
}

// getTranscript tries YouTube captions first, falls back to yt-dlp + Whisper.
func getTranscript(ctx context.Context, rawURL, openAIKey string) (transcript, method string, err error) {
	if ytID := extractYouTubeID(rawURL); ytID != "" {
		if t, e := getYouTubeCaptions(ytID); e == nil && len(strings.TrimSpace(t)) > 20 {
			return t, "youtube_captions", nil
		}
	}

	audioPath, err := downloadAudio(rawURL)
	if err != nil {
		return "", "", fmt.Errorf("could not download video — make sure it's public: %w", err)
	}
	defer os.RemoveAll(filepath.Dir(audioPath))

	text, err := transcribeWhisper(ctx, openAIKey, audioPath)
	if err != nil {
		return "", "", fmt.Errorf("transcription failed: %w", err)
	}
	return text, "whisper", nil
}

// ── AI helpers ────────────────────────────────────────────────────────────────

func stripFences(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		parts := strings.SplitN(s, "```", 3)
		if len(parts) >= 2 {
			s = parts[1]
			s = strings.TrimPrefix(s, "json")
		}
	}
	return strings.TrimSpace(s)
}

func extractClaims(ctx context.Context, transcript, claudeKey string) ([]Claim, error) {
	prompt := fmt.Sprintf(`You are a fact-checking assistant. Extract all distinct verifiable factual claims from this transcript.

Focus on: statistics/numbers, historical facts, scientific claims, current events, health claims, economic claims.
Ignore opinions, predictions, and obvious generalizations.

Return a JSON array of objects with "claim" and "context" fields only.

Transcript:
%s

Return ONLY valid JSON, no other text.`, transcript)

	raw, err := callClaude(ctx, claudeKey, prompt, 2000)
	if err != nil {
		return nil, err
	}
	var claims []Claim
	if err := json.Unmarshal([]byte(stripFences(raw)), &claims); err != nil {
		return nil, fmt.Errorf("parse claims JSON: %w", err)
	}
	return claims, nil
}

func factCheckClaim(ctx context.Context, cl Claim, claudeKey, tavilyKey string) CheckedClaim {
	result := CheckedClaim{
		Claim:      cl.Claim,
		Context:    cl.Context,
		Verdict:    "unverified",
		Confidence: "low",
		Sources:    []Source{},
	}

	answer, results, err := searchTavily(ctx, tavilyKey, "fact check: "+cl.Claim)
	if err != nil {
		result.Explanation = "Web search failed: " + err.Error()
		return result
	}

	var sb strings.Builder
	for _, r := range results {
		content := r.Content
		if len(content) > 300 {
			content = content[:300]
		}
		fmt.Fprintf(&sb, "- [%s](%s) — %s\n", r.Title, r.URL, content)
	}

	prompt := fmt.Sprintf(`You are a professional fact-checker. Evaluate this claim using the provided search results.

CLAIM: %s
CONTEXT: %s

SEARCH RESULTS:
%s

SEARCH SUMMARY: %s

Respond with a JSON object:
{
  "verdict": "<verified|unverified|misleading|incorrect>",
  "explanation": "<2-3 sentence explanation citing evidence>",
  "confidence": "<high|medium|low>",
  "sources": [{"title":"...","url":"...","relevance":"..."}]
}

Verdict definitions:
- verified:   Accurate and supported by reliable sources
- incorrect:  Factually wrong with clear contradicting evidence
- misleading: Some truth but framed deceptively or missing context
- unverified: Insufficient evidence to confirm or deny

Return ONLY valid JSON.`, cl.Claim, cl.Context, sb.String(), answer)

	raw, err := callClaude(ctx, claudeKey, prompt, 1000)
	if err != nil {
		result.Explanation = "Fact-check failed: " + err.Error()
		return result
	}

	var checked CheckedClaim
	if err := json.Unmarshal([]byte(stripFences(raw)), &checked); err != nil {
		result.Explanation = "Could not parse fact-check result"
		return result
	}
	checked.Claim = cl.Claim
	checked.Context = cl.Context

	// Supplement sources from Tavily if Claude returned too few
	if len(checked.Sources) < 2 && len(results) > 0 {
		checked.Sources = []Source{}
		for i, r := range results {
			if i >= 3 {
				break
			}
			checked.Sources = append(checked.Sources, Source{
				Title:     r.Title,
				URL:       r.URL,
				Relevance: "Supporting evidence",
			})
		}
	}
	return checked
}

func computeOverall(ctx context.Context, transcript string, claims []CheckedClaim, claudeKey string) OverallResult {
	var sb strings.Builder
	for _, c := range claims {
		fmt.Fprintf(&sb, "- [%s] %s\n", strings.ToUpper(c.Verdict), c.Claim)
	}
	summary := transcript
	if len(summary) > 800 {
		summary = summary[:800]
	}

	prompt := fmt.Sprintf(`You are a senior fact-checker reviewing a completed fact-check of a video.

TRANSCRIPT:
%s

CLAIM VERDICTS:
%s

Return a JSON object:
{
  "overall_verdict": "<true|false|misleading|unverified>",
  "overall_summary": "<2-3 sentence plain-English verdict on the video's central story>",
  "claim_types": {"<first 80 chars of claim text>": "<core|general>"}
}

Verdict definitions:
- true:       Central story is accurate
- false:      Central story is factually wrong
- misleading: Has truth but framed deceptively
- unverified: Not enough evidence to decide

Return ONLY valid JSON.`, summary, sb.String())

	raw, err := callClaude(ctx, claudeKey, prompt, 1000)
	if err != nil {
		return OverallResult{OverallVerdict: "unverified"}
	}
	var result OverallResult
	if err := json.Unmarshal([]byte(stripFences(raw)), &result); err != nil {
		return OverallResult{OverallVerdict: "unverified"}
	}
	return result
}

// ── HTTP handler ──────────────────────────────────────────────────────────────

func analyzeHandler(c *gin.Context) {
	var req AnalyzeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid request body"})
		return
	}

	anthropicKey := firstNonEmpty(req.AnthropicAPIKey, os.Getenv("ANTHROPIC_API_KEY"))
	openaiKey := firstNonEmpty(req.OpenAIAPIKey, os.Getenv("OPENAI_API_KEY"))
	tavilyKey := firstNonEmpty(req.TavilyAPIKey, os.Getenv("TAVILY_API_KEY"))

	var missing []string
	if anthropicKey == "" {
		missing = append(missing, "Anthropic")
	}
	if openaiKey == "" {
		missing = append(missing, "OpenAI")
	}
	if tavilyKey == "" {
		missing = append(missing, "Tavily")
	}
	if len(missing) > 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"detail": fmt.Sprintf("Missing API key(s): %s. Please enter them via the API Keys button.",
				strings.Join(missing, ", ")),
		})
		return
	}

	if strings.TrimSpace(req.URL) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "URL is required"})
		return
	}

	ctx := c.Request.Context()

	// 1. Transcript
	transcript, method, err := getTranscript(ctx, req.URL, openaiKey)
	if err != nil {
		c.JSON(http.StatusUnprocessableEntity, gin.H{"detail": err.Error()})
		return
	}
	if len(strings.TrimSpace(transcript)) < 20 {
		c.JSON(http.StatusUnprocessableEntity, gin.H{
			"detail": "Video has little or no speech to analyze. Try a video with spoken content.",
		})
		return
	}

	// 2. Extract claims
	claims, err := extractClaims(ctx, transcript, anthropicKey)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "Claim extraction failed: " + err.Error()})
		return
	}
	if len(claims) == 0 {
		c.JSON(http.StatusOK, AnalyzeResponse{
			Transcript:       transcript,
			TranscriptMethod: method,
			OverallVerdict:   "unverified",
			OverallSummary:   "No verifiable factual claims were found in this video.",
			Claims:           []CheckedClaim{},
		})
		return
	}
	if len(claims) > 6 {
		claims = claims[:6]
	}

	// 3. Fact-check all claims in parallel — Go's goroutines shine here
	checkedClaims := make([]CheckedClaim, len(claims))
	var wg sync.WaitGroup
	for i, cl := range claims {
		wg.Add(1)
		go func(idx int, claim Claim) {
			defer wg.Done()
			checkedClaims[idx] = factCheckClaim(ctx, claim, anthropicKey, tavilyKey)
		}(i, cl)
	}
	wg.Wait()

	// 4. Overall verdict
	overall := computeOverall(ctx, transcript, checkedClaims, anthropicKey)

	// Tag claim types
	for i, cc := range checkedClaims {
		key := cc.Claim
		if len(key) > 80 {
			key = key[:80]
		}
		if t, ok := overall.ClaimTypes[key]; ok {
			checkedClaims[i].ClaimType = t
		} else {
			checkedClaims[i].ClaimType = "core"
		}
	}

	c.JSON(http.StatusOK, AnalyzeResponse{
		Transcript:       transcript,
		TranscriptMethod: method,
		OverallVerdict:   overall.OverallVerdict,
		OverallSummary:   overall.OverallSummary,
		Claims:           checkedClaims,
	})
}

// ── Utilities ─────────────────────────────────────────────────────────────────

// ytDLP resolves the yt-dlp binary — checks common install locations.
func ytDLP() string {
	for _, p := range []string{
		"/opt/homebrew/bin/yt-dlp", // macOS Homebrew (Apple Silicon)
		"/usr/local/bin/yt-dlp",    // macOS Homebrew (Intel) / Linux
		"/usr/bin/yt-dlp",          // Linux apt/snap
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return "yt-dlp" // fall back to PATH
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v = strings.TrimSpace(v); v != "" {
			return v
		}
	}
	return ""
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	_ = godotenv.Load()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())

	// CORS
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type")
		if c.Request.Method == http.MethodOptions {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	})

	r.Static("/static", "./static")
	r.GET("/", func(c *gin.Context) { c.File("./static/index.html") })
	r.POST("/analyze", analyzeHandler)

	log.Printf("🔍 FactCheck AI → http://localhost:%s", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatal(err)
	}
}
