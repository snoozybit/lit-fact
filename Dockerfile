# ── Stage 1: Build the Go binary ─────────────────────────────────────────────
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o factcheck .

# ── Stage 2: Lean runtime with ffmpeg + yt-dlp ───────────────────────────────
FROM python:3.11-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    pip install --no-cache-dir yt-dlp && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/factcheck .
COPY static/ ./static/

EXPOSE 8000
CMD ["./factcheck"]
