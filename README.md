# AI Search Middleware

A production-quality AI-powered search middleware that sits between your application and SearXNG, enriching queries with Claude Haiku intent analysis and re-ranking results with a cross-encoder model.

## What This Is

This service acts as an intelligent proxy over a local [SearXNG](https://searxng.github.io/searxng/) instance. Given a plain-text query, it:

1. Classifies the intent and expands the query via Claude Haiku
2. Routes to the right SearXNG engines for that intent (arxiv for research, pubmed for biomedical, github for code, etc.)
3. Fetches, deduplicates, and scores results using domain authority, recency, and engine trust
4. Re-ranks the top candidates with a cross-encoder (semantic relevance) model
5. Returns a clean, ranked JSON response with per-result score breakdowns

It fits into a larger stack where other services (such as OpenClaw) call it via a simple HTTP GET to get authoritative, ranked search results without managing SearXNG or ML models directly.

---

## Prerequisites

- **Python 3.12**
- **Redis** running on `localhost:6379` (default)
- **SearXNG** running on `http://localhost:8888`
- An **Anthropic API key** for Claude Haiku

---

## Installation

```bash
# Clone / enter the project
cd ~/projects/ai-search

# Install dependencies
pip install -r requirements.txt
# or
make install
```

On first start, `sentence-transformers` will download the `cross-encoder/ms-marco-MiniLM-L-6-v2` model (~85 MB) automatically.

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable             | Default                        | Description                        |
|----------------------|--------------------------------|------------------------------------|
| `ANTHROPIC_API_KEY`  | *(required)*                   | Anthropic API key for Claude Haiku |
| `REDIS_URL`          | `redis://localhost:6379/0`     | Redis connection URL               |
| `SEARXNG_URL`        | `http://localhost:8888`        | SearXNG base URL                   |
| `CACHE_TTL_HOURS`    | `24`                           | Result cache lifetime in hours     |
| `MAX_RESULTS`        | `8`                            | Default results per query          |
| `PORT`               | `7777`                         | Port to listen on                  |
| `LOG_LEVEL`          | `INFO`                         | Python log level                   |
| `ENVIRONMENT`        | `development`                  | `development` or `production`      |

In `development` mode, logs are pretty-printed to stdout. In `production`, they are JSON.

---

## Running

### Development (with auto-reload)

```bash
make dev
# or
uvicorn src.ai_search.main:app --reload --host 0.0.0.0 --port 7777
```

### Production

```bash
make start
# or
uvicorn src.ai_search.main:app --host 0.0.0.0 --port 7777 --workers 1
```

### As a systemd Service

```bash
sudo cp systemd/ai-search.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-search
sudo journalctl -u ai-search -f
```

---

## API Reference

### `GET /search`

Run an AI-enhanced search query.

**Parameters**

| Name          | Type    | Required | Default | Description                      |
|---------------|---------|----------|---------|----------------------------------|
| `q`           | string  | yes      | —       | Search query (1–512 chars)       |
| `limit`       | integer | no       | `8`     | Max results to return (1–20)     |
| `domain_hint` | string  | no       | —       | Optional domain bias (reserved)  |

**Example request**

```bash
curl "http://localhost:7777/search?q=transformer+attention+mechanism&limit=5"
```

**Example response**

```json
{
  "query": "transformer attention mechanism",
  "expanded_query": "transformer attention mechanism self-attention neural network 2017 2023",
  "intent": "research",
  "cache_hit": false,
  "query_time_ms": 743.2,
  "results": [
    {
      "title": "Attention Is All You Need",
      "url": "https://arxiv.org/abs/1706.03762",
      "snippet": "The dominant sequence transduction models...",
      "domain": "arxiv.org",
      "source_engine": "arxiv",
      "published_date": "2017-06",
      "final_score": 0.9134,
      "score_breakdown": {
        "semantic": 0.882,
        "authority": 1.0,
        "recency": 0.201,
        "engine_trust": 1.0,
        "position": 0.91
      }
    }
  ]
}
```

---

### `GET /health`

Check service liveness and dependency health.

```bash
curl http://localhost:7777/health
# or
make health
```

**Example response**

```json
{
  "status": "ok",
  "redis": "connected",
  "searxng": "reachable",
  "cross_encoder": "loaded",
  "uptime_seconds": 3600.0
}
```

`status` is `"ok"` when both Redis and SearXNG are reachable, otherwise `"degraded"`.

---

### `GET /stats`

Return cumulative query statistics from Redis.

```bash
curl http://localhost:7777/stats
```

**Example response**

```json
{
  "queries_total": 142,
  "cache_hit_rate": 0.338,
  "avg_latency_ms": 812.4,
  "queries_by_intent": {
    "research": 80,
    "general": 40,
    "biomedical": 22
  }
}
```

---

## Wiring into OpenClaw

Call the search endpoint with `web_fetch`:

```python
web_fetch("http://localhost:7777/search?q=your+query")
```

Or with a limit:

```python
web_fetch("http://localhost:7777/search?q=BERT+pretraining&limit=10")
```

The response JSON is self-contained — parse `results[*].url` and `results[*].snippet` for the content you need, or use `results[*].final_score` to prioritise which URLs to fetch in full.

---

## Architecture Overview

The pipeline runs sequentially across five stages for each non-cached query:

```
User query
    │
    ▼
[Stage 0] Cache lookup (SHA256 key → Redis)
    │  hit: return immediately
    │  miss: continue
    ▼
[Stage 1] Claude Haiku — intent classification & query expansion
    │  intent: research | biomedical | code | news | general
    │  expanded_query: synonym-enriched version
    ▼
[Stage 2] Engine router — maps intent → SearXNG engines & categories
    │  e.g. research → [arxiv, semantic_scholar, ddg] / [science]
    ▼
[Stage 3] SearXNG fetch — HTTP GET /search, parse results
    │  URL normalisation (strip tracking params)
    │  Deduplication (exact URL + title similarity ≥ 0.85)
    │  Target: ~15 unique results
    ▼
[Stage 4A] Coarse filter — metadata scoring
    │  domain authority × recency × engine trust × position
    │  Keep top 12
    ▼
[Stage 4B] Cross-encoder re-rank — semantic relevance
    │  model: cross-encoder/ms-marco-MiniLM-L-6-v2
    │  final score: 45% semantic + 20% authority + 15% recency
    │              + 10% engine trust + 10% position
    │  Keep top 8
    ▼
[Stage 5] Cache write (24h TTL) + stats counters + JSON response
```

**Graceful degradation:**
- Claude Haiku failure → fallback to `intent=general`, original query
- Cross-encoder failure → metadata scores only
- Redis failure → skip cache (query still succeeds)
- SearXNG failure → HTTP 502 with error detail

---

## Development

```bash
# Run tests
make test

# Lint
make lint

# Auto-format
make format
```

Tests mock all external dependencies (Redis, SearXNG, Anthropic API) so they run offline.
