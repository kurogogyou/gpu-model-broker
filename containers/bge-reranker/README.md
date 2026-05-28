# bge-reranker

Cross-encoder reranking service. Default model: `BAAI/bge-reranker-v2-m3` (XLM-RoBERTa-large family, multilingual). Pair with `tei-embed` for two-stage retrieval (embed → vector search → rerank).

## Build

```
docker build -t gpu-broker/bge-reranker:latest .
```

## Run

```
docker run --rm --gpus all -p 8082:8082 gpu-broker/bge-reranker:latest
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | model id, device, dtype, VRAM allocated |
| `POST` | `/rerank` | rerank a list of candidate docs against a query |

```
curl -X POST http://localhost:8082/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is the task management system",
    "docs": [
      "The brain repo uses Obsidian markdown files for tasks.",
      "Football matches are played on weekends.",
      "Recurring tasks live in */recurring-tasks.md."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

Response shape:
```json
{
  "object": "rerank",
  "model": "BAAI/bge-reranker-v2-m3",
  "results": [
    {"index": 0, "score": 0.92, "document": "The brain repo..."},
    {"index": 2, "score": 0.85, "document": "Recurring tasks..."}
  ],
  "rerank_ms": 1480,
  "num_docs": 3
}
```

## Measured VRAM (GTX 1070, 2026-05-28)

| Stage | VRAM | Latency |
|---|---|---|
| Loaded idle (fp16) | ~1.2 GiB | — |
| K=8 × 1500 chars (realistic) | ~1.3 GiB | 1.5s |
| K=20 × 25000 chars (pathological) | ~5.4 GiB | 9.6 min |

The pathological-case latency is a Pascal architecture cost (no tensor cores). Blackwell / 4060 Ti / 3090 should cut this by 10-20×. RAGLite's default candidate pool (~25 chunks × ~1500 chars each) stays in the realistic band.

## Caller notes

The broker's `acquire(role="rerank")` implicitly warms `embed` as well — a rerank call without a preceding query embed is meaningless, so they're treated as a single workflow. Ingest-only flows should `acquire(role="embed")` directly and leave reranker shut down (frees ~1.2 GiB).

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | HF model id (must be pre-baked) |
| `RERANK_DTYPE` | `fp16` | `fp16` / `bf16` / `fp32` (fp16 required to fit on 8 GiB) |
| `RERANK_HOST` | `0.0.0.0` | Bind host |
| `RERANK_PORT` | `8082` | Bind port |
