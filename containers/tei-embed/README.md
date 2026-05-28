# tei-embed

OpenAI-compatible `/v1/embeddings` server backed by `BAAI/bge-m3` (default) or any sentence-transformers model. Model weights are baked into the image at build time so the running container has zero network dependency.

## Build

```
docker build -t gpu-broker/tei-embed:latest .
```

First build is ~10 min (CUDA base + torch + model download). Incremental rebuilds reuse the model layer.

## Run

```
docker run --rm --gpus all -p 8081:8081 gpu-broker/tei-embed:latest
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | model id, device, dtype, dim, VRAM allocated |
| `POST` | `/v1/embeddings` | OpenAI-compatible embedding endpoint |

```
curl -X POST http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello world", "second doc"]}'
```

## Measured VRAM (GTX 1070, 2026-05-28)

| Stage | VRAM |
|---|---|
| Loaded idle (fp16) | ~1.2 GiB |
| Max-context embed peak (~4500 tokens) | ~1.4 GiB |

Disk-to-VRAM ratio ~3.6× (4.3 GiB HF cache → 1.2 GiB on GPU at fp16).

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `TEI_MODEL` | `BAAI/bge-m3` | HF model id (must be pre-baked) |
| `TEI_DTYPE` | `fp16` | `fp16` / `bf16` / `fp32` |
| `TEI_NORMALIZE` | `1` | L2-normalize embeddings (BGE family preference) |
| `TEI_HOST` | `0.0.0.0` | Bind host |
| `TEI_PORT` | `8081` | Bind port |
