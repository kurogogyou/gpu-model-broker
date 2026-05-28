# gpu-broker

VRAM-aware orchestrator for sharing a single consumer GPU between multiple model-serving workloads. Built for the NVIDIA GeForce GTX 1070 (8 GiB) but generalizes to any single-GPU host.

## What it does

Manages a small set of model-serving Docker containers (text embeddings, cross-encoder reranking, ASR + diarization) and decides which can be loaded together based on measured VRAM budgets. Workers acquire/release GPU residency through a tiny HTTP API; the broker handles container lifecycle, idle shutdown, and just-in-time eviction.

## Why it exists

On an 8 GiB GPU, naively running:

- bge-m3 embedder (always-on serving) → ~1.2 GiB
- bge-reranker-v2-m3 (lazy, query-time) → ~1.2 GiB
- whisperx large-v3 + pyannote diarize → up to 5.8 GiB at `batch_size=4`, OOMs at `batch_size=8`

…together totals ~8.2 GiB and OOMs deterministically on the 8 GiB ceiling. Without orchestration, the typical workflow (RAG queries during a long transcription job) crashes. The broker enforces a single rule — `evict reranker before transcribe acquire` — which brings peak to 7.0 GiB with headroom.

## Design

```
┌─────────────────────────────────────────────────────────────┐
│  Host: any single-GPU Linux box (initial target: GTX 1070)  │
│                                                             │
│  ┌──────────────┐    ┌──────────────────┐                  │
│  │  consumer    │───▶│ broker (FastAPI) │◀───consumer      │
│  │  (RAG MCP,   │    │ port 8090        │                  │
│  │  transcribe, │    │ owns VRAM ledger │                  │
│  │  …)          │    └──────┬───────────┘                  │
│  └──────────────┘           │ start/stop containers        │
│                             ▼                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────┐ │
│  │ tei-embed    │    │ bge-reranker     │    │ whisperx │ │
│  │ :8081 GPU    │    │ :8082 GPU lazy   │    │ :8083 GPU│ │
│  │ ~1.2 GiB     │    │ ~1.2 GiB         │    │ ~5.8 GiB │ │
│  └──────────────┘    └──────────────────┘    └──────────┘ │
│         All workers: --gpus all + pinned requirements      │
└─────────────────────────────────────────────────────────────┘
```

## API (v1)

See [`config/roles.yaml`](config/roles.yaml) for the policy config. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/acquire` | request a worker; broker starts container if needed, returns endpoint URL + cold-start latency + which workers were evicted |
| `POST` | `/release` | drop hold on a previously-acquired worker |
| `GET` | `/status` | inspect VRAM ledger + per-worker state |
| `POST` | `/admin/pin/{role}` | mark a role eviction-immune |
| `POST` | `/admin/unpin/{role}` | release the pin |

## Repo layout

```
gpu-broker/
├── README.md              # this file
├── requirements.txt       # broker process deps (FastAPI + docker SDK)
├── config/
│   └── roles.yaml         # per-role VRAM budgets + idle timers + eviction policy
├── broker/                # FastAPI app
├── containers/
│   ├── tei-embed/         # bge-m3 embedder (HTTP server, OpenAI-compatible)
│   ├── bge-reranker/      # cross-encoder reranker (HTTP server)
│   └── whisperx-server/   # whisperx + pyannote (HTTP server)
└── systemd/
    └── gpu-broker.service # always-on broker unit
```

## Status

Phase 1 (design + spike) complete on the consumer-side planning artifact (see [related project](../../brain/Mind/projects/active/gpu-model-broker.md) on the planning side). Phase 2 (containerize workers) in progress here.

## License

MIT (TBD — public-ready from day one for portfolio purposes).
