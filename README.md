# gpu-broker

VRAM-aware orchestrator for sharing a single consumer GPU between multiple model-serving workloads. Built for the NVIDIA GeForce GTX 1070 (8 GiB) but generalizes to any single-GPU host where total worker residency exceeds physical VRAM.

**Status:** shipped 2026-06-14. All four in-vault consumers migrated. See [Status & history](#status--history) at the bottom.

---

## Quickstart

```bash
# Install (one-time on a fresh host)
git clone <this repo> /opt/brain/src/gpu-broker
python3 -m venv /opt/fast/venvs/gpu-broker
/opt/fast/venvs/gpu-broker/bin/pip install -r requirements.txt
cp systemd/gpu-broker.service ~/.config/systemd/user/

# Start
systemctl --user daemon-reload
systemctl --user enable --now gpu-broker.service

# Verify
curl http://127.0.0.1:8090/status | jq
```

Default port: `127.0.0.1:8090` (localhost only). Override via `BROKER_HOST`/`BROKER_PORT` env in the systemd unit.

---

## What it does

Manages a small set of model-serving Docker containers (text embeddings, cross-encoder reranking, ASR + diarization, optional LLM) and decides which can be loaded together based on measured VRAM budgets. Consumers acquire/release GPU residency through a tiny HTTP API; the broker handles container lifecycle, idle shutdown, and just-in-time eviction.

**One rule does most of the work:** evict the reranker before the transcriber acquires. Everything else is bookkeeping.

---

## Why it exists

On an 8 GiB GPU, naïvely running:

- `bge-m3` embedder (always-on serving) → ~1.2 GiB
- `bge-reranker-v2-m3` (lazy, query-time) → ~1.2 GiB
- `whisperx large-v3` + `pyannote` diarize → up to 5.8 GiB at `batch_size=4`, OOMs at `batch_size=8`

…peaks at ~8.2 GiB and OOMs deterministically on the 8 GiB ceiling. The originating incident (2026-05-26): a single 32-minute transcription OOM'd four times in a row until `tei-shim` was manually stopped.

**Measured result with broker:**

| Configuration | Peak VRAM | Headroom |
|---|---|---|
| Pre-broker compound load (3 workers loaded) | 8046 MiB | OOM at `batch_size=8` |
| Post-broker, `evict_on_acquire: [rerank]` | 7000 MiB | 1.2 GiB free |

The broker also serves as a generic lifecycle supervisor — idle shutdown for always-on services, restart-survives-rediscovery via container labels, multi-tenant arbitration when future workers (LLM) land.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Host: any single-GPU Linux box (initial target: GTX 1070)  │
│                                                             │
│  ┌──────────────┐    ┌──────────────────┐                   │
│  │  consumer    │───▶│ broker (FastAPI) │◀───consumer       │
│  │  (RAG MCP,   │    │ port 8090        │                   │
│  │  transcribe, │    │ owns VRAM ledger │                   │
│  │  future LLM) │    └──────┬───────────┘                   │
│  └──────────────┘           │ start/stop containers         │
│                             ▼                               │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────┐   │
│  │ tei-embed    │    │ bge-reranker     │    │ whisperx │   │
│  │ :8081 GPU    │    │ :8082 GPU lazy   │    │ :8083 GPU│   │
│  │ ~1.2 GiB     │    │ ~1.2 GiB         │    │ ~5.8 GiB │   │
│  └──────────────┘    └──────────────────┘    └──────────┘   │
│         All workers: --gpus all + pinned requirements       │
└─────────────────────────────────────────────────────────────┘
```

**Process model:**

- Broker = one FastAPI process. Owns the VRAM ledger + container lifecycle. Does NOT serve model traffic itself.
- Workers = one Docker container per role. Each exposes its own port. Consumer talks directly to the worker after broker hands back the endpoint.
- `KillMode=process` on the broker's systemd unit: worker containers survive broker restarts. On boot, the broker walks `docker ps -f label=gpu-broker.role` and rebuilds its in-memory state.

---

## Roles

Defined in [`config/roles.yaml`](config/roles.yaml). Each role declares its image, port, VRAM budget (loaded + activation peak), idle-shutdown policy, and eviction relationships.

| Role | Port | Image | Loaded | Peak | Idle | Policy |
|---|---|---|---|---|---|---|
| `embed` | 8081 | `gpu-broker/tei-embed:0.1.0` | 1.2 GiB | +0.4 GiB | 300s | co-resident with `rerank` |
| `rerank` | 8082 | `gpu-broker/bge-reranker:0.1.0` | 1.2 GiB | +0.3 GiB | 300s | co-resident with `embed`, implies `embed` |
| `transcribe` | 8083 | `gpu-broker/whisperx-server:0.1.0` | 0 (lazy) | 5.8 GiB | 30s | `evict_on_acquire: [rerank]`, `batch_size=4` pinned |
| `llm` | 8084 | placeholder | 5.0 GiB | +1.5 GiB | 600s | `evict_on_acquire: [rerank, embed, transcribe]`, `enabled: false` |

**Semantics:**

- **`co_resident_with`**: declarative pairing — these two are expected to run together. The broker won't evict one to make room for the other.
- **`implies`**: acquiring `rerank` warms `embed` for free (reranking typically follows embedding).
- **`evict_on_acquire`**: surgical eviction — when this role is acquired, listed roles are stopped first. This is the load-bearing policy on the 8 GiB ceiling.
- **`idle_shutdown_seconds`**: after the last release, the container is stopped after this many seconds. `active_handles > 0` shields from idle shutdown.
- **`enabled: false`**: role is recognized in the config but `/acquire` returns 503 `InfeasibleError`. Useful for staging future workers (the `llm` slot is wired but no image exists).

**Pinning:** during the consumer-migration bridge week (2026-06-13 → 2026-06-20), the `embed` worker is pinned (`POST /admin/pin/embed`) so legacy `brain-rag` callers keep working at `:8081` while the new path lands. Pinned workers are eviction-immune and never idle-shut-down.

---

## HTTP API (v1)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/acquire` | request a worker; starts container if needed, returns endpoint URL + cold-start latency + which workers were evicted |
| `POST` | `/release` | drop a previously-acquired handle |
| `GET` | `/status` | inspect VRAM ledger + per-worker state |
| `POST` | `/admin/pin/{role}` | mark a role eviction-immune (also disables idle shutdown for that role) |
| `POST` | `/admin/unpin/{role}` | release the pin |

Schemas in [`broker/app.py`](broker/app.py). Errors:

- `503 InfeasibleError` — request can't be satisfied even after eviction (insufficient VRAM budget or role disabled).
- `409 ConflictError` — release with a stale handle.

---

## Python client

Reference client at [`client/gpu_broker_client.py`](client/gpu_broker_client.py). Single `requests` dependency. Surface:

```python
from gpu_broker_client import GpuBrokerClient

broker = GpuBrokerClient("http://127.0.0.1:8090", client_id="brain-raglite")

# Context-manager: acquires, returns handle, releases on exit
with broker.acquire("rerank") as handle:
    # handle.endpoint = "http://127.0.0.1:8082"
    # handle.evicted = []  (or list of roles that were evicted to make room)
    # handle.warmed_implies = ["embed"]  (sibling roles warmed by implies)
    # handle.cold_start_ms = 0  (or measured latency if a container was started)
    response = requests.post(f"{handle.endpoint}/rerank", json=payload)

# Long-hold for batch jobs (e.g. transcribe queue)
with broker.acquire("transcribe", hold_seconds=3600) as handle:
    for clip in batch:
        transcribe_one(handle.endpoint, clip)

# Ops
broker.status()                  # full /status JSON
broker.workers()                 # typed WorkerStatus list
broker.pin("embed")              # bridge state
broker.unpin("embed")            # tear down bridge
```

CLI shim for ops: `python -m client status`, `python -m client pin embed`, etc.

---

## Ops recipes

```bash
# Live state (all-in-one)
curl -s http://127.0.0.1:8090/status | jq

# Watch a specific role's lifecycle
docker logs -f gpu-broker-transcribe

# Tail the broker process itself
journalctl --user -u gpu-broker.service -f

# Force-evict everything (e.g. before a host reboot)
for r in transcribe rerank embed; do curl -X POST http://127.0.0.1:8090/admin/unpin/$r; done
systemctl --user stop gpu-broker.service
docker stop $(docker ps -q -f label=gpu-broker.role)

# Restart broker without losing workers
systemctl --user restart gpu-broker.service
# → workers stay running; broker rediscovers via labels on startup
# → active handles are reset to 0 (callers must re-acquire)

# Rebuild a worker image
cd containers/whisperx-server
docker build -t gpu-broker/whisperx-server:0.1.0 .
# → next /acquire transcribe picks up the new image (broker checks image digest on start)
# → currently-loaded transcribe must be evicted first (it's still running the old image)
curl -X POST http://127.0.0.1:8090/admin/unpin/transcribe  # if pinned
docker stop gpu-broker-transcribe                          # forces cold-restart on next acquire
```

---

## Deployment

| Item | Path |
|---|---|
| Broker code | `/opt/brain/src/gpu-broker/` |
| Broker venv | `/opt/fast/venvs/gpu-broker/` (only `fastapi`, `uvicorn`, `docker`, `pydantic`, `PyYAML`, `httpx`) |
| Worker images | local Docker, tagged `gpu-broker/{tei-embed,bge-reranker,whisperx-server}:0.1.0` |
| Config | `config/roles.yaml` (edit + restart broker to apply) |
| Systemd unit | `~/.config/systemd/user/gpu-broker.service` |
| Env file | `/opt/brain/repo/env/.env` (HF_TOKEN passes through to worker containers) |
| Logs | `journalctl --user -u gpu-broker.service` (broker) + `docker logs gpu-broker-<role>` (per worker) |

**Bake weights into images.** Docker storage stays at `/var/lib/docker/` (SSD on FATEBURN). Per-image weight bake (fp16 single-format) was measured at ~10 GB total across the three worker images. See `docs/decisions.md` in the planning project file for the rationale and inventory baseline.

---

## Worker development guide

To add a new role:

1. **Measure first.** Run the candidate model in isolation under `nvidia-smi` polling. Capture loaded VRAM, peak activation under realistic load, and cold-start time. The 2026-05-28 Phase 1 measurements (planning project file) are the template — without these numbers, eviction policy is guesswork.

2. **Write the Dockerfile** under `containers/<role>/`. Conventions:
   - Base on `nvidia/cuda:11.8.0-runtime-ubuntu22.04` (Pascal-compat). Use `cudnn8-runtime` ONLY if the model linker needs it (`ctranslate2`, some `whisperx` versions). cuDNN8 base is ~3.4 GB bigger.
   - **Don't strip nvidia cu12 wheels blindly.** ctranslate2 4.5+ dlopens `libcublasLt.so.12` at encode-time. Stripping `nvidia-cublas-cu12` causes exit-139 SIGSEGV on first inference call (not at health-check time — `/healthz` will pass). The Phase 2.5 Dockerfile strip list specifically keeps `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`.
   - **Register cu12 .so files with ldconfig** (`/etc/ld.so.conf.d/nvidia-cu12.conf`). Pip's `nvidia/*/lib/` layout is not on the default loader path.
   - Bake model weights into the image (fp16 single-format only, NOT the full HF cache). Set `HF_HUB_OFFLINE=1` at runtime so the container can't drift to a network fetch.
   - Expose `/healthz` and the inference endpoint. `EXPOSE` the chosen port.
   - Add labels: `gpu-broker.role=<role>` and `gpu-broker.image-digest=<digest>` (broker uses these for rediscovery).

3. **Add the role to `config/roles.yaml`** with measured numbers. Set `enabled: false` initially while you debug; flip to `true` once `/healthz` + a real inference call both pass under the broker.

4. **Smoke test through the broker.** `/healthz`-only smoke is not sufficient (Phase 2.5 lesson) — `acquire → /infer → release` is the minimum bar.

---

## Troubleshooting

**`exit-139` / SIGSEGV on first `/transcribe`** — missing `libcublasLt.so.12`. The image build stripped `nvidia-cublas-cu12`. Rebuild with the package kept (see Phase 4 fix in `Mind/projects/done/gpu-model-broker.md` Decisions log).

**`libcublas.so.12 not found or cannot be loaded`** — the package is installed but not on the loader path. Bake `/etc/ld.so.conf.d/nvidia-cu12.conf` pointing at `nvidia/cublas/lib` and `nvidia/cudnn/lib`, then `ldconfig` at image build time.

**`/transcribe --diarize true` fails with HF 401** — `HF_TOKEN` not visible inside the container. Check `docker exec gpu-broker-transcribe env | grep HF_TOKEN`. If empty: refresh `/opt/brain/repo/env/.env`, then `docker stop gpu-broker-transcribe` so the next acquire pulls the new env.

**`/transcribe` works on first session but OOMs on 2nd or 3rd in a bulk batch** — PyTorch CUDA allocator fragmentation. The model holds ~5.7 GiB of cached allocations between sessions, and after a few session-end deallocations the cache is fragmented enough that the next encoder reservation OOMs even though absolute usage is in budget. Symptom: `RuntimeError: CUDA failed with error out of memory` from `whisperx/asr.py:96`, sometimes followed by `cudaErrorInvalidDevice` (corrupted CUDA context). Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the container env by `docker_mgr.py` — uses PyTorch's segment allocator that grows without fragmenting. PyTorch 2.1+ required.

**`/transcribe` returns 400 Bad Request for every session** — the audio path the caller is POSTing isn't visible inside the container. Most common cause: the caller's path resolves through a symlink to a host tree that isn't bind-mounted. Diagnose with `docker exec gpu-broker-transcribe ls -la <path>` — if the directory doesn't exist inside the container, add the underlying tree to the `volumes` dict in `broker/docker_mgr.py` and restart the broker. Currently mounted (RO): `/home/mario`, `/opt/brain`, `/mnt/bigrepo`. Symlink-resolution gotcha: `~/bigrepo` → `/mnt/bigrepo/bigrepo` on FATEBURN; clients call `Path.resolve()` before POSTing, exposing the canonical path.

**`/acquire llm` returns `InfeasibleError`** — `embed` is pinned (1.2 GiB) and the role isn't enabled. Either `POST /admin/unpin/embed` first, or set `enabled: true` after the image exists.

**Broker restart loses my handle** — by design. Worker containers survive (`KillMode=process`), but the broker's in-memory `active_handles` ledger resets to 0. Consumers must re-acquire. Localhost-only deployment makes this cheap (cold-acquire is millisecond-scale on a pinned/loaded worker).

**Pascal GPU + `float16`/`int8_float16`** — CTranslate2 rejects these on Pascal. Whisperx images use `compute_type=int8`. Lifts on Ampere+ (RTX 30/40/50 series) — see [Sunset criteria](#sunset-criteria).

---

## Consumer migration (Phase 4, 2026-06-14)

The broker shipped alongside migration patches across three repos. Per-consumer revert SHAs:

| Consumer | Repo | SHA | What changed |
|---|---|---|---|
| Reference client | gpu-broker | `357d4d0` | shipped `client/gpu_broker_client.py` + CLI + README |
| Whisperx mount | gpu-broker | `a71fa2d` | added `/opt/brain:/opt/brain:ro` bind mount for `audio_path` pass-through |
| Whisperx image fix | gpu-broker | `b2ab2f0` | reinstated `nvidia-cublas-cu12` + ldconfig-registered cu12 .so files |
| `brain-raglite` MCP | brain-repo | `04d0f0d2` | reranker → `BrokerHTTPReranker` HTTP wrapper; ingest wrapped in `acquire("embed")` |
| `/transcribe` skill | brain-repo | `1c1b4bdb` | dropped 5-item pre-flight checklist; broker owns the lifecycle |
| `brain-rag-upgrade` doc | brain-repo | `41deb55f` | deployment-shape callout under Substrate Role |
| `55places` forward-note | brain-repo | `57cfbddf` | future RAGLite-based MCP inherits broker via `BrokerHTTPReranker` |
| `ai-transcriber` | ai-transcriber | `98ec9d4` | repo rewrite — thin dispatcher to `/transcribe`, venv shrunk 9.6 GB → 19 MB |

Rollback sequence (per the planning project file's locked decision):

1. `systemctl --user stop gpu-broker && systemctl --user start tei-shim` (until 2026-06-21 — see Sunset criteria for after-window state)
2. `git revert` the SHAs above per consumer repo
3. Reinstate the manual `systemctl stop tei-shim` recipe in `.claude/skills/transcribe/SKILL.md` Step 2b until the broker issue is resolved.

---

## Sunset criteria

The broker's eviction subsystem is justified by the 8 GiB ceiling on the GTX 1070. When that ceiling lifts, the broker simplifies — eviction becomes dormant code, not removed code.

**Trigger:** GPU upgrade to ≥16 GiB VRAM (RTX 4090 24 GB, RTX 5060 Ti 16 GB, or RTX 5090 32 GB per `Mind/projects/pending/self-hosted-brain-sovereignty.md`).

**Config changes when the trigger fires:**

```yaml
# config/roles.yaml — post-upgrade
roles:
  embed:
    idle_shutdown_seconds: 3600   # was 300; no contention reason to recycle
    # co_resident_with stays — semantic relationship, not VRAM-policy

  rerank:
    idle_shutdown_seconds: 3600   # was 300
    # implies: [embed] stays

  transcribe:
    idle_shutdown_seconds: 600    # was 30; still bursty but no VRAM pressure
    evict_on_acquire: []          # was [rerank]; harmful on 16+ GiB (evicts a warm worker for no VRAM reason)
    default_extra_args:
      - "--batch_size"
      - "16"                      # was "4"; Pascal-OOM-safe default no longer needed

  llm:
    enabled: true                 # was false
    evict_on_acquire: []          # was [rerank, embed, transcribe]
    loaded_mb: <measured>         # 7B-quant or larger
```

**What the broker still does after the upgrade:**

- Lifecycle supervision (start/stop containers; survives broker restarts via labels)
- Multi-tenant arbitration (acquire/release semantics; `active_handles` counter; pinned roles)
- Pinned-dependency surface (consumers still call `acquire("rerank")` — they don't know whether the upgrade landed; the contract is stable)
- Idle shutdown for cost/heat reasons (long timers, not contention timers)
- Observability (the `/status` endpoint is the canonical "what's loaded on the GPU" view)

**What becomes dormant:**

- `policy.py` eviction engine — code stays, paths just stop firing because no role declares `evict_on_acquire`
- Pascal `compute_type=int8` workarounds in `containers/whisperx-server/`
- The `evict_on_acquire: [rerank]` doctrine baked into `/transcribe` SKILL.md (already abstracted by the broker, so callers don't need updates)

**Reactivation triggers** (if eviction ever needs to come back):

- Multi-tenant LLM serving (concurrent inference requests across 2+ users, each needing the LLM at peak activation)
- New worker types beyond what fits comfortably on the upgraded GPU (e.g. a vision model + LLM + transcribe all running together for an agent framework)
- Pinned-residency requirements that change (e.g. embed-pinned for one consumer + rerank-pinned for another, leaving insufficient headroom for transcribe)

The broker stays. Only the policy switches lift.

---

## Status & history

**Operational since:** 2026-06-13 (Phase 3 systemd cutover).
**Consumer migration complete:** 2026-06-14 (Phase 4 — all four in-vault consumers).
**E2E smoke verified:** 2026-06-14 — cold rerank-warm via broker → `/transcribe` cold_start 24.6s + transcribe 14.5s on 204s clip → rerank evicted then cold-warmed → transcribe container idle-shutdown verified at 30s.

Planning project file (full Decisions log, Phase 1 measurements, compound-test numbers): [`Mind/projects/done/gpu-model-broker.md`](../../brain/Mind/projects/done/gpu-model-broker.md).

---

## License

MIT (TBD — public-ready from day one for portfolio purposes; see related portfolio entry under `Mind/projects/active/portfolio-projects-from-past-work.md`).
