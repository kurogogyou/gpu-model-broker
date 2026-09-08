# whisperx-server

WhisperX large-v3 + pyannote diarization, wrapped in a small FastAPI service
for broker-managed lifecycle.

## Build

```
# Standard build — WITH diarization. The BuildKit secret is mandatory.
docker buildx build \
  --secret id=hf_token,src=/home/mario/.config/gpu-broker/hf-token \
  -t gpu-broker/whisperx-server:0.2.1 \
  containers/whisperx-server/

# Transcribe + align only, no diarization — must be requested explicitly
docker buildx build \
  --build-arg WITH_DIARIZATION=0 \
  -t gpu-broker/whisperx-server:0.2.1-noattr \
  containers/whisperx-server/
```

⚠️ **`--build-arg HF_TOKEN=...` no longer does anything.** Phase 2.5 (2026-06-13)
moved the token to a BuildKit `--mount=type=secret`, but this file kept
documenting the build-arg form for eleven weeks. A stray build arg is ignored
without error, so the build "succeeded" and produced an image with no pyannote
weights — see the 2026-09-08 diagnosis. Both the missing secret and an empty
cache after the bake are now hard build failures.

The token is only used at build time to download the pyannote weights into the
image cache. It is **not** persisted as an ENV in the running image — for
diarization at runtime, pass `-e HF_TOKEN=...` to `docker run` (the broker does
this from its configured secret store).

Build prerequisites (once per HuggingFace account):

1. Accept the licenses at
   <https://huggingface.co/pyannote/speaker-diarization-3.1> and
   <https://huggingface.co/pyannote/segmentation-3.0>.
2. Create a read-only HF token at
   <https://huggingface.co/settings/tokens>.

## Run

```
docker run --rm --gpus all \
  -p 8083:8083 \
  -e HF_TOKEN=hf_xxxxxxxxxxxx \
  -v /home/mario:/home/mario:ro \
  gpu-broker/whisperx-server:0.1.0
```

The `-v /home/mario:/home/mario:ro` mount is the broker convention so callers
can pass host paths directly in the `audio_path` field. For non-same-host
callers (future: 55places), a v2 multipart-upload endpoint is on the
roadmap — gated on the consumer waking up.

## API

```
GET  /healthz
  → 200 {"status":"ok"}
  Liveness only; no GPU calls. Used by Docker HEALTHCHECK.

GET  /health
  → 200 {
      "status":"ok",
      "model":"large-v3",
      "compute_type":"float16",
      "device":"cuda",
      "default_batch_size":4,
      "vram_allocated_mb":3120,
      "align_languages_loaded":["en"],
      "diarize_loaded":false
    }
  Readiness + state. Broker polls this for the /status endpoint.

POST /transcribe
  body: {
    "audio_path":"/home/mario/sample.mp3",   # required, container-visible path
    "language":"en",                          # optional, null = auto-detect
    "batch_size":4,                           # optional, default from env
    "align":true,                             # optional, default true
    "diarize":false,                          # optional, default false
    "min_speakers":null,                      # optional, diarize hint
    "max_speakers":null                       # optional, diarize hint
  }
  → 200 {
    "object":"transcribe",
    "model":"large-v3",
    "language":"en",
    "segments":[{"start":0.5,"end":3.2,"text":"...","speaker":"SPEAKER_00","words":[...]}, ...],
    "duration_s":723.4,
    "transcribe_ms":86120.3,
    "align_ms":4022.1,
    "diarize_ms":12118.5
  }
```

## Measured VRAM (GTX 1070, 2026-05-28)

| Scenario | Loaded | Active peak | Notes |
|---|---|---|---|
| ASR model loaded idle | ~2.0 GiB | — | float16 + CTranslate2 |
| 26s clip, batch=8, no diarize | — | +2960 MiB | clean exit |
| **12-min clip, batch=8, diarize** | — | **OOM 7963 MiB** | broker forbids batch=8 |
| **12-min clip, batch=4, diarize** | — | **+5786 MiB** | 86s wall, 2406 MiB headroom |

The broker enforces `default_extra_args: ["--batch_size", "4"]` in
`roles.yaml` so callers can't trigger the batch=8 OOM by accident. The env
var `WHISPERX_DEFAULT_BATCH_SIZE` sets the in-container default; callers can
still pass a `batch_size` in the request body to override (the broker
should reject overrides above 4 at the API layer once that gate ships).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `WHISPERX_MODEL` | `large-v3` | model size — only `large-v3` is baked into the image; others need a rebuild |
| `WHISPERX_COMPUTE_TYPE` | `float16` | `int8` is an option for lower-VRAM hosts but unused on FATEBURN |
| `WHISPERX_DEVICE` | `cuda` | switch to `cpu` for hostless debugging |
| `WHISPERX_DEFAULT_BATCH_SIZE` | `4` | per Phase 1 measurements |
| `WHISPERX_HOST` | `0.0.0.0` | bind host |
| `WHISPERX_PORT` | `8083` | bind port |
| `HF_TOKEN` | (none) | required if any request uses `diarize=true` |

## Caller notes

- Path-based audio only in v1. `audio_path` must resolve inside the container
  via a bind mount the broker set up. Mismatches return `400`.
- The first request in a given language pays a ~10s cold-load for the
  wav2vec2 alignment model (only `en`/`es` are pre-baked; other languages
  download on demand, which means HF_HUB_OFFLINE=1 will block them).
- The first `diarize=true` request pays a ~15s cold-load for the pyannote
  pipeline.
- The broker's `evict_on_acquire: [rerank]` ensures the reranker is not
  loaded when whisperx is serving a batch=4 + diarize job. Embedder stays
  resident — the headroom math fits.

## Pinned versions

See `requirements.txt`. The whole pin set unwinds when the GPU upgrade
lands (5060 Ti, 3-6 month horizon per
[`self-hosted-brain-sovereignty`](../../../../../home/mario/brain/Mind/projects/pending/self-hosted-brain-sovereignty.md)).
Until then, treat the pin set as the source of truth and don't bump
upstream without a full bench rerun against the Phase 1 baseline.
