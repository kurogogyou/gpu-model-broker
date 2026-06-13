# syntax=docker/dockerfile:1.7
# whisperx-server — WhisperX large-v3 + pyannote diarization HTTP service.
#
# Build (BuildKit required for --secret):
#   docker buildx build \
#     --secret id=hf_token,src=/home/mario/.config/gpu-broker/hf-token \
#     -t gpu-broker/whisperx-server:0.1.0 \
#     containers/whisperx-server/
#
# Run:    docker run --rm --gpus all -p 8083:8083 \
#             -e HF_TOKEN=hf_xxx \
#             -v /home/mario:/home/mario:ro \
#             gpu-broker/whisperx-server:0.1.0
# Health: curl http://localhost:8083/health
# Test:   curl -X POST http://localhost:8083/transcribe \
#           -H "Content-Type: application/json" \
#           -d '{"audio_path":"/home/mario/sample.mp3","language":"en"}'
#
# Model weights baked into the image at build time:
#   - whisperx large-v3 (~1.5 GiB fp16 via faster-whisper / CTranslate2)
#   - whisperx wav2vec2 alignment models (downloaded lazily per-language;
#     prefetch a few common ones at build time)
#   - whisperx bundled VAD weights (frozen at the image's whisperx==3.3.2
#     SHA — kills the "bundled-SHA dance" once and for all)
#   - pyannote/speaker-diarization-3.1 + pyannote/segmentation-3.0 (only if
#     the hf_token BuildKit secret is supplied; otherwise diarization fails
#     at runtime with a clear HF-token error)
#
# Runtime: HF_HUB_OFFLINE=1 — no network calls, all models load from disk.
# VRAM at batch_size=4 + diarize on a 12-min clip: ~5.8 GiB peak (Phase 1).
# Batch_size=8 OOMs on the 8 GiB GTX 1070; broker enforces 4 via roles.yaml.

# Phase 2.5 (2026-06-13): cudnn8-runtime base RETAINED for this image —
# CTranslate2 (the engine behind faster-whisper / whisperx) links against
# cuDNN at runtime. tei-embed + bge-reranker switched to the plain runtime
# base. Per-image base-image divergence is intentional.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf-cache \
    HF_HUB_OFFLINE=1 \
    TORCH_HOME=/models/torch-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv ca-certificates \
        ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --upgrade pip==24.3.1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Phase 2.5 (2026-06-13): strip nvidia-cu12 sibling wheels — BUT NOT
# nvidia-cudnn-cu12. CTranslate2 (the engine behind faster-whisper / whisperx)
# links cuDNN 9 at runtime, and the base image only ships cuDNN 8 — so
# nvidia-cudnn-cu12==9.1.0.70 from requirements.txt is the load-bearing way
# we get cuDNN 9 into the image. Stripping it produces the runtime error
# `ImportError: libcudnn.so.9: cannot open shared object file` at model bake.
# This divergence vs tei-embed / bge-reranker is intentional; the other cu12
# sibling packages are still dead weight here (torch links cu11 base libs
# for everything except cuDNN).
RUN pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
        nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 \
        nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
        nvidia-nccl-cu12 nvidia-nvtx-cu12 2>/dev/null || true

# Bake whisperx large-v3 (CTranslate2 weights) + bundled VAD into a layer.
# HF_HUB_OFFLINE flipped to 0 only during this RUN so the download succeeds;
# the runtime ENV (set above) stays at 1, blocking any drift.
RUN HF_HUB_OFFLINE=0 python3 -c "\
import whisperx; \
m = whisperx.load_model('large-v3', device='cpu', compute_type='int8', language='en'); \
print('whisperx large-v3 cached')"

# Bake common alignment models (en, es) so first-request latency in those
# languages doesn't pay a download cost. Other languages will fail under
# HF_HUB_OFFLINE=1 — extend this list when a new language is regularly used.
RUN HF_HUB_OFFLINE=0 python3 -c "\
import whisperx; \
[whisperx.load_align_model(language_code=lc, device='cpu') for lc in ('en', 'es')]; \
print('alignment models cached: en, es')"

# Bake pyannote diarization pipeline only if the hf_token BuildKit secret was
# supplied at build (mount it with `--secret id=hf_token,src=<path>`). Without
# it, /transcribe with diarize=true will fail at runtime with a clear HF-token
# error — building without the secret is supported for transcribe+align-only
# workflows.
#
# Phase 2.5 (2026-06-13): switched from `ARG HF_TOKEN` (which leaks into
# `docker history` and image-layer metadata) to BuildKit's `--mount=type=secret`,
# which exposes the secret only at /run/secrets/hf_token during this single RUN
# (tmpfs-backed; never persisted to image or history).
RUN --mount=type=secret,id=hf_token,target=/run/secrets/hf_token \
    if [ -s /run/secrets/hf_token ]; then \
        HF_HUB_OFFLINE=0 HF_TOKEN="$(tr -d '\n\r' < /run/secrets/hf_token)" python3 -c "\
import os; \
from pyannote.audio import Pipeline; \
Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=os.environ['HF_TOKEN']); \
print('pyannote/speaker-diarization-3.1 cached')" ; \
    else \
        echo 'hf_token secret not supplied at build time; diarization will fail at runtime' ; \
    fi

# Phase 2.5 (2026-06-13): audit HF + torch cache layouts so a future trim pass
# can target safely. Whisperx CTranslate2 weights live in /models/torch-cache;
# alignment + diarization models live in /models/hf-cache.
RUN echo '=== HF cache (Phase 2.5 audit) ===' && \
    du -ah /models/hf-cache 2>/dev/null | sort -rh | head -20 || true ; \
    echo '=== torch cache (Phase 2.5 audit) ===' && \
    du -ah /models/torch-cache 2>/dev/null | sort -rh | head -20 || true

# Phase 2.5 (2026-06-13): strip __pycache__ + .pyc cruft from site-packages.
RUN find /usr/local/lib/python3.10 /usr/lib/python3.10 \
        -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true ; \
    find /usr/local/lib/python3.10 /usr/lib/python3.10 \
        -name '*.pyc' -delete 2>/dev/null || true

COPY server.py /app/server.py

ENV WHISPERX_MODEL=large-v3 \
    WHISPERX_COMPUTE_TYPE=int8 \
    WHISPERX_DEVICE=cuda \
    WHISPERX_DEFAULT_BATCH_SIZE=4 \
    WHISPERX_HOST=0.0.0.0 \
    WHISPERX_PORT=8083 \
    WORKER_ROLE=transcribe \
    BROKER_VRAM_MB=5800
# Phase 2.5 (2026-06-13): default compute_type pinned to int8 (was float16).
# Pascal compute capability 6.1 (GTX 1070) rejects both float16 AND
# int8_float16 via CTranslate2 with "do not support efficient ... computation"
# — only int8 and float32 are accepted. int8 is what the bake step uses
# anyway, and it's what ai-transcriber's known-good worker runs on. Lifts
# when the GPU upgrade (5060 Ti / 4090) lands and we can use float16
# natively — same condition that lifts the whole 11.8/cudnn8 pin stack.

EXPOSE 8083

# 240s start-period — whisperx large-v3 cold-load on a Pascal GPU is slow
# (~30s for the CTranslate2 weights, plus alignment model warmup).
HEALTHCHECK --interval=30s --timeout=10s --start-period=240s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8083/healthz || exit 1

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8083"]
