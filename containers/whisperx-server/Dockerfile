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
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

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

# The Phase 2.5/Phase 4 `pip uninstall nvidia-*-cu12` block was REMOVED
# 2026-08-28 with the CUDA 12.8 rebase. It stripped cu12 sibling wheels as
# dead weight because the BASE was cu11 and torch linked cu11 for everything
# except cuDNN/cuBLAS. Under torch 2.7.1+cu128 those siblings are exactly the
# libs torch links against, so the block would now break the image at import.
# Do not reinstate it on a cu12x base. (Its two carve-outs for cudnn/cublas
# are also moot: the 12.8.1-cudnn base ships both natively.)

# Phase 4 (2026-06-14): register the nvidia/{cublas,cudnn}/lib dirs with
# ldconfig so CTranslate2's dlopen of libcublas.so.12 / libcudnn.so.9 resolves
# at runtime. The base image's LD_LIBRARY_PATH only points at
# /usr/local/cuda/targets/x86_64-linux/lib (cu11 libs); the cu12 .so files
# live in pip's nvidia/* layout but aren't on the loader path by default.
# The pre-Phase-4 ai-transcriber bash script papered over this with an
# inline `export LD_LIBRARY_PATH=...` at invocation; here we bake it into
# the image's ldconfig cache so every subprocess (uvicorn worker + threads)
# sees it without arranging env vars at startup.
RUN echo '/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib' > /etc/ld.so.conf.d/nvidia-cu12.conf && \
    echo '/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib' >> /etc/ld.so.conf.d/nvidia-cu12.conf && \
    ldconfig && \
    ldconfig -p | grep -E 'libcublas.so.12|libcublasLt.so.12|libcudnn.so.9' || \
    (echo 'FATAL: cu12 libs still not on loader path' && exit 1)

# Bake whisperx large-v3 (CTranslate2 weights) + bundled VAD into a layer.
# HF_HUB_OFFLINE flipped to 0 only during this RUN so the download succeeds;
# the runtime ENV (set above) stays at 1, blocking any drift.
# torch 2.6+ flipped torch.load(weights_only) to True and pyannote 3.3.2's
# checkpoints carry omegaconf objects; torch_compat allowlists those specific
# types rather than disabling the safe default globally. Needed at BAKE time as
# well as runtime, so it is copied before this step. See torch_compat.py.
COPY torch_compat.py /app/torch_compat.py

RUN HF_HUB_OFFLINE=0 python3 -c "\
import torch_compat; torch_compat.install(); \
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
    WHISPERX_COMPUTE_TYPE=float16 \
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
    CMD curl -fsS http://127.0.0.1:8083/readyz || exit 1

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8083"]
