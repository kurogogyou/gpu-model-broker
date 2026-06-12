# whisperx-server — WhisperX large-v3 + pyannote diarization HTTP service.
#
# Build:  docker build --build-arg HF_TOKEN=hf_xxx -t gpu-broker/whisperx-server:0.1.0 containers/whisperx-server/
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
#   - pyannote/speaker-diarization-3.1 + pyannote/segmentation-3.0 (if
#     HF_TOKEN build arg supplied)
#
# Runtime: HF_HUB_OFFLINE=1 — no network calls, all models load from disk.
# VRAM at batch_size=4 + diarize on a 12-min clip: ~5.8 GiB peak (Phase 1).
# Batch_size=8 OOMs on the 8 GiB GTX 1070; broker enforces 4 via roles.yaml.

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG HF_TOKEN=""

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

# Bake pyannote diarization pipeline only if HF_TOKEN was supplied at build.
# Without it, /transcribe with diarize=true will fail at runtime. The error
# path is clean (HF complains about missing token) so building without a
# token is supported for transcribe+align-only workflows.
#
# HF_TOKEN is a build ARG, not an ENV — it's exported into this RUN's env
# explicitly so the Python pyannote loader can pick it up.
RUN if [ -n "$HF_TOKEN" ]; then \
        HF_HUB_OFFLINE=0 HF_TOKEN="$HF_TOKEN" python3 -c "\
import os; \
from pyannote.audio import Pipeline; \
Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=os.environ['HF_TOKEN']); \
print('pyannote/speaker-diarization-3.1 cached')" ; \
    else \
        echo 'HF_TOKEN not supplied at build time; diarization will fail at runtime' ; \
    fi

COPY server.py /app/server.py

ENV WHISPERX_MODEL=large-v3 \
    WHISPERX_COMPUTE_TYPE=float16 \
    WHISPERX_DEVICE=cuda \
    WHISPERX_DEFAULT_BATCH_SIZE=4 \
    WHISPERX_HOST=0.0.0.0 \
    WHISPERX_PORT=8083 \
    WORKER_ROLE=transcribe \
    BROKER_VRAM_MB=5800

EXPOSE 8083

# 240s start-period — whisperx large-v3 cold-load on a Pascal GPU is slow
# (~30s for the CTranslate2 weights, plus alignment model warmup).
HEALTHCHECK --interval=30s --timeout=10s --start-period=240s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8083/healthz || exit 1

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8083"]
