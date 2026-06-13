# bge-reranker — cross-encoder reranking service.
#
# Build:  docker build -t gpu-broker/bge-reranker:latest containers/bge-reranker/
# Run:    docker run --rm --gpus all -p 8082:8082 gpu-broker/bge-reranker:latest
# Health: curl http://localhost:8082/health
# Test:   curl -X POST http://localhost:8082/rerank \
#           -H "Content-Type: application/json" \
#           -d '{"query":"task management","docs":["task tracking system","football match"]}'
#
# Model weights (~1.2 GiB fp16 on GPU) baked into the image. Reranker
# activation memory at realistic K=8 × ~1500-char candidates: +130 MiB.
# Pathological K=20 × 25K chars: +3.6 GiB — broker enforces sensible
# candidate-pool size on the caller side.

# Phase 2.5 (2026-06-13): switched from cudnn8-runtime → runtime base.
# rerankers + bge-reranker-v2-m3 use torch via the transformers cross-encoder
# path, not cuDNN directly. Saves ~3.4 GB per image.
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf-cache \
    HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --upgrade pip==24.3.1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Phase 2.5 (2026-06-13): strip nvidia-cu12 sibling wheels. See tei-embed
# Dockerfile for rationale. Saves ~700 MB – 1 GB per image.
RUN pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
        nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
        nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
        nvidia-nccl-cu12 nvidia-nvtx-cu12 2>/dev/null || true

# Bake reranker weights into a separate layer.
RUN HF_HOME=/models/hf-cache HF_HUB_OFFLINE=0 python3 -c "\
import torch; \
from rerankers import Reranker; \
r = Reranker('BAAI/bge-reranker-v2-m3', model_type='cross-encoder', dtype=torch.float16, verbose=0); \
print('reranker loaded')"

# Phase 2.5 (2026-06-13): audit the HF cache layout. See tei-embed Dockerfile
# for rationale (bge-reranker-v2-m3 also ships fp32 weights only; no safe
# delete pattern without a custom export-fp16 step).
RUN echo '=== HF cache layout (Phase 2.5 audit) ===' && \
    du -ah /models/hf-cache 2>/dev/null | sort -rh | head -20 || true

# Phase 2.5 (2026-06-13): strip __pycache__ + .pyc cruft from site-packages.
RUN find /usr/local/lib/python3.10 /usr/lib/python3.10 \
        -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true ; \
    find /usr/local/lib/python3.10 /usr/lib/python3.10 \
        -name '*.pyc' -delete 2>/dev/null || true

COPY server.py /app/server.py

ENV RERANK_MODEL=BAAI/bge-reranker-v2-m3 \
    RERANK_DTYPE=fp16 \
    RERANK_HOST=0.0.0.0 \
    RERANK_PORT=8082 \
    WORKER_ROLE=rerank \
    BROKER_VRAM_MB=1500

EXPOSE 8082

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8082/healthz || exit 1

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082"]
