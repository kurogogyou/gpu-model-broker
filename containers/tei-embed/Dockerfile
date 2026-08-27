# tei-embed — bge-m3 embedder, OpenAI-compatible /v1/embeddings endpoint.
#
# Build:  docker build -t gpu-broker/tei-embed:latest containers/tei-embed/
# Run:    docker run --rm --gpus all -p 8081:8081 gpu-broker/tei-embed:latest
# Health: curl http://localhost:8081/health
# Test:   curl -X POST http://localhost:8081/v1/embeddings \
#           -H "Content-Type: application/json" \
#           -d '{"input":"hello world"}'
#
# Model weights (~1.2 GiB fp16) are baked into the image at build time so
# the running container has zero network dependency and cold-start is
# bounded by SSD read speed, not network. See README for rationale.

# Phase 2.5 (2026-06-13): switched from cudnn8-runtime → runtime base.
# sentence-transformers + bge-m3 don't need cuDNN — only CTranslate2 (whisperx-server) does.
# Saves ~3.4 GB per image.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

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

# Phase 2.5's `pip uninstall nvidia-*-cu12` block was REMOVED 2026-08-27 with the
# CUDA 12.8 rebase. It stripped cu12 sibling wheels as dead weight under the cu11
# base; under torch 2.7.1+cu128 those are exactly the libs torch links against, so
# keeping it would break the image at import. Do not reinstate it on a cu12x base.

# Bake the model weights into the image layer. Done as a separate RUN so the
# (large) weights layer is cached independently of code changes.
RUN HF_HOME=/models/hf-cache HF_HUB_OFFLINE=0 python3 -c "\
import os; \
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('BAAI/bge-m3'); \
print('model dim:', m.get_sentence_embedding_dimension()); \
print('cached under:', os.environ['HF_HOME'])"

# Phase 2.5 (2026-06-13): audit the HF cache layout per file so we KNOW what's
# present before deciding what's safe to delete. bge-m3 ships only fp32
# pytorch_model.bin (~2.27 GB) by default — no fp16 sibling exists — so the
# handoff's `find -name '*.bin' -size +500M -delete` would delete the entire
# model. Audit only at this stage; selective trim is a future optimization.
RUN echo '=== HF cache layout (Phase 2.5 audit) ===' && \
    du -ah /models/hf-cache 2>/dev/null | sort -rh | head -20 || true

# Phase 2.5 (2026-06-13): strip __pycache__ + .pyc cruft from site-packages.
RUN find /usr/local/lib/python3.10 /usr/lib/python3.10 \
        -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true ; \
    find /usr/local/lib/python3.10 /usr/lib/python3.10 \
        -name '*.pyc' -delete 2>/dev/null || true

COPY server.py /app/server.py

ENV TEI_MODEL=BAAI/bge-m3 \
    TEI_DTYPE=fp16 \
    TEI_NORMALIZE=1 \
    TEI_HOST=0.0.0.0 \
    TEI_PORT=8081 \
    WORKER_ROLE=embed \
    BROKER_VRAM_MB=1600

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8081/readyz || exit 1

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8081"]
