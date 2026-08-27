"""tei-embed — OpenAI-compatible /v1/embeddings server.

Loads a sentence-transformers model once at startup and serves embeddings
over HTTP. Designed for broker-managed lifecycle: container starts → model
loads → /health flips to ok → broker routes acquire requests here.

Environment variables (all consumed once at import time):
    TEI_MODEL        HF model id (default: BAAI/bge-m3)
    TEI_DTYPE        fp16 | bf16 | fp32  (default: fp16, BGE family preference)
    TEI_NORMALIZE    "1" to L2-normalize embeddings (default: "1", BGE preference)
    TEI_HOST         bind host (default: 0.0.0.0)
    TEI_PORT         bind port (default: 8081)

The container runs with HF_HUB_OFFLINE=1; the model must be pre-cached at
build time (Dockerfile does this). Network calls at runtime would fail.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Literal

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("TEI_MODEL", "BAAI/bge-m3")
DTYPE_NAME = os.environ.get("TEI_DTYPE", "fp16").lower()
NORMALIZE = os.environ.get("TEI_NORMALIZE", "1") == "1"
WORKER_ROLE = os.environ.get("WORKER_ROLE", "embed")
BROKER_VRAM_MB = int(os.environ.get("BROKER_VRAM_MB", "1600"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tei-embed")

_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
    DTYPE_NAME, torch.float16
)

log.info(f"Loading {MODEL_NAME} dtype={DTYPE_NAME} normalize={NORMALIZE}")
_t0 = time.time()
_model = SentenceTransformer(MODEL_NAME, model_kwargs={"torch_dtype": _dtype})
log.info(
    f"Loaded in {time.time() - _t0:.1f}s "
    f"device={_model.device} dim={_model.get_sentence_embedding_dimension()}"
)

app = FastAPI(title="tei-embed", version="0.1.0")


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = MODEL_NAME
    encoding_format: Literal["float", "base64"] = "float"
    user: str | None = None


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


def _vram_mb() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated() / (1024 * 1024))


@app.get("/healthz")
def healthz() -> dict:
    """Liveness only — no GPU calls, fast for HEALTHCHECK polling."""
    return {"status": "ok"}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "worker_role": WORKER_ROLE,
        "broker_vram_mb": BROKER_VRAM_MB,
        "model": MODEL_NAME,
        "device": str(_model.device),
        "dtype": DTYPE_NAME,
        "dim": _model.get_sentence_embedding_dimension(),
        "normalize": NORMALIZE,
        "vram_allocated_mb": _vram_mb(),
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    texts = [req.input] if isinstance(req.input, str) else req.input
    t0 = time.time()
    try:
        vectors = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=NORMALIZE)
    except Exception as e:
        _mark_model_broken(e)
        raise
    log.info(f"embedded {len(texts)} in {(time.time() - t0) * 1000:.1f}ms")
    return EmbeddingResponse(
        data=[EmbeddingData(index=i, embedding=v.tolist()) for i, v in enumerate(vectors)],
        model=req.model,
        usage=EmbeddingUsage(prompt_tokens=sum(len(t.split()) for t in texts)),
    )


# --- Readiness that actually exercises the model (added 2026-08-27) ---------
# `/healthz` is liveness only and returns 200 even when every inference fails —
# on 2026-08-27 this container reported `health=healthy` with 0 restarts while
# 100% of requests died on `CUDA error: no kernel image is available` (sm_120
# vs a cu118 build). Nothing alerted. `/readyz` reflects real model health.
#
# Deliberately NOT a periodic GPU call: the broker idle-shuts-down workers, so
# a healthcheck that touched the GPU every 30s would pin VRAM forever. Instead
# we self-test ONCE at startup and latch on the first inference failure.
_MODEL_OK: bool = False
_MODEL_ERR: str | None = None


def _mark_model_broken(e: BaseException) -> None:
    global _MODEL_OK, _MODEL_ERR
    _MODEL_OK = False
    _MODEL_ERR = f"{type(e).__name__}: {e}"
    log.error(f"model marked UNHEALTHY: {_MODEL_ERR}")


try:
    _model.encode(["readiness self-test"], convert_to_numpy=True)
    _MODEL_OK = True
    log.info("startup self-test PASSED — model executes on this device")
except Exception as _e:  # noqa: BLE001
    _mark_model_broken(_e)
    log.error("startup self-test FAILED — /readyz will report 503")


@app.get("/readyz")
def readyz():
    """Readiness — 503 unless the model has actually executed on this device."""
    if _MODEL_OK:
        return {"status": "ok", "arch_list": torch.cuda.get_arch_list()}
    return JSONResponse(
        status_code=503,
        content={"status": "unhealthy", "error": _MODEL_ERR,
                 "arch_list": torch.cuda.get_arch_list(),
                 "device_capability": list(torch.cuda.get_device_capability())
                 if torch.cuda.is_available() else None},
    )

