"""bge-reranker — cross-encoder reranking HTTP service.

Loads bge-reranker-v2-m3 (or any rerankers-supported model) once at startup,
serves rerank requests over a small JSON API. Designed for broker-managed
lifecycle.

Environment variables:
    RERANK_MODEL     HF model id (default: BAAI/bge-reranker-v2-m3)
    RERANK_DTYPE     fp16 | bf16 | fp32 (default: fp16)
    RERANK_HOST      bind host (default: 0.0.0.0)
    RERANK_PORT      bind port (default: 8082)

HF_HUB_OFFLINE=1 at runtime; model must be pre-cached at build time.
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
from rerankers import Reranker

MODEL_NAME = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DTYPE_NAME = os.environ.get("RERANK_DTYPE", "fp16").lower()
WORKER_ROLE = os.environ.get("WORKER_ROLE", "rerank")
BROKER_VRAM_MB = int(os.environ.get("BROKER_VRAM_MB", "1500"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bge-reranker")

_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
    DTYPE_NAME, torch.float16
)

log.info(f"Loading reranker {MODEL_NAME} dtype={DTYPE_NAME}")
_t0 = time.time()
_reranker = Reranker(MODEL_NAME, model_type="cross-encoder", dtype=_dtype, verbose=0)
log.info(f"Loaded in {time.time() - _t0:.1f}s")

# Warmup pass with a tiny query so first real request doesn't pay the cold cost.
_ = _reranker.rank(query="warmup", docs=["short test doc"])
log.info("Warmup pass complete")

app = FastAPI(title="bge-reranker", version="0.1.0")


class RerankRequest(BaseModel):
    query: str
    docs: list[str]
    top_n: int | None = None
    return_documents: bool = False


class RerankResultItem(BaseModel):
    index: int
    score: float
    document: str | None = None


class RerankResponse(BaseModel):
    object: Literal["rerank"] = "rerank"
    model: str
    results: list[RerankResultItem]
    rerank_ms: float
    num_docs: int


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
        "dtype": DTYPE_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "vram_allocated_mb": _vram_mb(),
    }


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if not req.docs:
        return RerankResponse(model=MODEL_NAME, results=[], rerank_ms=0.0, num_docs=0)
    t0 = time.time()
    try:
        ranking = _reranker.rank(query=req.query, docs=req.docs)
    except Exception as e:
        _mark_model_broken(e)
        raise
    dt = (time.time() - t0) * 1000
    log.info(f"reranked {len(req.docs)} docs in {dt:.1f}ms")

    # rerankers.Reranker.rank returns a RankedResults object; iterate to a list.
    items: list[RerankResultItem] = []
    for r in ranking:
        idx = getattr(r, "doc_id", None)
        if idx is None:
            idx = getattr(r, "document_id", None)
        score = float(getattr(r, "score", 0.0))
        doc_text = None
        if req.return_documents:
            doc_text = req.docs[idx] if idx is not None and 0 <= idx < len(req.docs) else None
        items.append(RerankResultItem(index=int(idx if idx is not None else 0), score=score, document=doc_text))

    if req.top_n is not None:
        items = items[: req.top_n]

    return RerankResponse(
        model=MODEL_NAME,
        results=items,
        rerank_ms=dt,
        num_docs=len(req.docs),
    )


# --- Readiness that actually exercises the model (added 2026-08-27) ---------
# `/healthz` is liveness only and returns 200 even when every inference fails —
# on 2026-08-27 the sibling embed container reported `health=healthy` with 0
# restarts while 100% of requests died on `CUDA error: no kernel image is
# available` (sm_120 vs a cu118 build). Nothing alerted. `/readyz` reflects
# real model health.
#
# Deliberately NOT a periodic GPU call: the broker idle-shuts-down workers, so
# a healthcheck touching the GPU every 30s would pin VRAM forever. Self-test
# ONCE at startup, then latch on the first inference failure.
_MODEL_OK: bool = False
_MODEL_ERR: str | None = None


def _mark_model_broken(e: BaseException) -> None:
    global _MODEL_OK, _MODEL_ERR
    _MODEL_OK = False
    _MODEL_ERR = f"{type(e).__name__}: {e}"
    log.error(f"model marked UNHEALTHY: {_MODEL_ERR}")


try:
    _reranker.rank(query="readiness self-test", docs=["a", "b"])
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
