"""CPU encoder server: bge-m3 embeddings + bge-reranker-v2-m3 on a host with no GPU.

The RAGLite fallback when no GPU broker is reachable (docs/host-routing.md).
Speaks the same two wire formats as the GPU workers, so a client switches by
base URL alone:
  POST /v1/embeddings  {"input": str | [str]}       -> {"data": [{"index", "embedding"}]}
      (OpenAI-compatible, like tei-embed; CLS pooling + L2 norm = bge-m3 dense)
  POST /rerank         {"query", "docs": [str]}     -> {"results": [{"index", "score"}]}
      (broker dialect, like bge-reranker; raw logits, best first)
  GET  /healthz        always 200; reports what is loaded

Models load on first request and unload after ENCODER_IDLE_UNLOAD_S without
traffic, so the fallback costs no RAM while the GPU path is up.

Env:
  ENCODER_RUNTIME       torch | ov | ort                (default torch: the 2026-10-28
                        benchmark found int8 no faster on an AVX2 CPU without VNNI;
                        README.md)
  ENCODER_MODEL_DIR     dir with embed/ and rerank/ subdirs (exported variant, or
                        for torch the pinned snapshots under those names)
  ENCODER_THREADS       intra-op threads (default 8; 16 was slower on an 8C/16T i7)
  ENCODER_IDLE_UNLOAD_S default 1800; 0 = never unload
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

RUNTIME = os.environ.get("ENCODER_RUNTIME", "torch")
MODEL_DIR = Path(os.environ.get("ENCODER_MODEL_DIR", "/models"))
THREADS = int(os.environ.get("ENCODER_THREADS", "8"))
IDLE_UNLOAD_S = int(os.environ.get("ENCODER_IDLE_UNLOAD_S", "1800"))
EMBED_MAX_LEN, RERANK_MAX_LEN, BATCH = 8192, 512, 16

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("encoder-cpu")
app = FastAPI(title="encoder-cpu")

_models: dict[str, tuple] = {}
_last_use = time.time()
_lock = threading.Lock()


def _load(role: str):
    path = MODEL_DIR / role
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    t = time.monotonic()
    if RUNTIME == "torch":
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification
        torch.set_num_threads(THREADS)
        cls = AutoModel if role == "embed" else AutoModelForSequenceClassification
        model = cls.from_pretrained(path).eval()
    elif RUNTIME == "ov":
        from optimum.intel import OVModelForFeatureExtraction, OVModelForSequenceClassification
        cls = OVModelForFeatureExtraction if role == "embed" else OVModelForSequenceClassification
        model = cls.from_pretrained(path, ov_config={"INFERENCE_NUM_THREADS": str(THREADS),
                                                     "PERFORMANCE_HINT": "LATENCY"})
    elif RUNTIME == "ort":
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification
        so = ort.SessionOptions(); so.intra_op_num_threads = THREADS
        cls = ORTModelForFeatureExtraction if role == "embed" else ORTModelForSequenceClassification
        onnx = next(p.name for p in path.glob("*.onnx"))
        model = cls.from_pretrained(path, file_name=onnx, session_options=so)
    else:
        raise ValueError(f"unknown ENCODER_RUNTIME {RUNTIME!r}")
    log.info("loaded %s runtime=%s from %s in %.1fs", role, RUNTIME, path, time.monotonic() - t)
    return tok, model


def _get(role: str):
    global _last_use
    with _lock:
        _last_use = time.time()
        if role not in _models:
            _models[role] = _load(role)
        return _models[role]


def _forward(model, enc):
    if RUNTIME == "torch":
        import torch
        with torch.inference_mode():
            return model(**{k: torch.as_tensor(v) for k, v in enc.items()})
    return model(**enc)


def _as_np(x):
    return x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)


def _idle_unloader():
    while IDLE_UNLOAD_S:
        time.sleep(30)
        with _lock:
            if _models and time.time() - _last_use > IDLE_UNLOAD_S:
                _models.clear()
                import gc; gc.collect()
                log.info("idle %ds: models unloaded", IDLE_UNLOAD_S)


threading.Thread(target=_idle_unloader, daemon=True).start()


class EmbedReq(BaseModel):
    input: str | list[str]
    model: str | None = None


class RerankReq(BaseModel):
    query: str
    docs: list[str]
    return_documents: bool = False


@app.get("/healthz")
def healthz():
    return {"status": "ok", "runtime": RUNTIME, "loaded": sorted(_models), "threads": THREADS}


@app.post("/v1/embeddings")
def embeddings(req: EmbedReq):
    texts = [req.input] if isinstance(req.input, str) else req.input
    tok, model = _get("embed")
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                  max_length=EMBED_MAX_LEN, return_tensors="np")
        cls = _as_np(_forward(model, dict(enc)).last_hidden_state)[:, 0]
        cls = cls / np.linalg.norm(cls, axis=1, keepdims=True)
        out.extend(cls.tolist())
    return {"object": "list", "model": "BAAI/bge-m3",
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(out)]}


@app.post("/rerank")
def rerank(req: RerankReq):
    tok, model = _get("rerank")
    scores: list[float] = []
    for i in range(0, len(req.docs), BATCH):
        docs = req.docs[i:i + BATCH]
        enc = tok([req.query] * len(docs), docs, padding=True, truncation=True,
                  max_length=RERANK_MAX_LEN, return_tensors="np")
        scores.extend(_as_np(_forward(model, dict(enc)).logits).reshape(-1).tolist())
    order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    return {"results": [{"index": k, "score": scores[k]} for k in order]}
