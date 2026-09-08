"""whisperx-server — WhisperX large-v3 + pyannote diarization HTTP service.

Loads the whisperx model once at startup, holds it in process memory,
serves transcription requests over a small JSON API. Designed for
broker-managed lifecycle: container starts → model loads → /healthz
flips to ok → broker routes /transcribe requests here.

Audio input is path-based: the caller passes a host path that's bind-
mounted into the container (broker convention: mount /home/mario at the
same path so paths just work for same-host callers). Multipart upload
support is a v2 concern — gated on the 55places consumer waking up
(currently paused per [[job-search-paused]]).

Environment variables (all consumed at import or first-request time):
    WHISPERX_MODEL              model size (default: large-v3)
    WHISPERX_COMPUTE_TYPE       float16 | int8 | float32 (default: float16)
    WHISPERX_DEVICE             cuda | cpu (default: cuda)
    WHISPERX_DEFAULT_BATCH_SIZE batch size if request doesn't specify
                                (default: 4 — measured peak 5.8 GiB on 12-min
                                clip; batch_size=8 OOMs on 8 GiB GTX 1070)
    WHISPERX_HOST               bind host (default: 0.0.0.0)
    WHISPERX_PORT               bind port (default: 8083)
    HF_TOKEN                    required for diarize=true requests
"""
from __future__ import annotations

import logging
import os
import time
from typing import Literal

import torch
import whisperx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_NAME = os.environ.get("WHISPERX_MODEL", "large-v3")
COMPUTE_TYPE = os.environ.get("WHISPERX_COMPUTE_TYPE", "float16")
DEVICE = os.environ.get("WHISPERX_DEVICE", "cuda")
DEFAULT_BATCH_SIZE = int(os.environ.get("WHISPERX_DEFAULT_BATCH_SIZE", "4"))
WORKER_ROLE = os.environ.get("WORKER_ROLE", "transcribe")
BROKER_VRAM_MB = int(os.environ.get("BROKER_VRAM_MB", "5800"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("whisperx-server")

import torch_compat  # noqa: E402 — must run before whisperx/pyannote load checkpoints
torch_compat.install()

log.info(f"Loading whisperx {MODEL_NAME} compute_type={COMPUTE_TYPE} device={DEVICE}")
_t0 = time.time()
_asr_model = whisperx.load_model(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    language=None,
)
log.info(f"ASR model loaded in {time.time() - _t0:.1f}s")

# Alignment + diarization models load lazily on first request that needs them.
# Alignment is per-language; diarization is one global pipeline.
_align_cache: dict[str, tuple] = {}
_diarize_pipeline = None


def _get_align_model(language_code: str):
    if language_code not in _align_cache:
        log.info(f"Loading alignment model for language={language_code}")
        t0 = time.time()
        model, metadata = whisperx.load_align_model(
            language_code=language_code, device=DEVICE
        )
        _align_cache[language_code] = (model, metadata)
        log.info(f"Alignment model {language_code} loaded in {time.time() - t0:.1f}s")
    return _align_cache[language_code]


# --- Diarization capability probe (added 2026-09-08) -----------------------
# The image can be built WITHOUT the pyannote weights: the Dockerfile bake step
# was guarded on a BuildKit secret and downgraded a missing secret to an echo,
# so 0.2.0 shipped with ASR weights only. With HF_HUB_OFFLINE=1 the runtime
# cannot fetch them either, so every diarize=true request died ~60s in with an
# opaque HTTP 500 (LocalEntryNotFoundError) — after paying a cold model load.
#
# Probe the cache at STARTUP instead, and answer diarize=true in milliseconds
# when the answer is going to be no. This is a capability report, not a
# liveness gate: transcribe+align-only is a supported build, so a missing
# pipeline must not take the container down or fail /readyz.
# All three repos, and the pyannote cache — NOT HF_HOME. pyannote's
# Pipeline.from_pretrained defaults to PYANNOTE_CACHE (~/.cache/torch/pyannote
# when unset), so a probe pointed at HF_HOME reports "missing" for weights that
# are present and 503s requests that would have worked. The image pins
# PYANNOTE_CACHE=/models/pyannote-cache so bake and runtime agree.
_PYANNOTE_CACHE = os.environ.get("PYANNOTE_CACHE") or None
_DIARIZE_REPOS = (
    ("pyannote/speaker-diarization-3.1", "config.yaml"),
    ("pyannote/segmentation-3.0", "pytorch_model.bin"),
    ("pyannote/wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin"),
)
_DIARIZE_AVAILABLE: bool = False
_DIARIZE_ERR: str | None = None


def _probe_diarization() -> None:
    """Set _DIARIZE_AVAILABLE / _DIARIZE_ERR from cache + token state. No GPU."""
    global _DIARIZE_AVAILABLE, _DIARIZE_ERR
    if not os.environ.get("HF_TOKEN"):
        _DIARIZE_ERR = ("HF_TOKEN env var not set in the container; pyannote "
                        "refuses to instantiate without it. The broker passes it "
                        "from its secret store — see gpu-broker docker_mgr.py.")
        return
    from huggingface_hub import try_to_load_from_cache

    missing = [
        f"{repo}:{fname}" for repo, fname in _DIARIZE_REPOS
        if not isinstance(
            try_to_load_from_cache(repo, fname, cache_dir=_PYANNOTE_CACHE), str
        )
    ]
    if missing:
        _DIARIZE_ERR = (
            "diarization unavailable: pyannote weights missing from the image "
            f"(not in {_PYANNOTE_CACHE or '~/.cache/torch/pyannote'}): "
            + ", ".join(missing)
            + ". HF_HUB_OFFLINE="
            + os.environ.get("HF_HUB_OFFLINE", "0")
            + " so they cannot be fetched at runtime. Rebuild the image WITH the "
            "BuildKit secret: docker buildx build --secret "
            "id=hf_token,src=/home/mario/.config/gpu-broker/hf-token ..."
        )
        return
    _DIARIZE_AVAILABLE = True


_probe_diarization()
if _DIARIZE_AVAILABLE:
    log.info("diarization available: pyannote weights present in cache")
else:
    log.warning("DIARIZATION UNAVAILABLE — %s", _DIARIZE_ERR)


def _get_diarize_pipeline():
    global _diarize_pipeline
    if _diarize_pipeline is None:
        if not _DIARIZE_AVAILABLE:
            raise HTTPException(status_code=503, detail=_DIARIZE_ERR)
        hf_token = os.environ.get("HF_TOKEN")
        log.info("Loading pyannote diarization pipeline")
        t0 = time.time()
        _diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=DEVICE
        )
        log.info(f"Diarization pipeline loaded in {time.time() - t0:.1f}s")
    return _diarize_pipeline


app = FastAPI(title="whisperx-server", version="0.1.0")


class TranscribeRequest(BaseModel):
    audio_path: str = Field(..., description="Host path to audio file (must be readable inside container via bind mount)")
    language: str | None = Field(None, description="Language code (en, es, ...). None = auto-detect")
    batch_size: int | None = Field(None, description=f"Whisper batch size (default: {DEFAULT_BATCH_SIZE})")
    align: bool = Field(True, description="Run wav2vec2 alignment for word-level timestamps")
    diarize: bool = Field(False, description="Run pyannote speaker diarization (requires HF_TOKEN)")
    min_speakers: int | None = Field(None, description="Hint for diarization")
    max_speakers: int | None = Field(None, description="Hint for diarization")


class TranscribeSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[dict] | None = None


class TranscribeResponse(BaseModel):
    object: Literal["transcribe"] = "transcribe"
    model: str
    language: str
    segments: list[TranscribeSegment]
    duration_s: float
    transcribe_ms: float
    align_ms: float | None = None
    diarize_ms: float | None = None


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
    """Readiness — includes model + VRAM info, slightly more expensive."""
    return {
        "status": "ok",
        "worker_role": WORKER_ROLE,
        "broker_vram_mb": BROKER_VRAM_MB,
        "model": MODEL_NAME,
        "compute_type": COMPUTE_TYPE,
        "device": DEVICE,
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "vram_allocated_mb": _vram_mb(),
        "align_languages_loaded": sorted(_align_cache.keys()),
        "diarize_loaded": _diarize_pipeline is not None,
        "diarize_available": _DIARIZE_AVAILABLE,
        "diarize_error": _DIARIZE_ERR,
    }


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    if not os.path.isfile(req.audio_path):
        raise HTTPException(
            status_code=400,
            detail=f"audio_path not found inside container: {req.audio_path}",
        )

    # Refuse an impossible request in milliseconds rather than after a cold
    # model load + a full transcription pass (2026-09-08: 57s to an opaque 500).
    if req.diarize and not _DIARIZE_AVAILABLE:
        raise HTTPException(status_code=503, detail=_DIARIZE_ERR)

    batch_size = req.batch_size if req.batch_size is not None else DEFAULT_BATCH_SIZE
    log.info(f"transcribe audio={req.audio_path} lang={req.language} batch={batch_size} align={req.align} diarize={req.diarize}")

    audio = whisperx.load_audio(req.audio_path)
    duration_s = len(audio) / 16000.0  # whisperx resamples to 16kHz

    # Pass 1: transcription
    t0 = time.time()
    try:
        result = _asr_model.transcribe(audio, batch_size=batch_size, language=req.language)
    except Exception as e:
        _mark_model_broken(e)
        raise
    transcribe_ms = (time.time() - t0) * 1000
    detected_language = result["language"]
    log.info(f"transcribed {duration_s:.1f}s audio in {transcribe_ms / 1000:.1f}s (language={detected_language})")

    align_ms = None
    if req.align:
        try:
            align_model, metadata = _get_align_model(detected_language)
            t0 = time.time()
            result = whisperx.align(
                result["segments"], align_model, metadata, audio, device=DEVICE,
                return_char_alignments=False,
            )
            align_ms = (time.time() - t0) * 1000
            log.info(f"aligned in {align_ms / 1000:.1f}s")
        except Exception as e:
            log.warning(f"alignment failed for language={detected_language}: {e}")

    diarize_ms = None
    if req.diarize:
        diarize_pipe = _get_diarize_pipeline()
        t0 = time.time()
        diarize_kwargs = {}
        if req.min_speakers is not None:
            diarize_kwargs["min_speakers"] = req.min_speakers
        if req.max_speakers is not None:
            diarize_kwargs["max_speakers"] = req.max_speakers
        diarize_segments = diarize_pipe(audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        diarize_ms = (time.time() - t0) * 1000
        log.info(f"diarized in {diarize_ms / 1000:.1f}s")

    segments_out: list[TranscribeSegment] = []
    for seg in result.get("segments", []):
        segments_out.append(
            TranscribeSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")),
                speaker=seg.get("speaker"),
                words=seg.get("words"),
            )
        )

    return TranscribeResponse(
        model=MODEL_NAME,
        language=detected_language,
        segments=segments_out,
        duration_s=duration_s,
        transcribe_ms=transcribe_ms,
        align_ms=align_ms,
        diarize_ms=diarize_ms,
    )


# --- Readiness that actually exercises the model (added 2026-08-28) ---------
# `/healthz` is liveness only. On 2026-08-27 the sibling embed container
# reported health=healthy with 0 restarts while 100% of requests died on
# `CUDA error: no kernel image is available` (sm_120 vs a cu118 build), because
# its healthcheck probed the port and not the model. `/readyz` reflects real
# model health and reports arch_list + device_capability in its 503 body, so
# the next occurrence of that class names its own cause.
#
# Deliberately NOT a periodic GPU call: the broker idle-shuts-down workers, so
# a healthcheck touching the GPU every 30s would pin VRAM forever. Self-test
# ONCE at startup, then latch on the first inference failure.
#
# whisperx note: the startup self-test runs 1s of SILENCE through the real ASR
# path rather than just poking torch. CTranslate2 (the engine) and torch are
# SEPARATE CUDA stacks here -- ct2 4.8.0 already ran on sm_120 while torch
# could not -- so a torch-only probe would have passed during the outage and
# proved nothing about transcription.
_MODEL_OK: bool = False
_MODEL_ERR: str | None = None


def _mark_model_broken(e: BaseException) -> None:
    global _MODEL_OK, _MODEL_ERR
    _MODEL_OK = False
    _MODEL_ERR = f"{type(e).__name__}: {e}"
    log.error(f"model marked UNHEALTHY: {_MODEL_ERR}")


try:
    import numpy as _np
    _asr_model.transcribe(_np.zeros(16000, dtype=_np.float32), batch_size=1)
    _MODEL_OK = True
    log.info("startup self-test PASSED — ASR executes on this device")
except Exception as _e:  # noqa: BLE001
    _mark_model_broken(_e)
    log.error("startup self-test FAILED — /readyz will report 503")


@app.get("/readyz")
def readyz():
    """Readiness — 503 unless the ASR model has actually executed here."""
    import torch as _t
    if _MODEL_OK:
        return {"status": "ok", "compute_type": COMPUTE_TYPE,
                "arch_list": _t.cuda.get_arch_list(),
                "diarize_available": _DIARIZE_AVAILABLE,
                "diarize_error": _DIARIZE_ERR}
    return JSONResponse(
        status_code=503,
        content={"status": "unhealthy", "error": _MODEL_ERR,
                 "compute_type": COMPUTE_TYPE,
                 "arch_list": _t.cuda.get_arch_list(),
                 "device_capability": list(_t.cuda.get_device_capability())
                 if _t.cuda.is_available() else None},
    )
