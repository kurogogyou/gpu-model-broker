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


def _get_diarize_pipeline():
    global _diarize_pipeline
    if _diarize_pipeline is None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise HTTPException(
                status_code=400,
                detail="HF_TOKEN env var not set; diarization unavailable",
            )
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
    }


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    if not os.path.isfile(req.audio_path):
        raise HTTPException(
            status_code=400,
            detail=f"audio_path not found inside container: {req.audio_path}",
        )

    batch_size = req.batch_size if req.batch_size is not None else DEFAULT_BATCH_SIZE
    log.info(f"transcribe audio={req.audio_path} lang={req.language} batch={batch_size} align={req.align} diarize={req.diarize}")

    audio = whisperx.load_audio(req.audio_path)
    duration_s = len(audio) / 16000.0  # whisperx resamples to 16kHz

    # Pass 1: transcription
    t0 = time.time()
    result = _asr_model.transcribe(audio, batch_size=batch_size, language=req.language)
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
