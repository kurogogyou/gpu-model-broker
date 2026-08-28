"""torch.load compatibility shim for pyannote 3.3.2 under torch >= 2.6.

Why this exists
---------------
PyTorch 2.6 flipped the default of `torch.load(weights_only=...)` from False to
True. pyannote-audio 3.3.2's checkpoints are legacy-format pickles carrying
omegaconf config objects, torch version tags and persistent-id instructions, so
under torch >= 2.6 they fail to load at all.

This is a genuine torch-2.6 incompatibility that the DECLARED metadata does not
express: whisperx 3.3.2 requires only `torch>=2` and pyannote-audio 3.3.2 only
`torch>=2.0.0`. Both are satisfied by torch 2.7.1, and both still break. The
CUDA 12.8 rebase had to be tested, not reasoned about from version ranges.

What was tried first, and why it was abandoned
----------------------------------------------
1. `add_safe_globals([...])` — allowlist the classes. Whack-a-mole with a
   ~3-minute image rebuild per round: allowing `omegaconf.listconfig.ListConfig`
   surfaced `torch.torch_version.TorchVersion`, and behind that a
   `persistent_id` instruction that no allowlist can address, because it is a
   property of the legacy pickle FORMAT rather than of any one class.
2. Gating the fallback on the checkpoint's filesystem path. Correct in spirit,
   but the paths are chosen inside lightning/pyannote rather than by us, so the
   check rejected loads it should have permitted.

What this does instead
----------------------
Retry with `weights_only=False`, but only after a `weights_only=True` load has
actually raised, and log the file every single time so the set of affected
checkpoints stays visible in the container log rather than becoming invisible
policy.

Why that is bounded HERE (this argument does not transfer)
----------------------------------------------------------
The risk of `weights_only=False` is executing code from an attacker-chosen
pickle. In this container there is no path from user input to `torch.load`:

  * the ONLY user-supplied value is `TranscribeRequest.audio_path`, and it goes
    to `whisperx.load_audio()`, which shells out to ffmpeg — never to pickle;
  * every `torch.load` call is model-loading code (whisperx ASR, the alignment
    model, the pyannote diarization pipeline) reading checkpoints baked into an
    image layer at BUILD time from official HuggingFace repos;
  * the container runs `HF_HUB_OFFLINE=1`, so nothing at runtime can fetch a
    different checkpoint.

A hostile pickle would therefore have to already be inside the image, at which
point unpickling is not the weak link. If this server ever gains an endpoint
that accepts a model/checkpoint path from a caller, THIS SHIM MUST GO — that is
the change that would break the argument above.
"""
from __future__ import annotations

import logging
import pickle

log = logging.getLogger(__name__)

_installed = False


def install() -> None:
    """Idempotently wrap torch.load. Safe to call more than once."""
    global _installed
    if _installed:
        return

    import torch

    if not hasattr(torch.serialization, "add_safe_globals"):
        log.info("torch < 2.6 — weights_only already defaults to False, shim not needed")
        _installed = True
        return

    _orig = torch.load

    def _patched(f, *args, **kwargs):
        try:
            return _orig(f, *args, **kwargs)
        except pickle.UnpicklingError as exc:
            if kwargs.get("weights_only") is False:
                raise  # already permissive; the failure is real
            log.warning(
                "torch.load(weights_only=True) rejected %r (%s); retrying with "
                "weights_only=False. Bounded: no user-supplied path reaches "
                "torch.load in this container — see torch_compat docstring.",
                f, type(exc).__name__,
            )
            # REWIND FIRST. lightning hands torch.load an open file-like object
            # (fsspec LocalFileOpener), and the failed attempt has already
            # consumed part of the stream. Retrying without seeking reads from
            # the middle of the pickle and fails as
            #   "A load persistent id instruction was encountered"
            # — which looks like a DIFFERENT, deeper incompatibility and sent
            # this debugging down a wrong path for two rebuilds. The retry only
            # ever worked once the handle was rewound.
            if hasattr(f, "seek"):
                try:
                    f.seek(0)
                except Exception:  # noqa: BLE001 — non-seekable stream
                    log.error("cannot rewind %r; retry will likely fail", f)
            kwargs["weights_only"] = False
            return _orig(f, *args, **kwargs)

    torch.load = _patched
    _installed = True
    log.info("torch.load weights_only fallback shim installed (pyannote 3.3.2 + torch>=2.6)")
