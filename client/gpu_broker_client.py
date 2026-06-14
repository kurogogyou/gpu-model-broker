"""Reference Python client for the gpu-broker HTTP API.

Single runtime dependency: `requests`. Designed to be vendored into consumer
repos (brain-raglite, /transcribe skill, ai-transcriber, 55places) or installed
from this repo once it ships as a package.

API contract source of truth: `gpu-broker/broker/app.py` (FastAPI app, see
`AcquireRequest` / `AcquireResponse` / `ReleaseRequest` / `StatusResponse` /
`InfeasibleResponse`). The /openapi.json served at `http://<broker>/openapi.json`
is the live machine-readable copy.

Run `python -m client --help` for the CLI (entry point is `client/__main__.py`).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

import requests

log = logging.getLogger("gpu_broker_client")

Role = Literal["embed", "rerank", "transcribe", "llm"]

DEFAULT_BASE_URL = "http://127.0.0.1:8090"

# Cold-start ceilings (measured Phase 1, 2026-05-28):
#   embed cold-load        ~25s
#   rerank cold-load       ~10s
#   transcribe cold-load   ~30s + bake-warm
#   transcribe acquire under eviction adds ~3s for evict+stop
# 600s is a generous ceiling that also accommodates the worst-case whisperx
# alignment-model bake on first-ever acquire. Callers can override per-acquire.
DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 600

# Release should be near-instant — broker just flips a handle to released.
DEFAULT_RELEASE_TIMEOUT_SECONDS = 10

# Status is a synchronous PG-free read of in-memory state.
DEFAULT_STATUS_TIMEOUT_SECONDS = 5


class BrokerError(RuntimeError):
    """Base class for all broker-side failures (HTTP non-2xx, transport errors)."""


class InfeasibleError(BrokerError):
    """Raised when the broker returns 503 — VRAM cannot be reserved for this role.

    `reason`, `free_mb`, and `would_need_evict` mirror the broker's
    InfeasibleResponse so callers can inspect/print/retry intelligently.
    """

    def __init__(self, reason: str, free_mb: int, would_need_evict: list[str]):
        super().__init__(f"infeasible: {reason} (free={free_mb} MiB, would_evict={would_need_evict})")
        self.reason = reason
        self.free_mb = free_mb
        self.would_need_evict = would_need_evict


@dataclass
class BrokerHandle:
    """Active acquire token. Returned by `GpuBrokerClient.acquire` context manager.

    - `endpoint` is the URL the caller should send role-specific requests to
      (e.g. `f"{h.endpoint}/v1/embeddings"`, `f"{h.endpoint}/rerank"`,
      `f"{h.endpoint}/transcribe"`).
    - `evicted` lists roles the broker had to stop to make room.
    - `warmed_implies` lists roles the broker also warmed because the acquired
      role declares `implies: [...]` in roles.yaml (e.g. acquiring `rerank`
      warms `embed`).
    - `cold_start_ms` is the wall-clock the broker spent starting + healthz-ing
      the container; 0 if the worker was already loaded.
    """

    role: str
    endpoint: str
    handle: str
    evicted: list[str] = field(default_factory=list)
    warmed_implies: list[str] = field(default_factory=list)
    cold_start_ms: int = 0


@dataclass
class WorkerStatus:
    """One row of the broker's /status workers table."""

    role: str
    container_id: str
    state: str
    loaded_mb_estimate: int
    last_request_at: float
    idle_shutdown_at: float | None
    cpu_rss_mb: int | None
    pinned: bool
    active_handles: int


class GpuBrokerClient:
    """Thin sync HTTP client wrapping the broker's 5 endpoints + /admin.

    Construct once per consumer; reuse across acquire calls. Carries a single
    `client_id` for broker-side telemetry/observability — pass a stable string
    identifying the calling service (e.g. `"brain-raglite-mcp"`,
    `"transcribe-skill"`, `"ai-transcriber"`, `"55places-rag"`).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        client_id: str = "anonymous",
        session: requests.Session | None = None,
    ):
        if not client_id:
            raise ValueError("client_id must be a non-empty string (broker requires minLength=1)")
        self.base = base_url.rstrip("/")
        self.client_id = client_id
        self._session = session or requests.Session()

    # ----- core lifecycle -----

    @contextlib.contextmanager
    def acquire(
        self,
        role: Role,
        hold_seconds: int | None = None,
        hint_vram_mb: int | None = None,
        timeout: float = DEFAULT_ACQUIRE_TIMEOUT_SECONDS,
    ) -> Iterator[BrokerHandle]:
        """Acquire a worker for `role`, yield a BrokerHandle, release on exit.

        Always releases — even on exception inside the with-block — so callers
        don't have to wrap in try/finally. If the broker is unreachable during
        release, logs a warning rather than raising (the original exception, if
        any, is more interesting; broker-side cleanup catches up via idle sweep).
        """
        payload: dict[str, Any] = {"role": role, "client_id": self.client_id}
        if hold_seconds is not None:
            payload["hold_seconds"] = hold_seconds
        if hint_vram_mb is not None:
            payload["hint_vram_mb"] = hint_vram_mb

        log.debug("acquire role=%s payload=%s", role, payload)
        try:
            r = self._session.post(f"{self.base}/acquire", json=payload, timeout=timeout)
        except requests.RequestException as exc:
            raise BrokerError(f"acquire transport error: {exc}") from exc

        if r.status_code == 503:
            self._raise_infeasible(r)
        if not r.ok:
            raise BrokerError(f"acquire failed: HTTP {r.status_code}: {r.text[:500]}")

        data = r.json()
        h = BrokerHandle(
            role=data["role"],
            endpoint=data["endpoint"],
            handle=data["handle"],
            evicted=data.get("evicted", []),
            warmed_implies=data.get("warmed_implies", []),
            cold_start_ms=data.get("cold_start_ms", 0),
        )
        log.info(
            "acquired role=%s endpoint=%s cold_start_ms=%d evicted=%s warmed_implies=%s",
            h.role, h.endpoint, h.cold_start_ms, h.evicted, h.warmed_implies,
        )
        try:
            yield h
        finally:
            try:
                self.release(h.handle)
            except BrokerError as exc:
                log.warning("release failed (worker will idle out): %s", exc)

    def release(self, handle: str, timeout: float = DEFAULT_RELEASE_TIMEOUT_SECONDS) -> None:
        """Release a handle previously returned by acquire. Idempotent on the broker side."""
        try:
            r = self._session.post(
                f"{self.base}/release", json={"handle": handle}, timeout=timeout,
            )
        except requests.RequestException as exc:
            raise BrokerError(f"release transport error: {exc}") from exc
        if r.status_code == 404:
            log.debug("release: handle %s already released or unknown", handle[:8])
            return
        if not (r.status_code == 204 or r.ok):
            raise BrokerError(f"release failed: HTTP {r.status_code}: {r.text[:500]}")

    # ----- read-only -----

    def status(self, timeout: float = DEFAULT_STATUS_TIMEOUT_SECONDS) -> dict[str, Any]:
        """Return the broker's /status payload as a dict (broker_version, gpu_*_mb, workers, ...)."""
        try:
            r = self._session.get(f"{self.base}/status", timeout=timeout)
        except requests.RequestException as exc:
            raise BrokerError(f"status transport error: {exc}") from exc
        if not r.ok:
            raise BrokerError(f"status failed: HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def workers(self) -> list[WorkerStatus]:
        """Convenience: return the workers list as typed WorkerStatus dataclasses."""
        return [WorkerStatus(**w) for w in self.status().get("workers", [])]

    def health(self, timeout: float = DEFAULT_STATUS_TIMEOUT_SECONDS) -> bool:
        """True if the broker process answers /health with 2xx."""
        try:
            r = self._session.get(f"{self.base}/health", timeout=timeout)
            return r.ok
        except requests.RequestException:
            return False

    # ----- admin (pin/unpin) -----

    def pin(self, role: Role, timeout: float = DEFAULT_STATUS_TIMEOUT_SECONDS) -> None:
        """Mark a role eviction-immune. Used by the bridge-state setup (embed pinned)."""
        self._admin("pin", role, timeout)

    def unpin(self, role: Role, timeout: float = DEFAULT_STATUS_TIMEOUT_SECONDS) -> None:
        """Release a pin. After unpin, the role is subject to idle-shutdown and eviction."""
        self._admin("unpin", role, timeout)

    # ----- internals -----

    def _admin(self, op: str, role: str, timeout: float) -> None:
        try:
            r = self._session.post(f"{self.base}/admin/{op}/{role}", timeout=timeout)
        except requests.RequestException as exc:
            raise BrokerError(f"admin/{op} transport error: {exc}") from exc
        if not (r.status_code == 204 or r.ok):
            raise BrokerError(f"admin/{op}/{role} failed: HTTP {r.status_code}: {r.text[:500]}")

    @staticmethod
    def _raise_infeasible(r: requests.Response) -> None:
        try:
            detail = r.json().get("detail", {})
        except ValueError:
            detail = {}
        raise InfeasibleError(
            reason=detail.get("reason", "infeasible"),
            free_mb=int(detail.get("free_mb", 0)),
            would_need_evict=list(detail.get("would_need_evict", [])),
        )


# ===== CLI =====
# `python -m client.gpu_broker_client status`
# `python -m client.gpu_broker_client acquire rerank --hold 60`
# `python -m client.gpu_broker_client pin embed`
# Mostly useful for ops sanity checks; consumers integrate via the class above.

def _cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gpu_broker_client", description="gpu-broker reference client")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--client-id", default="cli")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="print /status JSON")
    sub.add_parser("health", help="exit 0 if broker /health is 2xx")

    sp_acq = sub.add_parser("acquire", help="acquire a role, hold briefly, release")
    sp_acq.add_argument("role", choices=["embed", "rerank", "transcribe", "llm"])
    sp_acq.add_argument("--hold", type=int, default=None, help="hold_seconds payload")
    sp_acq.add_argument("--dwell", type=float, default=0.0, help="seconds to sleep before release")

    for op in ("pin", "unpin"):
        sp = sub.add_parser(op, help=f"{op} a role")
        sp.add_argument("role", choices=["embed", "rerank", "transcribe", "llm"])

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    c = GpuBrokerClient(base_url=args.base_url, client_id=args.client_id)

    if args.cmd == "status":
        json.dump(c.status(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.cmd == "health":
        return 0 if c.health() else 1
    if args.cmd == "acquire":
        import time as _time
        with c.acquire(args.role, hold_seconds=args.hold) as h:
            print(json.dumps({
                "role": h.role, "endpoint": h.endpoint, "handle": h.handle,
                "evicted": h.evicted, "warmed_implies": h.warmed_implies,
                "cold_start_ms": h.cold_start_ms,
            }, indent=2))
            if args.dwell > 0:
                _time.sleep(args.dwell)
        return 0
    if args.cmd in ("pin", "unpin"):
        getattr(c, args.cmd)(args.role)
        print(f"{args.cmd} {args.role}: ok")
        return 0
    return 2
