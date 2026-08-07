"""FastAPI app — the broker HTTP entrypoint.

5 endpoints + a /health for the broker itself. All state transitions go through
State.lock so eviction + acquire don't race.

Run via `python -m broker` (see __main__.py) or `uvicorn broker.app:app`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from . import __version__, config, observability, policy
from .docker_mgr import DockerManager
from .state import State, candidates_to_stop_for_idle

logging.basicConfig(
    level=os.environ.get("BROKER_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("broker")

# How long after start_worker() to wait for /healthz before giving up.
WORKER_HEALTH_TIMEOUT_S = 300
WORKER_HEALTH_POLL_S = 2
# Idle-shutdown sweep interval.
IDLE_SWEEP_INTERVAL_S = 5


# ---------- request/response models ----------

class AcquireRequest(BaseModel):
    # Deliberately NOT a Literal. Role names are defined in config/roles.yaml;
    # hardcoding them here meant a role added to config could never be acquired
    # (hit 2026-08-07 adding llm_small / llm_large -- the config parsed, the
    # broker started, and /acquire returned 422 against a stale literal that
    # still said "llm"). Validated against the loaded config in the handler,
    # matching how /admin/pin/{role} already does it.
    role: str = Field(min_length=1, max_length=64)
    client_id: str = Field(min_length=1, max_length=128)
    hint_vram_mb: int | None = None
    hold_seconds: int | None = Field(default=None, ge=0)


class AcquireResponse(BaseModel):
    handle: str
    endpoint: str
    role: str
    evicted: list[str] = Field(default_factory=list)
    warmed_implies: list[str] = Field(default_factory=list)
    cold_start_ms: int = 0


class InfeasibleResponse(BaseModel):
    reason: str
    free_mb: int
    would_need_evict: list[str] = Field(default_factory=list)


class ReleaseRequest(BaseModel):
    handle: str


class WorkerStatus(BaseModel):
    role: str
    container_id: str
    state: Literal["loaded"]
    loaded_mb_estimate: int
    last_request_at: float
    idle_shutdown_at: float | None
    cpu_rss_mb: int | None
    pinned: bool
    active_handles: int


class StatusResponse(BaseModel):
    broker_version: str
    gpu_used_mb: int
    gpu_total_mb: int
    budget_free_mb: int
    budget_total_mb: int
    workers: list[WorkerStatus]
    pinned_roles: list[str]
    active_handles: int


# ---------- app factory + lifespan ----------

def _build_state() -> State:
    cfg = config.load()
    docker_mgr = DockerManager(version=__version__)
    state = State(cfg=cfg, docker=docker_mgr)
    state.rediscover_from_docker()
    return state


async def _idle_sweep_loop(state: State) -> None:
    """Periodic task: stop workers whose idle timer has fired."""
    while True:
        try:
            await asyncio.sleep(IDLE_SWEEP_INTERVAL_S)
            async with state.lock:
                to_stop = candidates_to_stop_for_idle(state)
                for role in to_stop:
                    w = state.workers[role]
                    log.info("idle-shutdown role=%s container=%s", role, w.handle.container.short_id)
                    await asyncio.to_thread(state.docker.stop_worker, w.handle.container)
                    state.remove_worker(role)
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("idle sweep failed (continuing)")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state = _build_state()
    app.state.broker = state
    sweep_task = asyncio.create_task(_idle_sweep_loop(state))
    log.info("broker started version=%s budget=%d/%d MiB",
             __version__, state.cfg.gpu.broker_budget_mb, state.cfg.gpu.total_mb)
    try:
        yield
    finally:
        sweep_task.cancel()
        try:
            await sweep_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="gpu-broker",
    version=__version__,
    description="VRAM-aware Docker orchestrator for single-GPU model serving.",
    lifespan=_lifespan,
)


def _state() -> State:
    return app.state.broker


# ---------- worker readiness ----------

async def _wait_for_worker_health(endpoint: str, timeout_s: int,
                                  health_path: str = "/healthz") -> int:
    """Poll <endpoint><health_path> until 200 or timeout. Returns elapsed ms.

    health_path is per-role because it is a property of the WORKER IMAGE, not
    of the broker. The three encoder images implement /healthz; ollama does not
    (it answers "/" and "/api/tags"), so hardcoding /healthz made an ollama
    worker hang until timeout while the container was up and serving normally
    -- a readiness probe that can never pass looks exactly like a broken model.
    """
    start = time.monotonic()
    deadline = start + timeout_s
    path = health_path if health_path.startswith("/") else f"/{health_path}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{endpoint}{path}")
                if r.status_code == 200:
                    return int((time.monotonic() - start) * 1000)
            except httpx.HTTPError:
                # Deliberately broad. This loop already has its own deadline, so
                # ANY transport-level failure before that deadline is just "not
                # ready yet" and must be retried. The previous tuple
                # (ConnectError, ReadError, RemoteProtocolError) omitted
                # TimeoutException, so a worker that was merely SLOW to answer
                # -- rather than not listening -- aborted the entire acquire with
                # a 500. Hit 2026-08-07: ollama answered /healthz-equivalent in
                # microseconds normally, but went over the 3s client timeout
                # while a concurrent model pull saturated the same disk.
                pass
            await asyncio.sleep(WORKER_HEALTH_POLL_S)
    raise TimeoutError(
        f"worker at {endpoint} did not answer 200 on {path} in {timeout_s}s"
    )


# ---------- endpoints ----------

@app.get("/health")
async def broker_health():
    return {"status": "ok", "version": __version__}


@app.post("/acquire", response_model=AcquireResponse,
          responses={503: {"model": InfeasibleResponse}})
async def acquire(req: AcquireRequest):
    state = _state()
    if req.role not in state.cfg.roles:
        raise HTTPException(
            status_code=404,
            detail=f"unknown role: {req.role}. "
                   f"known: {sorted(state.cfg.roles)}",
        )
    if not state.cfg.roles[req.role].enabled:
        raise HTTPException(
            status_code=409,
            detail=f"role {req.role} is declared but disabled (enabled: false)",
        )
    async with state.lock:
        plan = policy.plan_acquire(state, req.role)
        if not plan.feasible:
            log.warning("acquire infeasible role=%s reason=%s", req.role, plan.infeasible_reason)
            raise HTTPException(
                status_code=503,
                detail=InfeasibleResponse(
                    reason=plan.infeasible_reason or "infeasible",
                    free_mb=state.vram_budget_free_mb(),
                    would_need_evict=plan.to_evict,
                ).model_dump(),
            )

        cold_start_total = 0

        # 1. Evictions
        for victim_role in plan.to_evict:
            victim = state.workers.get(victim_role)
            if victim is None:
                continue
            log.info("evicting role=%s for acquire of %s", victim_role, req.role)
            await asyncio.to_thread(state.docker.stop_worker, victim.handle.container)
            state.remove_worker(victim_role)

        # 2. Start target (or just touch)
        if plan.will_start:
            cfg = state.cfg.roles[req.role]
            start_t = time.monotonic()
            handle = await asyncio.to_thread(state.docker.start_worker, req.role, cfg)
            worker = state.add_worker(req.role, handle)
            wait_ms = await _wait_for_worker_health(
                worker.endpoint, WORKER_HEALTH_TIMEOUT_S,
                state.cfg.roles[req.role].health_path,
            )
            cold_start_total += int((time.monotonic() - start_t) * 1000)
        else:
            state.touch_worker(req.role)
            worker = state.workers[req.role]

        # 3. Implies warming (e.g. rerank -> embed)
        warmed: list[str] = []
        for implied_role in plan.to_touch_implies:
            state.touch_worker(implied_role)
        for implied_role in plan.to_warm_implies:
            icfg = state.cfg.roles[implied_role]
            ihandle = await asyncio.to_thread(state.docker.start_worker, implied_role, icfg)
            iworker = state.add_worker(implied_role, ihandle)
            await _wait_for_worker_health(
                iworker.endpoint, WORKER_HEALTH_TIMEOUT_S,
                state.cfg.roles[implied_role].health_path,
            )
            warmed.append(implied_role)

        # 4. Create client handle
        client_h = state.create_handle(req.role, req.client_id, req.hold_seconds)

        log.info("acquired role=%s client=%s handle=%s evicted=%s warmed=%s cold_ms=%d",
                 req.role, req.client_id, client_h.handle_id[:8],
                 plan.to_evict, warmed, cold_start_total)

        return AcquireResponse(
            handle=client_h.handle_id,
            endpoint=worker.endpoint,
            role=req.role,
            evicted=plan.to_evict,
            warmed_implies=warmed,
            cold_start_ms=cold_start_total,
        )


@app.post("/release", status_code=status.HTTP_204_NO_CONTENT)
async def release(req: ReleaseRequest):
    state = _state()
    async with state.lock:
        h = state.release_handle(req.handle)
        if h is None:
            raise HTTPException(status_code=404, detail="unknown handle")
        # Touch the worker so its idle clock starts now (not at next /acquire).
        state.touch_worker(h.role)
        log.info("released handle=%s role=%s", h.handle_id[:8], h.role)
    return None


@app.get("/status", response_model=StatusResponse)
async def get_status():
    state = _state()
    workers_out: list[WorkerStatus] = []
    async with state.lock:
        for role, w in state.workers.items():
            rss = observability.container_cpu_rss_mb(w.handle.container_pid) if w.handle.container_pid else None
            workers_out.append(WorkerStatus(
                role=role,
                container_id=w.handle.container_id[:12],
                state="loaded",
                loaded_mb_estimate=w.cfg.loaded_mb,
                last_request_at=w.last_request_at,
                idle_shutdown_at=w.idle_shutdown_at,
                cpu_rss_mb=rss,
                pinned=role in state.pinned_roles,
                active_handles=len(state.active_handles_for_role(role)),
            ))
        return StatusResponse(
            broker_version=__version__,
            gpu_used_mb=observability.gpu_used_mb(),
            gpu_total_mb=state.cfg.gpu.total_mb,
            budget_free_mb=state.vram_budget_free_mb(),
            budget_total_mb=state.cfg.gpu.broker_budget_mb,
            workers=workers_out,
            pinned_roles=sorted(state.pinned_roles),
            active_handles=len(state.handles),
        )


@app.post("/admin/pin/{role}", status_code=status.HTTP_204_NO_CONTENT)
async def pin_role(role: str):
    state = _state()
    if role not in state.cfg.roles:
        raise HTTPException(status_code=404, detail=f"unknown role: {role}")
    async with state.lock:
        state.pin(role)
        log.info("pinned role=%s", role)
    return None


@app.post("/admin/unpin/{role}", status_code=status.HTTP_204_NO_CONTENT)
async def unpin_role(role: str):
    state = _state()
    if role not in state.cfg.roles:
        raise HTTPException(status_code=404, detail=f"unknown role: {role}")
    async with state.lock:
        state.unpin(role)
        log.info("unpinned role=%s", role)
    return None
