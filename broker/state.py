"""In-memory state machine for the broker.

Tracks:
- Which workers are loaded (per-role Worker objects + container handles)
- Which client handles are active (handle -> role + acquired_at + hold_until)
- Which roles are pinned (eviction-immune)
- Per-role idle timer (last-request timestamp, idle_shutdown_at)

State is rebuilt on broker startup by discovering managed containers via labels —
no SQLite WAL needed for v1. (The 2026-05-28 Phase 3 plan offered either; labels
are simpler and survive broker process restarts equally well.)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable

from .config import BrokerConfig, RoleConfig
from .docker_mgr import DockerManager, WorkerHandle
from .gpu_ledger import GpuBudget, GpuLedgerError, resolve_placement

log = logging.getLogger(__name__)


@dataclass
class Worker:
    role: str
    cfg: RoleConfig
    handle: WorkerHandle
    last_request_at: float
    idle_shutdown_at: float | None  # epoch seconds; None = never (always-on or pinned)
    # Which physical GPU this worker's VRAM is charged to. Set at add_worker()
    # from the same resolution docker_mgr uses to pin the container, so the
    # ledger charge and the actual placement cannot disagree.
    gpu_uuid: str | None = None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.cfg.port}"


@dataclass
class ClientHandle:
    handle_id: str
    role: str
    client_id: str
    acquired_at: float
    hold_until: float | None  # epoch seconds; None = until /release


@dataclass
class State:
    cfg: BrokerConfig
    docker: DockerManager
    # uuid -> GpuBudget, resolved from nvidia-smi + cfg.gpu.cards at startup.
    gpu_budgets: dict[str, GpuBudget] = field(default_factory=dict)
    # role -> Worker (only loaded workers; stopped ones aren't tracked here)
    workers: dict[str, Worker] = field(default_factory=dict)
    # handle_id -> ClientHandle
    handles: dict[str, ClientHandle] = field(default_factory=dict)
    pinned_roles: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ---------- introspection ----------

    def is_loaded(self, role: str) -> bool:
        return role in self.workers

    def active_handles_for_role(self, role: str) -> list[ClientHandle]:
        return [h for h in self.handles.values() if h.role == role]

    def role_has_active_holds(self, role: str) -> bool:
        return any(h.role == role for h in self.handles.values())

    def total_loaded_vram_mb(self, gpu_uuid: str | None = None) -> int:
        """loaded_mb of resident workers, optionally scoped to one GPU."""
        return sum(w.cfg.loaded_mb for w in self.workers.values()
                   if gpu_uuid is None or w.gpu_uuid == gpu_uuid)

    def gpu_for_role(self, role: str) -> str:
        """Which GPU `role` is charged against.

        Delegates to gpu_ledger.resolve_placement — the SAME call docker_mgr
        makes to pin the container — so the VRAM charge and the physical
        placement are one decision, not two that happen to agree.
        """
        return resolve_placement(
            self.cfg.roles[role].gpu_selector,
            [self._as_info(u) for u in self.gpu_budgets],
            role=role,
        )

    def _as_info(self, uuid_: str):
        from .gpu_ledger import GpuInfo
        b = self.gpu_budgets[uuid_]
        return GpuInfo(uuid=b.uuid, name=b.name, total_mb=b.total_mb)

    def vram_budget_free_mb(self, gpu_uuid: str | None = None) -> int:
        """Budget-side free VRAM on one GPU (or summed across all, for /status).

        Distinct from gpu_used_mb() which reflects the actual nvidia-smi reading
        (includes host overhead + activation peaks).

        The summed form is a REPORTING convenience only — never schedule against
        it. Free VRAM does not pool across cards: 8 GiB free on each of two cards
        does not fit a 12 GiB model.
        """
        if gpu_uuid is not None:
            b = self.gpu_budgets.get(gpu_uuid)
            if b is None:
                raise GpuLedgerError(f"no ledger entry for GPU {gpu_uuid}")
            return b.budget_mb - self.total_loaded_vram_mb(gpu_uuid)
        return sum(b.budget_mb - self.total_loaded_vram_mb(u)
                   for u, b in self.gpu_budgets.items())

    def budget_total_mb(self) -> int:
        """Sum of per-GPU budgets. Reporting only — see vram_budget_free_mb."""
        return sum(b.budget_mb for b in self.gpu_budgets.values())

    def gpu_total_mb(self) -> int:
        """Sum of MEASURED card totals. Reporting only."""
        return sum(b.total_mb for b in self.gpu_budgets.values())

    # ---------- mutations (must be called under self.lock) ----------

    def add_worker(self, role: str, handle: WorkerHandle) -> Worker:
        cfg = self.cfg.roles[role]
        now = time.time()
        shutdown_at = (
            now + cfg.idle_shutdown_seconds
            if cfg.idle_shutdown_seconds and role not in self.pinned_roles
            else None
        )
        w = Worker(role=role, cfg=cfg, handle=handle, last_request_at=now,
                   idle_shutdown_at=shutdown_at, gpu_uuid=self.gpu_for_role(role))
        self.workers[role] = w
        return w

    def remove_worker(self, role: str) -> Worker | None:
        return self.workers.pop(role, None)

    def touch_worker(self, role: str) -> None:
        """Reset idle timer on the named worker (called on /acquire reuse)."""
        w = self.workers.get(role)
        if w is None:
            return
        w.last_request_at = time.time()
        if w.cfg.idle_shutdown_seconds and role not in self.pinned_roles:
            w.idle_shutdown_at = w.last_request_at + w.cfg.idle_shutdown_seconds
        else:
            w.idle_shutdown_at = None

    def create_handle(self, role: str, client_id: str, hold_seconds: int | None) -> ClientHandle:
        h = ClientHandle(
            handle_id=str(uuid.uuid4()),
            role=role,
            client_id=client_id,
            acquired_at=time.time(),
            hold_until=(time.time() + hold_seconds) if hold_seconds else None,
        )
        self.handles[h.handle_id] = h
        return h

    def release_handle(self, handle_id: str) -> ClientHandle | None:
        return self.handles.pop(handle_id, None)

    def pin(self, role: str) -> None:
        self.pinned_roles.add(role)
        # disable idle shutdown for pinned roles
        w = self.workers.get(role)
        if w:
            w.idle_shutdown_at = None

    def unpin(self, role: str) -> None:
        self.pinned_roles.discard(role)
        # re-arm idle timer if the worker is loaded
        w = self.workers.get(role)
        if w and w.cfg.idle_shutdown_seconds:
            w.idle_shutdown_at = time.time() + w.cfg.idle_shutdown_seconds

    # ---------- discovery on restart ----------

    def rediscover_from_docker(self) -> None:
        """Walk `docker ps -a -f label=gpu-broker.role` and rebuild self.workers
        for any RUNNING managed containers. Stopped containers are left alone
        (the broker will reuse or remove them on first /acquire for that role)."""
        for container in self.docker.list_managed_containers():
            role = container.labels.get("gpu-broker.role")
            if not role or role not in self.cfg.roles:
                continue
            container.reload()
            if container.status != "running":
                continue
            cfg = self.cfg.roles[role]
            pid = self.docker.main_pid(container)
            handle = WorkerHandle(container=container, role=role, container_pid=pid)
            self.add_worker(role, handle)
            log.info("rediscovered running worker role=%s container=%s pid=%s",
                     role, container.short_id, pid)


def candidates_to_stop_for_idle(state: State) -> list[str]:
    """Roles whose idle timer has fired AND have no active client handles."""
    now = time.time()
    out: list[str] = []
    for role, worker in state.workers.items():
        if worker.idle_shutdown_at is None:
            continue
        if now < worker.idle_shutdown_at:
            continue
        if state.role_has_active_holds(role):
            continue
        if role in state.pinned_roles:
            continue
        out.append(role)
    return out
