"""Docker SDK wrapper. The broker calls these to start/stop workers; no
shelling out to `docker run`.

Container naming + labels follow docker-compose.yml conventions so a `docker ps`
shows the same names the broker would set, and the broker can re-discover live
workers across restarts.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass

from .gpu_ledger import nvidia_inventory, resolve_placement

# Interface worker container ports are published on. Loopback by default --
# see the note at the ports= kwarg in start_worker(). Override only once an
# authenticated path (reverse proxy / WireGuard) fronts the workers.
WORKER_BIND_HOST = os.environ.get("BROKER_WORKER_BIND_HOST", "127.0.0.1")

import docker
from docker.models.containers import Container

from .config import RoleConfig

log = logging.getLogger(__name__)

LABEL_ROLE = "gpu-broker.role"
LABEL_VERSION = "gpu-broker.version"
LABEL_HANDLE = "gpu-broker.handle"   # set when an /acquire is active; cleared on /release
CONTAINER_NAME_FMT = "gpu-broker-{role}"


@dataclass
class WorkerHandle:
    container: Container
    role: str
    container_pid: int  # main process PID on the host

    @property
    def container_id(self) -> str:
        return self.container.id

    @property
    def name(self) -> str:
        return self.container.name


class DockerManager:
    """Thin wrapper around the docker.from_env client. Single-host, single-GPU.

    NOT thread-safe on its own — callers (broker state machine) serialize
    start/stop transitions with an asyncio.Lock.
    """

    def __init__(self, version: str = "0.1.0", hf_token: str | None = None):
        self._client = docker.from_env()
        self._version = version
        self._hf_token = hf_token or os.environ.get("HF_TOKEN", "")

    # ---------- discovery ----------

    def list_managed_containers(self) -> list[Container]:
        """All containers (running or stopped) carrying the gpu-broker.role label."""
        return self._client.containers.list(
            all=True, filters={"label": LABEL_ROLE}
        )

    def find_for_role(self, role: str) -> Container | None:
        for c in self.list_managed_containers():
            if c.labels.get(LABEL_ROLE) == role:
                return c
        return None

    def is_running(self, container: Container) -> bool:
        container.reload()
        return container.status == "running"

    def main_pid(self, container: Container) -> int:
        container.reload()
        return container.attrs.get("State", {}).get("Pid", 0)

    # ---------- GPU affinity ----------

    @staticmethod
    def _nvidia_inventory():
        """Installed GPUs. Delegates to gpu_ledger so that pinning and VRAM
        budgeting resolve selectors through ONE implementation — if these two
        ever disagreed, a role would be charged to one card and placed on
        another."""
        return nvidia_inventory()

    def _device_requests_for(self, role: str, cfg: RoleConfig):
        """Build the DeviceRequest list, honouring `gpu_selector` if set.

        `count=-1` (all GPUs) is correct ONLY while exactly one card is
        installed. With two cards it hands every worker every GPU and lets CUDA
        pick by index -- which is how a 21 GB model ends up on an 8 GB card, or
        an encoder silently lands on the wrong card and invalidates a thermal
        test. Selecting by UUID is immune to slot moves and enumeration order.
        """
        sel = cfg.gpu_selector
        inventory = self._nvidia_inventory()
        if not inventory:
            raise RuntimeError(
                f"role={role} cannot be placed: the GPU inventory could not be "
                f"read. Refusing to fall back to all-GPUs -- that is how work "
                f"lands on the wrong card silently."
            )

        # Unpinned + exactly one card is the historical count=-1 case. Keep it:
        # it is proven, and handing the sole GPU to the container is identical
        # in effect to naming it.
        if not sel and len(inventory) == 1:
            return [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

        # Everything else -- pinned, or unpinned on a multi-GPU box -- goes
        # through the shared decision, which raises on genuine ambiguity and
        # falls back only when one card is the sole option.
        uuid_ = resolve_placement(sel, inventory, role=role)
        log.info("role=%s placed on GPU %s (selector=%r)", role, uuid_, sel)
        return [docker.types.DeviceRequest(
            device_ids=[uuid_], capabilities=[["gpu"]]
        )]

    # ---------- lifecycle ----------

    def start_worker(self, role: str, cfg: RoleConfig) -> WorkerHandle:
        """Start a container for `role`. If a matching managed container exists
        (running or stopped), reuse / restart it instead of double-creating.
        Container will live until explicitly stopped — never auto-removed.
        """
        existing = self.find_for_role(role)
        if existing is not None:
            existing.reload()
            if existing.status == "running":
                log.info("reusing running container for role=%s id=%s", role, existing.short_id)
                return WorkerHandle(
                    container=existing,
                    role=role,
                    container_pid=self.main_pid(existing),
                )
            log.info("removing stopped container for role=%s id=%s", role, existing.short_id)
            existing.remove(force=True)

        name = CONTAINER_NAME_FMT.format(role=role)

        env = {
            "WORKER_ROLE": role,
            "BROKER_VRAM_MB": str(cfg.loaded_mb + cfg.activation_mb_estimate),
        }
        if role == "transcribe":
            # PyTorch CUDA allocator fragments badly across back-to-back
            # transcribe calls (one session leaves cached allocations that
            # block the next session's encoder reservation even when absolute
            # usage is in budget). expandable_segments switches to a segment
            # allocator that handles fragmentation cleanly. Surfaced 2026-06-14
            # on Batch 03a — first session succeeded, second OOM'd at the
            # encode step despite the broker math fitting. PyTorch 2.1+.
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            if self._hf_token:
                env["HF_TOKEN"] = self._hf_token

        volumes = {}
        if role == "transcribe":
            # Same-host bind: the worker reads audio paths the caller passes
            # at face value, so the host tree must appear at the same path
            # inside the container. /home/mario covers ~/bigrepo, ~/Downloads,
            # etc. /opt/brain covers consumers that live under /opt/brain/src
            # (ai-transcriber post-Phase-4-Task-#4, future broker-aware tools)
            # and any audio they stage under their repo tree. /mnt/bigrepo
            # covers the canonicalized target of ~/bigrepo (the script does
            # readlink -f before POSTing, which resolves the symlink to its
            # /mnt/bigrepo/bigrepo backing). RO is safe — whisperx writes
            # nothing to its input path.
            volumes["/home/mario"] = {"bind": "/home/mario", "mode": "ro"}
            volumes["/opt/brain"] = {"bind": "/opt/brain", "mode": "ro"}
            volumes["/mnt/bigrepo"] = {"bind": "/mnt/bigrepo", "mode": "ro"}

        # Config-driven mounts + env (roles.yaml `volumes:` / `env:`), applied
        # after the hardcoded ones so a role can override if it ever needs to.
        for host_path, container_path in cfg.volumes.items():
            volumes[host_path] = {"bind": container_path, "mode": "rw"}
        env.update(cfg.env)

        device_requests = self._device_requests_for(role, cfg)

        log.info("starting container role=%s image=%s name=%s", role, cfg.image, name)
        container = self._client.containers.run(
            cfg.image,
            name=name,
            detach=True,
            remove=False,
            # Bind workers to LOOPBACK explicitly. docker-py's short form
            # ({"8081/tcp": 8081}) publishes on 0.0.0.0, which put every model
            # endpoint on the LAN unauthenticated -- and a caller reaching a
            # worker directly never goes through /acquire, so the VRAM ledger
            # silently stops describing reality. Found 2026-08-07. LAN access
            # is a deliberate feature to be built (auth at the network layer +
            # a reverse proxy), not a side effect of a default.
            ports={f"{cfg.port}/tcp": (WORKER_BIND_HOST, cfg.port)},
            environment=env,
            volumes=volumes,
            labels={
                LABEL_ROLE: role,
                LABEL_VERSION: self._version,
            },
            device_requests=device_requests,
        )

        # Wait briefly for the PID to appear (otherwise main_pid returns 0)
        for _ in range(20):
            time.sleep(0.1)
            container.reload()
            if container.attrs.get("State", {}).get("Pid", 0):
                break

        return WorkerHandle(
            container=container,
            role=role,
            container_pid=self.main_pid(container),
        )

    def stop_worker(self, container: Container, timeout: int = 10) -> None:
        try:
            container.stop(timeout=timeout)
        except docker.errors.NotFound:
            return
        log.info("stopped container id=%s", container.short_id)

    def remove_worker(self, container: Container) -> None:
        try:
            container.remove(force=True)
        except docker.errors.NotFound:
            return
        log.info("removed container id=%s", container.short_id)
