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
    def _nvidia_inventory() -> list[tuple[str, str]]:
        """[(uuid, name), ...] from nvidia-smi. Empty list if it can't be read."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=True,
            ).stdout
        except Exception as exc:  # noqa: BLE001 - any failure means "unknown"
            log.warning("nvidia-smi inventory failed (%s); cannot pin by name", exc)
            return []
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) == 2 and parts[0]:
                rows.append((parts[0], parts[1]))
        return rows

    def _device_requests_for(self, role: str, cfg: RoleConfig):
        """Build the DeviceRequest list, honouring `gpu_selector` if set.

        `count=-1` (all GPUs) is correct ONLY while exactly one card is
        installed. With two cards it hands every worker every GPU and lets CUDA
        pick by index -- which is how a 21 GB model ends up on an 8 GB card, or
        an encoder silently lands on the wrong card and invalidates a thermal
        test. Selecting by UUID is immune to slot moves and enumeration order.
        """
        sel = cfg.gpu_selector
        if not sel:
            return [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

        inventory = self._nvidia_inventory()
        if not inventory:
            raise RuntimeError(
                f"role={role} pins gpu_selector={sel!r} but the GPU inventory "
                f"could not be read. Refusing to fall back to all-GPUs: that is "
                f"how work lands on the wrong card silently."
            )

        # exact UUID first, then case-insensitive name substring
        matches = [u for u, _ in inventory if u == sel]
        if not matches:
            matches = [u for u, n in inventory if sel.lower() in n.lower()]

        if len(matches) != 1:
            avail = ", ".join(f"{n} ({u})" for u, n in inventory)
            raise RuntimeError(
                f"role={role} gpu_selector={sel!r} matched {len(matches)} GPUs; "
                f"need exactly 1. Installed: {avail}"
            )

        log.info("role=%s pinned to GPU %s (selector=%r)", role, matches[0], sel)
        return [docker.types.DeviceRequest(
            device_ids=[matches[0]], capabilities=[["gpu"]]
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
            ports={f"{cfg.port}/tcp": cfg.port},
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
