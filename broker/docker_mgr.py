"""Docker SDK wrapper. The broker calls these to start/stop workers; no
shelling out to `docker run`.

Container naming + labels follow docker-compose.yml conventions so a `docker ps`
shows the same names the broker would set, and the broker can re-discover live
workers across restarts.
"""
from __future__ import annotations

import logging
import os
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
        if role == "transcribe" and self._hf_token:
            env["HF_TOKEN"] = self._hf_token

        volumes = {}
        if role == "transcribe":
            volumes["/home/mario"] = {"bind": "/home/mario", "mode": "ro"}

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
            device_requests=[
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]]),
            ],
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
