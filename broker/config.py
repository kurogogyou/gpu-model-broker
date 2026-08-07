"""Role config loader. Parses config/roles.yaml into typed Pydantic models.

Source of truth for lifecycle policy. The Docker-SDK call shapes (image, port,
env, volumes) are duplicated minimally in docker-compose.yml for declarative
visibility and `docker compose pull`, but this file is what the broker reads
at startup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

# NOTE: declarative only -- nothing validates against this (BrokerConfig.roles
# is keyed by plain str). Kept in sync by hand so it does not mislead; the
# single `llm` role became `llm_small` / `llm_large` on 2026-08-07.
RoleName = Literal["embed", "rerank", "transcribe", "llm_small", "llm_large"]


class RoleConfig(BaseModel):
    image: str
    port: int
    loaded_mb: int = Field(ge=0)
    activation_mb_estimate: int = Field(ge=0)
    idle_shutdown_seconds: int | None = Field(default=300, ge=0)
    co_resident_with: list[str] = Field(default_factory=list)
    implies: list[str] = Field(default_factory=list)
    evict_on_acquire: list[str] = Field(default_factory=list)
    default_extra_args: list[str] = Field(default_factory=list)
    enabled: bool = True

    # --- multi-GPU + generic-worker support (added 2026-08-07) --------------
    # gpu_selector: which physical GPU this role's container may see. Accepts a
    # full UUID ("GPU-4c536cab-...") or a case-insensitive substring of the card
    # NAME ("3090", "3060"); resolved to a UUID at container-start time.
    # NEVER an index -- nvidia-smi and CUDA disagree about what "0" means once a
    # second card is present, and PCI paths renumber on a slot move.
    # None = every GPU visible, which is the pre-multi-GPU behaviour and is only
    # safe while exactly one card is installed.
    gpu_selector: str | None = None

    # Extra bind mounts, host_path -> container_path. Previously every mount was
    # hardcoded per-role inside docker_mgr; roles needing their own storage (the
    # LLM roles need a model store) declare it here instead.
    volumes: dict[str, str] = Field(default_factory=dict)

    # Extra environment for the worker container, merged over the broker's own.
    env: dict[str, str] = Field(default_factory=dict)

    # Readiness probe path. A property of the WORKER IMAGE, not of the broker.
    # The three encoder images serve /healthz; ollama does not (it answers "/"),
    # and a probe that can never pass is indistinguishable from a broken model.
    health_path: str = "/healthz"


class GpuConfig(BaseModel):
    total_mb: int = Field(ge=1)
    reserved_for_host_mb: int = Field(ge=0)

    @property
    def broker_budget_mb(self) -> int:
        return self.total_mb - self.reserved_for_host_mb


class BrokerConfig(BaseModel):
    roles: dict[str, RoleConfig]
    gpu: GpuConfig

    @field_validator("roles")
    @classmethod
    def _validate_cross_refs(cls, v: dict[str, RoleConfig]) -> dict[str, RoleConfig]:
        role_names = set(v.keys())
        for name, role in v.items():
            bad = set(role.co_resident_with) - role_names
            bad |= set(role.implies) - role_names
            bad |= set(role.evict_on_acquire) - role_names
            if bad:
                raise ValueError(f"Role '{name}' references unknown roles: {bad}")
        return v

    def enabled_roles(self) -> dict[str, RoleConfig]:
        return {n: r for n, r in self.roles.items() if r.enabled}


def load(path: Path | str | None = None) -> BrokerConfig:
    """Load and validate broker config. Defaults to config/roles.yaml at repo root.

    Override path for tests via the BROKER_ROLES_YAML env var or explicit arg.
    """
    if path is None:
        # Default: repo_root/config/roles.yaml (broker/ is one level below repo_root)
        path = Path(__file__).resolve().parent.parent / "config" / "roles.yaml"
    raw = yaml.safe_load(Path(path).read_text())
    return BrokerConfig(**raw)
