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

RoleName = Literal["embed", "rerank", "transcribe", "llm"]


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
