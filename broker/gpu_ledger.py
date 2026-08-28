"""Per-GPU VRAM ledger — what cards are installed, and what each one's budget is.

Replaces the single global `gpu.total_mb` scalar (added 2026-08-28). That scalar
was a hand-maintained number that nobody reconciled against the hardware: on
2026-08-27 it still read 24576 (an RTX 3090 that had been swapped out) while the
installed card was a 16311 MiB RTX 5060 Ti. Nothing detected it, and `/status`
actively laundered it by reporting `gpu_total_mb` from config while reporting
`gpu_used_mb` from nvidia-smi in the adjacent field.

The split this module enforces:

  * MEASURED values come from the card   — `total_mb`, uuid, name
  * POLICY values come from config       — `reserved_for_host_mb`

`total_mb` is deliberately NOT configurable. It is a property of the silicon; a
config file cannot have an opinion about it, and letting it have one is exactly
how the drift above happened.

`reserved_for_host_mb` stays in config because it cannot be measured: it is the
driver's own reserve PLUS fragmentation headroom, and the headroom half is a
judgement call. Measured driver reserve (total - used - free at idle, which the
driver never reports as "used"):

    RTX 3090     451 MiB   (measured 2026-08-27)
    RTX 5060 Ti  462 MiB   (measured 2026-08-27)

Near-constant across cards, so the same ~1024 policy carries: ~460 driver +
~560 fragmentation headroom. Headroom is absolute, not proportional — it does
not shrink just because the card is smaller.

Cards are keyed exactly like `RoleConfig.gpu_selector`: a full UUID, or a
case-insensitive substring of the card NAME. Never an index — nvidia-smi and
CUDA disagree about what "0" means once a second card is present, and PCI paths
renumber on a slot move. The matching helper here is the single implementation;
`docker_mgr` imports it so pinning and budgeting can never diverge.

A ledger entry whose card is not installed is simply inactive — that is how the
3090 entry sits dormant while the 5060 Ti is in the box, and how swapping the
cards back needs no config edit at all.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuInfo:
    """A physically installed GPU, as reported by nvidia-smi."""
    uuid: str
    name: str
    total_mb: int


@dataclass(frozen=True)
class GpuBudget:
    """An installed GPU matched to its ledger entry."""
    uuid: str
    name: str
    total_mb: int           # measured, from the card
    reserved_for_host_mb: int   # policy, from config
    ledger_key: str         # which config key matched, for diagnostics

    @property
    def budget_mb(self) -> int:
        return self.total_mb - self.reserved_for_host_mb


class GpuLedgerError(RuntimeError):
    """Raised when the installed hardware and the ledger cannot be reconciled."""


def nvidia_inventory() -> list[GpuInfo]:
    """Installed GPUs. Empty list if nvidia-smi cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001 — any failure means "unknown"
        log.warning("nvidia-smi inventory failed (%s)", exc)
        return []

    gpus: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or not parts[0]:
            continue
        try:
            gpus.append(GpuInfo(uuid=parts[0], name=parts[1], total_mb=int(parts[2])))
        except ValueError:
            log.warning("unparseable nvidia-smi row: %r", line)
    return gpus


def match_selector(selector: str, inventory: list[GpuInfo]) -> list[str]:
    """UUIDs matching `selector`: exact UUID first, then name substring.

    Shared by `gpu_selector` pinning and by ledger resolution so the two can
    never disagree about which card a key refers to.
    """
    exact = [g.uuid for g in inventory if g.uuid == selector]
    if exact:
        return exact
    return [g.uuid for g in inventory if selector.lower() in g.name.lower()]


def resolve(cards: dict, inventory: list[GpuInfo]) -> dict[str, GpuBudget]:
    """Match installed GPUs to ledger entries → {uuid: GpuBudget}.

    Refuses rather than guesses, matching this codebase's existing posture on
    GPU selection: an unbudgeted card is a config decision, not a default.
    """
    if not inventory:
        raise GpuLedgerError(
            "no GPUs reported by nvidia-smi; refusing to start. The broker "
            "cannot budget VRAM it cannot see."
        )

    budgets: dict[str, GpuBudget] = {}
    for key, card_cfg in cards.items():
        matches = match_selector(key, inventory)
        if not matches:
            log.info("ledger entry %r matches no installed GPU — inactive", key)
            continue
        if len(matches) > 1:
            raise GpuLedgerError(
                f"ledger key {key!r} matched {len(matches)} installed GPUs; "
                f"need at most 1. Use a full UUID to disambiguate. "
                f"Installed: {_fmt(inventory)}"
            )
        gpu = next(g for g in inventory if g.uuid == matches[0])
        if gpu.uuid in budgets:
            raise GpuLedgerError(
                f"GPU {gpu.name} ({gpu.uuid}) matched by two ledger keys: "
                f"{budgets[gpu.uuid].ledger_key!r} and {key!r}."
            )
        if card_cfg.reserved_for_host_mb >= gpu.total_mb:
            raise GpuLedgerError(
                f"ledger key {key!r}: reserved_for_host_mb="
                f"{card_cfg.reserved_for_host_mb} >= card total {gpu.total_mb} MiB."
            )
        budgets[gpu.uuid] = GpuBudget(
            uuid=gpu.uuid, name=gpu.name, total_mb=gpu.total_mb,
            reserved_for_host_mb=card_cfg.reserved_for_host_mb, ledger_key=key,
        )

    unbudgeted = [g for g in inventory if g.uuid not in budgets]
    if unbudgeted:
        raise GpuLedgerError(
            f"installed GPU(s) with no ledger entry: {_fmt(unbudgeted)}. "
            f"Add an entry under `gpu.cards` (keyed by a name substring or the "
            f"full UUID) with its reserved_for_host_mb. Refusing to start: a "
            f"card the broker cannot budget is a card it will over-commit."
        )
    return budgets


def _fmt(gpus: list[GpuInfo]) -> str:
    return ", ".join(f"{g.name} {g.total_mb}MiB ({g.uuid})" for g in gpus)


def resolve_placement(selector: str | None, inventory: list[GpuInfo],
                      role: str = "?") -> str:
    """Which GPU UUID a role runs on. THE single placement decision.

    Both the container pin (docker_mgr) and the VRAM charge (state) call this,
    so a role can never be budgeted against one card and placed on another.

    Sole-card fallback (added 2026-08-28 by request): a selector that matches
    nothing falls back to the only installed card. This is NOT the dangerous
    fallback the codebase already refuses -- that one was "pin failed, so use
    ALL GPUs" on a multi-card box, where CUDA picks by index and a 21 GB model
    lands on a 16 GB card. With exactly one card installed there is nothing to
    pick wrongly: one card is the only place work can go, and refusing would
    just make a single-GPU machine unserviceable because the config names the
    card it does not currently have.

    It stays SAFE because it is paired with the per-GPU budget: a role that
    falls back onto a card too small for it is still rejected by
    plan_acquire()'s feasibility check. Fallback decides WHERE, the ledger
    decides WHETHER. Both must agree.

    Always logged at WARNING — a fallback is a config/hardware mismatch that
    happens to be recoverable, not a normal state.
    """
    if not selector:
        if len(inventory) == 1:
            return inventory[0].uuid
        raise GpuLedgerError(
            f"role={role} has no gpu_selector but {len(inventory)} GPUs are "
            f"installed. Pin it with `gpu_selector` — an unpinned role on a "
            f"multi-GPU box lets CUDA pick by index, which is how a 21 GB "
            f"model ends up on a 16 GB card. Installed: {_fmt(inventory)}"
        )

    matches = match_selector(selector, inventory)
    if len(matches) == 1:
        return matches[0]

    if not matches and len(inventory) == 1:
        only = inventory[0]
        log.warning(
            "role=%s pins gpu_selector=%r which is NOT installed; falling back "
            "to the only card present: %s (%s). Safe because it is the sole "
            "placement option — the per-GPU budget still decides whether the "
            "role actually fits.",
            role, selector, only.name, only.uuid,
        )
        return only.uuid

    raise GpuLedgerError(
        f"role={role} gpu_selector={selector!r} matched {len(matches)} of "
        f"{len(inventory)} installed GPUs; need exactly 1, and sole-card "
        f"fallback does not apply with {len(inventory)} cards present. "
        f"Installed: {_fmt(inventory)}"
    )
