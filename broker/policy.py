"""Policy decisions: eviction, warming, feasibility.

The state machine (state.py) is the data; this module is the rules. Both are
called only under State.lock.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .state import State

log = logging.getLogger(__name__)


@dataclass
class PolicyPlan:
    """Result of plan_acquire(). Contains the sequence of operations the
    state-machine layer should execute, in order."""
    role: str
    to_evict: list[str] = field(default_factory=list)   # roles to stop before starting target
    to_warm_implies: list[str] = field(default_factory=list)  # extra roles to start (from role.implies)
    to_touch_implies: list[str] = field(default_factory=list)  # implied roles already loaded — just reset idle timer
    will_start: bool = True   # False if target role is already loaded (just touch)
    feasible: bool = True
    infeasible_reason: str | None = None
    free_mb_after_evict: int | None = None  # what budget will look like once evictions land


def plan_acquire(state: State, role: str) -> PolicyPlan:
    """Decide what to do to satisfy an /acquire {role}. Pure function over State.

    Order of work, encoded into the returned plan:
      1. evict_on_acquire roles get stopped (skip if pinned — pinned wins).
      2. If the target role isn't already loaded, also evict idle non-pinned
         workers as needed to make space for it (LRU by last_request_at).
      3. Start the target (will_start=True) OR touch it (will_start=False).
      4. Start any role in role.implies that isn't already loaded; treat
         their idle-timer / co_resident_with implicitly via add_worker.
    """
    cfg = state.cfg.roles.get(role)
    if cfg is None:
        return PolicyPlan(role=role, feasible=False,
                          infeasible_reason=f"unknown role: {role}")
    if not cfg.enabled:
        return PolicyPlan(role=role, feasible=False,
                          infeasible_reason=f"role disabled in roles.yaml: {role}")

    plan = PolicyPlan(role=role)

    # If already loaded, no new VRAM needed; just touch + handle implies.
    if state.is_loaded(role):
        plan.will_start = False
        plan.to_warm_implies, plan.to_touch_implies = _split_implies(state, cfg.implies)
        return plan

    # Hard evictions per role config (evict_on_acquire). Pin protection applies.
    plan.to_evict.extend(_filter_evictable(state, cfg.evict_on_acquire))

    # Compute free budget after the hard evictions.
    freed = sum(state.cfg.roles[r].loaded_mb for r in plan.to_evict)
    free_after_evict = state.vram_budget_free_mb() + freed

    needed = cfg.loaded_mb
    if free_after_evict >= needed:
        plan.free_mb_after_evict = free_after_evict
        plan.to_warm_implies, plan.to_touch_implies = _split_implies(state, cfg.implies)
        return plan

    # Need more — try LRU eviction of other non-pinned, no-active-holds workers.
    extra = _lru_eviction_candidates(state, exclude=set(plan.to_evict) | {role})
    for victim in extra:
        if free_after_evict >= needed:
            break
        plan.to_evict.append(victim)
        free_after_evict += state.cfg.roles[victim].loaded_mb

    if free_after_evict < needed:
        plan.feasible = False
        plan.infeasible_reason = (
            f"would need {needed - free_after_evict} MiB more even after "
            f"evicting {plan.to_evict or 'nothing evictable'}"
        )
        plan.free_mb_after_evict = free_after_evict
        return plan

    plan.free_mb_after_evict = free_after_evict
    plan.to_warm_implies, plan.to_touch_implies = _split_implies(state, cfg.implies)
    return plan


def _filter_evictable(state: State, roles: list[str]) -> list[str]:
    """Drop any pinned roles (or roles that aren't currently loaded) from the
    eviction list. Active-hold roles are still evicted (this is the hard policy
    rule — evict_on_acquire is contractually mandatory)."""
    out: list[str] = []
    for r in roles:
        if r not in state.workers:
            continue
        if r in state.pinned_roles:
            log.info("skipping eviction of pinned role: %s", r)
            continue
        out.append(r)
    return out


def _lru_eviction_candidates(state: State, exclude: set[str]) -> list[str]:
    """Roles sorted by last_request_at ascending (oldest first), excluding
    the given set, pinned roles, and roles with active client holds."""
    candidates = [
        w for w in state.workers.values()
        if w.role not in exclude
        and w.role not in state.pinned_roles
        and not state.role_has_active_holds(w.role)
    ]
    candidates.sort(key=lambda w: w.last_request_at)
    return [w.role for w in candidates]


def _split_implies(state: State, implies: list[str]) -> tuple[list[str], list[str]]:
    """Partition role.implies into (need-to-start, need-to-touch-only).
    Already-loaded ones get their idle timer reset by the caller (state.touch_worker);
    not-loaded ones need a fresh start_worker."""
    to_start: list[str] = []
    to_touch: list[str] = []
    for r in implies:
        if state.is_loaded(r):
            to_touch.append(r)
        else:
            to_start.append(r)
    return to_start, to_touch
