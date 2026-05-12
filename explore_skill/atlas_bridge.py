# SPDX-License-Identifier: MulanPSL-2.0
"""explore_rbnx atlas bridge — Capability + contract-typed MCP tools.

Tools are typed against the codegen Request/Response dataclasses for the
explore/srv/* contracts (Explore, GetExploreStatus, CancelExplore). The
JSON Schema each MCP tool advertises to the LLM is derived from those
classes via explore.mcp's introspection — no hand-written schemas.
"""
from __future__ import annotations

import logging
import time

from robonix_api import ATLAS, Skill, Ok, Err, Deferred

from .controller import ExploreController

logging.basicConfig(level=logging.INFO,
                    format="[explore] %(levelname)s %(message)s")
log = logging.getLogger("explore_rbnx")

explore_skill = Skill(id="explore", namespace="robonix/skill/explore")
ctrl: ExploreController | None = None

# Atlas-resolved inputs the skill consumes. Hard-fail if any required
# input isn't atlas-resolvable: the packaging-spec rule is "no hardcoded
# fallback for cross-package topics", and a skill that can't find its
# dependencies should not pretend to work.
REQUIRED_INPUTS = {
    # (contract_id, transport) — transport must be a concrete enum, not
    # "unspecified", so atlas can return a usable endpoint string. The
    # nav contracts are mode=rpc and simple_nav exposes them as MCP tools
    # (see examples/webots/services/simple_nav/package_manifest.yaml header).
    "map_topic":     ("robonix/service/map/occupancy_grid", "ros2"),
    "nav_navigate":  ("robonix/service/navigation/navigate", "mcp"),
    "nav_status":    ("robonix/service/navigation/status", "mcp"),
    "nav_cancel":    ("robonix/service/navigation/cancel", "mcp"),
}


def resolve_inputs(deadline_s: float = 60.0) -> dict[str, str]:
    resolved: dict[str, str] = {}
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        for key, (cid, transport) in REQUIRED_INPUTS.items():
            if key in resolved:
                continue
            try:
                cap_view = ATLAS.find_unique_capability(
                    contract_id=cid, transport=transport,
                )
                ch = explore_skill.connect_capability(cap_view, cid, transport)
            except Exception:  # noqa: BLE001
                continue
            ep = ch.endpoint
            ch.close()
            if ep:
                resolved[key] = ep
                log.info("resolved %s [%s] → %s", cid, transport, ep)
        if len(resolved) == len(REQUIRED_INPUTS):
            return resolved
        time.sleep(2.0)
    missing = [k for k in REQUIRED_INPUTS if k not in resolved]
    raise RuntimeError(
        f"explore skill cannot find dependencies on atlas: missing "
        f"{[REQUIRED_INPUTS[k][0] for k in missing]}. The skill needs a "
        f"running mapping service (occupancy_grid) and navigation service "
        f"(navigate/status/cancel) before it can start. There is "
        f"intentionally no hardcoded fallback — packaging-spec invariant #1."
    )


# ── MCP tools (typed against codegen Request/Response) ──────────────────────
from explore_mcp import (  # noqa: E402
    Explore_Request, Explore_Response,
    GetExploreStatus_Request, GetExploreStatus_Response,
    CancelExplore_Request, CancelExplore_Response,
)


@explore_skill.mcp("robonix/skill/explore/explore")
def explore(req: Explore_Request) -> Explore_Response:
    """Start an autonomous exploration task. Returns a task_id; poll
    status() to track."""
    if ctrl is None:
        return Explore_Response(accepted=False, task_id="", message="controller not initialized")
    try:
        handle = ctrl.start(area_hint=req.area_hint,
                            timeout_s=float(req.timeout_s),
                            max_speed_m_s=float(req.max_speed_m_s))
        return Explore_Response(accepted=True, task_id=handle.task_id,
                                message=handle.detail)
    except RuntimeError as e:
        return Explore_Response(accepted=False, task_id="", message=str(e))


@explore_skill.mcp("robonix/skill/explore/status")
def status(req: GetExploreStatus_Request) -> GetExploreStatus_Response:
    """Poll progress of a running exploration task. Empty task_id = most recent."""
    if ctrl is None:
        return GetExploreStatus_Response(
            known=False, state="idle", area_m2=0.0, frontiers_left=0,
            elapsed_s=0.0, eta_s=-1.0, detail="controller not initialized",
        )
    s = ctrl.status(req.task_id or None)
    if s is None:
        return GetExploreStatus_Response(
            known=False, state="idle", area_m2=0.0, frontiers_left=0,
            elapsed_s=0.0, eta_s=-1.0, detail="no task with that id",
        )
    return GetExploreStatus_Response(
        known=True,
        state=str(s.get("state", "unknown")),
        area_m2=float(s.get("area_m2", 0.0)),
        frontiers_left=int(s.get("frontiers_left", 0)),
        elapsed_s=float(s.get("elapsed_s", 0.0)),
        eta_s=float(s.get("eta_s", -1.0)),
        detail=str(s.get("detail", "")),
    )


@explore_skill.mcp("robonix/skill/explore/cancel")
def cancel(req: CancelExplore_Request) -> CancelExplore_Response:
    """Abort the active exploration. Idempotent."""
    if ctrl is None:
        return CancelExplore_Response(ok=False, message="controller not initialized")
    ok, msg = ctrl.cancel(req.task_id or None)
    return CancelExplore_Response(ok=ok, message=msg)


# ── lifecycle ────────────────────────────────────────────────────────────────
# Skills split init from activate: rbnx boot calls Driver(CMD_INIT) on
# every package and stops there for skills (state = INITIALIZED). The
# executor sends Driver(CMD_ACTIVATE) just-in-time on the first MCP
# call, which is when the skill actually allocates hot resources (ROS
# subs, frontier loop, …). See docs/cap-lifecycle.md for the full FSM.
@explore_skill.on_init
def init(cfg):
    """CMD_INIT: light. The state machine wants every cap to reach
    INITIALIZED at boot time even if its upstream peers are still warming
    up — so we deliberately don't query atlas for nav / map here. cfg is
    accepted for forward-compat (no manifest knobs declared yet)."""
    log.info("CMD_INIT ok")
    return Ok()


@explore_skill.on_activate
def activate():
    """CMD_ACTIVATE: heavy. Resolve the upstream contracts NOW (executor
    only sends CMD_ACTIVATE when there's actually a request to satisfy,
    by which point map / nav should be ACTIVE), then build the
    ExploreController and start the rclpy thread. Idempotent on re-entry."""
    global ctrl
    if ctrl is not None:
        log.info("CMD_ACTIVATE — already runnable, no-op")
        return Ok()
    inputs = resolve_inputs()
    log.info("dependencies resolved: %s", list(inputs.keys()))
    ctrl = ExploreController(
        map_topic=inputs["map_topic"],
        nav_navigate_endpoint=inputs["nav_navigate"],
        nav_status_endpoint=inputs["nav_status"],
        nav_cancel_endpoint=inputs["nav_cancel"],
    )
    ctrl.start_runtime()
    log.info("CMD_ACTIVATE ok — controller running")
    return Ok()


@explore_skill.on_deactivate
def deactivate():
    """CMD_DEACTIVATE: stop the rclpy thread + drop the controller. Safe
    to call repeatedly; the second call is a no-op. Executor's eviction
    policy fires this when the skill has been idle long enough."""
    global ctrl
    if ctrl is None:
        return Ok()
    try:
        ctrl.stop_runtime()
    finally:
        ctrl = None
    log.info("CMD_DEACTIVATE ok — controller stopped")
    return Ok()


def main() -> int:
    explore_skill.run()
    if ctrl is not None:
        ctrl.stop_runtime()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
