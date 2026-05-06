# SPDX-License-Identifier: MulanPSL-2.0
"""explore_rbnx atlas bridge — Capability + contract-typed MCP tools.

Tools are typed against the codegen Request/Response dataclasses for the
explore/srv/* contracts (Explore, GetExploreStatus, CancelExplore). The
JSON Schema each MCP tool advertises to the LLM is derived from those
classes via cap.mcp's introspection — no hand-written schemas.
"""
from __future__ import annotations

import logging
import time

from robonix_py import Capability

from .controller import ExploreController

logging.basicConfig(level=logging.INFO,
                    format="[explore] %(levelname)s %(message)s")
log = logging.getLogger("explore_rbnx")

cap = Capability(id="com.robonix.skill.explore", namespace="robonix/skill/explore")
ctrl: ExploreController | None = None

# Atlas-resolved inputs the skill consumes. Hard-fail if any required
# input isn't atlas-resolvable: the packaging-spec rule is "no hardcoded
# fallback for cross-package topics", and a skill that can't find its
# dependencies should not pretend to work.
REQUIRED_INPUTS = {
    "map_topic":     ("robonix/service/map/occupancy_grid", ""),
    "nav_navigate":  ("robonix/service/navigation/navigate", ""),
    "nav_status":    ("robonix/service/navigation/status", ""),
    "nav_cancel":    ("robonix/service/navigation/cancel", ""),
}


def resolve_inputs(deadline_s: float = 60.0) -> dict[str, str]:
    resolved: dict[str, str] = {}
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        for key, (cid, transport) in REQUIRED_INPUTS.items():
            if key in resolved:
                continue
            ep = cap.query(cid, transport=transport)
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


@cap.mcp("robonix/skill/explore/explore")
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


@cap.mcp("robonix/skill/explore/status")
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


@cap.mcp("robonix/skill/explore/cancel")
def cancel(req: CancelExplore_Request) -> CancelExplore_Response:
    """Abort the active exploration. Idempotent."""
    if ctrl is None:
        return CancelExplore_Response(ok=False, message="controller not initialized")
    ok, msg = ctrl.cancel(req.task_id or None)
    return CancelExplore_Response(ok=ok, message=msg)


# ── lifecycle ────────────────────────────────────────────────────────────────
@cap.on_init
def init(cfg):
    global ctrl
    inputs = resolve_inputs()
    log.info("dependencies resolved: %s", inputs)
    ctrl = ExploreController(
        map_topic=inputs["map_topic"],
        nav_navigate_endpoint=inputs["nav_navigate"],
        nav_status_endpoint=inputs["nav_status"],
        nav_cancel_endpoint=inputs["nav_cancel"],
    )
    ctrl.start_runtime()
    return cap.ready()


def main() -> int:
    cap.run()
    if ctrl is not None:
        ctrl.stop_runtime()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
