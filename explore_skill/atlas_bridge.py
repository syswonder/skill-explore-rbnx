# SPDX-License-Identifier: MulanPSL-2.0
"""explore_rbnx atlas bridge — registers + serves the explore skill.

Lifecycle:
  1. RegisterCapability (com.robonix.skill.explore).
  2. Resolve our two consumer dependencies via atlas:
       - service/map/occupancy_grid (subscribe, read frontiers off it)
       - service/navigation/navigate, /status, /cancel (call as RPC)
     Hard-fail if any required input isn't atlas-resolvable; the
     packaging-spec rule is "no hardcoded fallback for cross-package
     topics", and a skill that can't find its dependencies should not
     pretend to work.
  3. Spin up Controller (rclpy thread, frontier loop) + a FastMCP
     server exposing explore / status / cancel tools.
  4. DeclareInterface(MCP) for those three so pilot/executor can
     discover and call them.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from typing import Optional

logging.basicConfig(level=logging.INFO,
                    format="[explore] %(levelname)s %(message)s")
log = logging.getLogger("explore_rbnx")


# ── Generated proto stubs ─────────────────────────────────────────────
# entrypoint.sh sets PYTHONPATH=/explore:/explore/proto_gen.
import grpc  # noqa: E402

import atlas_pb2 as pb  # type: ignore
import atlas_pb2_grpc as pb_grpc  # type: ignore

from .controller import ExploreController, TaskHandle


# ── Config ────────────────────────────────────────────────────────────
ATLAS_ENDPOINT = os.environ.get("ROBONIX_ATLAS", "127.0.0.1:50051")
CAP_ID = os.environ.get("ROBONIX_CAPABILITY_ID", "com.robonix.skill.explore")
NAMESPACE = "robonix/skill/explore"
MCP_PORT = int(os.environ.get("EXPLORE_MCP_PORT", "50130"))
HEARTBEAT_PERIOD_S = 10.0

# Required atlas-resolved inputs. Keys are local names; values are
# (contract_id, transport). We declare transport explicitly because
# the same contract id can in principle be backed by either MCP or
# gRPC — the consumer asks for the transport it knows how to call.
# simple_nav currently exposes navigation/* as MCP (LLM-callable);
# we connect as MCP and call via HTTP. None of these have hardcoded
# fallbacks — packaging-spec invariant #1.
REQUIRED_INPUTS = {
    "map_topic":     ("robonix/service/map/occupancy_grid", "ros2"),
    "nav_navigate":  ("robonix/service/navigation/navigate", "mcp"),
    "nav_status":    ("robonix/service/navigation/status",   "mcp"),
    "nav_cancel":    ("robonix/service/navigation/cancel",   "mcp"),
}


# ── Atlas client helpers ──────────────────────────────────────────────
def _atlas() -> pb_grpc.AtlasStub:
    return pb_grpc.AtlasStub(grpc.insecure_channel(ATLAS_ENDPOINT))


def _connect(stub, contract_id: str, transport) -> Optional[str]:
    """QueryCapabilities + ConnectCapability for a contract over the
    given transport. Returns the producer's endpoint or None."""
    try:
        resp = stub.QueryCapabilities(pb.QueryCapabilitiesRequest(
            contract_id=contract_id, transport=transport))
    except grpc.RpcError as e:
        log.warning("query %s failed: %s", contract_id, e)
        return None
    for rec in resp.records:
        for iface in rec.interfaces:
            if iface.contract_id != contract_id or iface.transport != transport:
                continue
            try:
                conn = stub.ConnectCapability(pb.ConnectCapabilityRequest(
                    consumer_id=CAP_ID,
                    capability_id=rec.capability_id,
                    contract_id=contract_id,
                    transport=transport,
                ))
                if conn.endpoint:
                    return conn.endpoint
            except grpc.RpcError as e:
                log.warning("connect %s failed: %s", contract_id, e)
    return None


_TRANSPORT_MAP = {
    "ros2": pb.TRANSPORT_ROS2,
    "mcp":  pb.TRANSPORT_MCP,
    "grpc": pb.TRANSPORT_GRPC,
}


def _resolve_inputs(stub, deadline_s: float = 60.0) -> dict[str, str]:
    """Wait up to deadline for every REQUIRED_INPUTS contract to be
    discoverable on its declared transport. Returns key → endpoint;
    raises if any are missing at deadline."""
    resolved: dict[str, str] = {}
    deadline = time.time() + deadline_s

    while time.time() < deadline:
        for key, (cid, transport_name) in REQUIRED_INPUTS.items():
            if key in resolved:
                continue
            ep = _connect(stub, cid, _TRANSPORT_MAP[transport_name])
            if ep:
                resolved[key] = ep
                log.info("resolved %s [%s] → %s", cid, transport_name, ep)
        if len(resolved) == len(REQUIRED_INPUTS):
            return resolved
        time.sleep(2.0)

    missing = [k for k in REQUIRED_INPUTS if k not in resolved]
    raise RuntimeError(
        f"explore skill cannot find its dependencies on atlas: missing "
        f"{[REQUIRED_INPUTS[k][0] for k in missing]}. The skill needs a "
        f"running mapping service (occupancy_grid) and a running "
        f"navigation service (navigate/status/cancel) before it can "
        f"start. There is intentionally no hardcoded fallback — see "
        f"packaging-spec.md design invariant #1."
    )


def _register(stub) -> None:
    """Idempotent on re-deploy."""
    md_path = os.path.join(os.environ.get("ROBONIX_PKG_HOST_DIR", "/explore"),
                           "CAPABILITY.md")
    if not os.path.exists(md_path):
        md_path = ""
    try:
        stub.RegisterCapability(pb.RegisterCapabilityRequest(
            capability_id=CAP_ID,
            namespace=NAMESPACE,
            capability_md_path=md_path,
        ))
        log.info("registered cap %s namespace=%s", CAP_ID, NAMESPACE)
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.ALREADY_EXISTS:
            log.info("cap %s already registered (re-deploy)", CAP_ID)
        else:
            raise


def _declare_mcp_tools(stub) -> None:
    """Declare the 3 user-invocable contracts as MCP interfaces.
    Each contract id matches a TOML in capabilities/. Schemas are the
    LLM-side hints; the .srv files are the typed source of truth."""
    base = f"http://127.0.0.1:{MCP_PORT}/mcp/"
    tools = [
        ("robonix/skill/explore/explore",
         "Drive the robot to autonomously survey its surroundings, "
         "building an occupancy + semantic map. Returns a task_id; "
         "poll status() to track progress.",
         {"type": "object",
          "properties": {
              "area_hint":     {"type": "string", "default": ""},
              "timeout_s":     {"type": "number", "default": 600},
              "max_speed_m_s": {"type": "number", "default": 0.25},
          },
          "required": []}),
        ("robonix/skill/explore/status",
         "Get progress of a running exploration task. Empty task_id = "
         "the most recent.",
         {"type": "object",
          "properties": {"task_id": {"type": "string", "default": ""}}}),
        ("robonix/skill/explore/cancel",
         "Abort the active exploration. Idempotent.",
         {"type": "object",
          "properties": {"task_id": {"type": "string", "default": ""}}}),
    ]
    for contract_id, desc, schema in tools:
        try:
            stub.DeclareInterface(pb.DeclareInterfaceRequest(
                capability_id=CAP_ID,
                contract_id=contract_id,
                transport=pb.TRANSPORT_MCP,
                endpoint=base,
                params=pb.TransportParams(
                    mcp=pb.McpParams(description=desc,
                                      input_schema_json=json.dumps(schema)),
                ),
            ))
            log.info("declared %s → MCP %s", contract_id, base)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.ALREADY_EXISTS:
                log.info("%s already declared (re-deploy)", contract_id)
            else:
                log.warning("declare %s failed: %s", contract_id, e)


def _heartbeat_loop(stub) -> None:
    while True:
        try:
            stub.Heartbeat(pb.HeartbeatRequest(capability_id=CAP_ID))
        except Exception:
            pass
        time.sleep(HEARTBEAT_PERIOD_S)


# ── FastMCP server ────────────────────────────────────────────────────
def _make_mcp_server(ctrl: ExploreController):
    from fastmcp import FastMCP
    mcp = FastMCP("explore")

    @mcp.tool()
    def explore(area_hint: str = "", timeout_s: float = 600.0,
                max_speed_m_s: float = 0.25) -> dict:
        """Start an autonomous exploration task."""
        try:
            handle = ctrl.start(area_hint=area_hint,
                                timeout_s=float(timeout_s),
                                max_speed_m_s=float(max_speed_m_s))
            return {"accepted": True,
                    "task_id": handle.task_id,
                    "message": handle.detail}
        except RuntimeError as e:
            return {"accepted": False, "task_id": "", "message": str(e)}

    @mcp.tool()
    def status(task_id: str = "") -> dict:
        s = ctrl.status(task_id or None)
        if s is None:
            return {"known": False, "state": "idle",
                    "area_m2": 0.0, "frontiers_left": 0,
                    "elapsed_s": 0.0, "eta_s": -1.0,
                    "detail": "no task with that id"}
        return {"known": True, **s}

    @mcp.tool()
    def cancel(task_id: str = "") -> dict:
        ok, msg = ctrl.cancel(task_id or None)
        return {"ok": ok, "message": msg}

    return mcp


# ── Main ──────────────────────────────────────────────────────────────
def main() -> int:
    log.info("starting; atlas=%s mcp=:%d", ATLAS_ENDPOINT, MCP_PORT)

    stub = _atlas()
    for _ in range(10):
        try:
            _register(stub)
            break
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.ALREADY_EXISTS:
                break
            log.info("atlas not ready (%s); retrying", e.code())
            time.sleep(1.0)

    inputs = _resolve_inputs(stub)
    log.info("dependencies resolved: %s", inputs)

    ctrl = ExploreController(
        map_topic=inputs["map_topic"],
        nav_navigate_endpoint=inputs["nav_navigate"],
        nav_status_endpoint=inputs["nav_status"],
        nav_cancel_endpoint=inputs["nav_cancel"],
    )
    ctrl.start_runtime()  # rclpy thread + map subscriber

    mcp = _make_mcp_server(ctrl)

    threading.Thread(target=_heartbeat_loop, args=(stub,), daemon=True).start()
    _declare_mcp_tools(stub)

    log.info("FastMCP listening on 0.0.0.0:%d", MCP_PORT)
    try:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=MCP_PORT)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop_runtime()
    return 0


if __name__ == "__main__":
    sys.exit(main())
