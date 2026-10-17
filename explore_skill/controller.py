# SPDX-License-Identifier: MulanPSL-2.0
"""Exploration task lifecycle + frontier-loop driver.

State machine per task:

    explore() called ──► EXPLORING ──► (success) ──► DONE
                              │
                              ├──► (deadline)   ──► TIMEOUT
                              ├──► (cancel)     ──► CANCELED
                              └──► (nav errors) ──► ERROR

The controller owns one in-flight task at a time. Concurrent explore()
is rejected with accepted=false. Caller can cancel + retry.

Threading:
  - rclpy thread (ExploreController._spin_thread): hosts the /map
    subscriber and tf2 buffer.
  - task thread (per explore() invocation): runs the frontier loop.
  - heartbeat thread (in atlas_bridge): unrelated.

Nav integration:
  The skill calls service/navigation/navigate and its status/cancel controls as
  gRPC RPCs against the endpoints atlas resolved at startup. We do
  NOT subscribe to /goal_pose or /cmd_vel — that would skip the nav
  service's contract surface and break the abstraction.
"""
from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

log = logging.getLogger("explore_rbnx.controller")


@dataclass
class TaskHandle:
    task_id: str
    started_at: float
    timeout_s: float                  # 0 = no timeout
    max_speed_m_s: float
    state: str = "exploring"          # exploring | done | timeout | canceled | error
    detail: str = "started"
    last_target_xy: Optional[Tuple[float, float]] = None
    initial_area_m2: float = 0.0
    last_area_m2: float = 0.0
    last_progress_t: float = field(default_factory=time.time)
    area_at_last_progress_m2: float = 0.0
    cancel_requested: bool = False
    legs_completed: int = 0           # successful nav legs in this task
    # Frontiers navigation could not reach, and frontiers reached; both are
    # skipped afterwards (one still there after a visit cannot be cleared
    # from that goal: glass, beyond lidar range).
    failed_targets: list = field(default_factory=list)
    reached_targets: list = field(default_factory=list)
    close_visits: list = field(default_factory=list)   # second, close looks
    consecutive_nav_failures: int = 0
    empty_sweeps: int = 0             # sweeps in a row that found no target
    thread: Optional[threading.Thread] = None


@dataclass
class _View:
    """One round's map as frontier search sees it."""
    grid: Any
    pose: Optional[Tuple[float, float]]
    yaw: float
    area_m2: float
    frontiers: int
    min_size: int
    visited: set


@dataclass
class _Step:
    """What a round does: drive to `target`, wait for a pose, turn in
    place, or end the task with `end` = (state, detail)."""
    target: Any = None
    close: bool = False               # a second, close look at `target`
    wait: bool = False
    sweep: bool = False
    end: Optional[Tuple[str, str]] = None


class ExploreController:
    """Frontier-loop controller. One instance per service process;
    one active task at a time."""

    # Frontier search
    FRONTIER_MIN_SIZE_CELLS = 8           # smaller clusters are lidar speckle
    MAX_FRONTIER_DISTANCE_M = 6.0         # preferred radius; farther when none is nearer
    CLOSE_LOOK_MIN_SIZE_CELLS = 20        # frontiers this long get a second, close look
    FAILED_TARGET_RADIUS_M = 1.0          # skip frontiers this close to a failed one
    VISITED_RADIUS_M = 0.4                # cells around each pose marked visited
    VISITED_STEP_M = 0.2                  # poses are kept this far apart
    # The lidar's blind zone leaves the ground around the robot unknown;
    # these cells count as free so the robot is not ringed by frontiers.
    BLIND_RADIUS_M = 0.6
    # Goal clearance and standoff; None takes the robot radius and
    # frontier.DEFAULT_STANDOFF_M.
    TARGET_CLEARANCE_M = None
    TARGET_STANDOFF_M = None

    # When a task ends
    NO_PROGRESS_TIMEOUT_S = 180.0         # map not grown this long: stop
    PROGRESS_AREA_DELTA_M2 = 1.0          # growth that counts as progress
    MAX_CONSECUTIVE_NAV_FAILURES = 5
    MAX_EMPTY_SWEEPS = 2                  # in-place turns in a row with no goal

    # Navigation legs
    NAV_POLL_PERIOD_S = 1.0
    NAV_GOAL_TIMEOUT_S = 60.0             # minimum per leg
    LEG_TIMEOUT_PER_M_S = 7.0             # per metre of path, above the minimum
    LOOP_QUIET_PERIOD_S = 2.0             # pause between legs

    # Scene objects as keep-out circles
    KEEPOUT_TTL_S = 10.0                  # reuse a snapshot this long
    # Only objects whose vertical extent meets the robot's are in its way:
    # not a ceiling lamp or a picture on the wall, and not a rug.
    KEEPOUT_FLOOR_M = 0.05
    KEEPOUT_ROBOT_HEIGHT_M = 1.5

    def __init__(self, *, map_topic: str,
                 nav_navigate_endpoint: str,
                 nav_status_endpoint: str,
                 nav_cancel_endpoint: str,
                 scene_objects_endpoint: Optional[str] = None,
                 strategy: Optional[str] = None,
                 robot_radius_m: Optional[float] = None,
                 mrtsp: Optional[dict] = None,
                 costmap_topic: Optional[str] = None,
                 min_passage_m: float = 0.0):
        from .frontier import ROBOT_RADIUS_M
        from .strategies import DEFAULT_STRATEGY, validate
        self.map_topic = map_topic
        # Which frontier to go to next; see strategies.py.
        self.strategy = validate(strategy or DEFAULT_STRATEGY)
        self.robot_radius_m = robot_radius_m or ROBOT_RADIUS_M
        # mrtsp's parameters (strategies.MRTSP_DEFAULTS).
        self.mrtsp = dict(mrtsp or {})
        # Explore only through openings this wide (strategies.py); 0 is off.
        self.min_passage_m = float(min_passage_m or 0.0)
        # Navigation's global costmap (robonix/service/navigation/costmap),
        # whose lethal cells goals keep away from (frontier.overlay_lethal).
        # None: the map alone.
        self.costmap_topic = costmap_topic or None
        self._latest_costmap: Any = None
        self._costmap_received = 0.0
        self._costmap_notes: set = set()
        # Whether the last _with_costmap took the costmap; logged on change
        # when min_passage_m is on, since doorways are judged on both.
        self._costmap_applied = False
        self._gate_source: Optional[str] = None
        # All three nav endpoints typically point at the same FastMCP
        # server (http://host:port/mcp/) — atlas hands us the URL each
        # tool registered under. Keep separate fields so a future
        # multi-endpoint nav setup (e.g. nav2-wrapper for some modes,
        # simple_nav for others) still works.
        self._nav_endpoints = {
            "navigate": nav_navigate_endpoint,
            "status":   nav_status_endpoint,
            "cancel":   nav_cancel_endpoint,
        }
        # Optional: Scene sees what the lidar cannot. None means this
        # deployment has no Scene, and exploration proceeds without the
        # extra keep-outs exactly as it did before.
        self._scene_endpoint = scene_objects_endpoint
        self._keepout_cache: List[Tuple[float, float, float]] = []
        self._keepout_stamp = 0.0
        self._lock = threading.Lock()
        self._latest_map: Any = None     # latest OccupancyGrid msg
        self._latest_pose_xyyaw: Optional[Tuple[float, float, float]] = None
        # Where the robot has been this task, in map metres rounded to
        # VISITED_STEP_M. Positions, not cells: a SLAM map that grows moves
        # its origin, and cells recorded against an earlier map would land
        # somewhere else on the next one.
        self._visited_xy: set = set()
        self._task: Optional[TaskHandle] = None

        self._ros: Optional[dict] = None
        self._node = None
        self._spin_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        self._tf_buffer = None
        self._tf_listener = None

    # ── ROS runtime ─────────────────────────────────────────────────
    def start_runtime(self) -> None:
        """Spin up rclpy node + /map subscriber + tf2 listener."""
        if self._ros is not None:
            return
        self._ros = _import_ros()
        rclpy = self._ros["rclpy"]

        # Context.init() must only be called once
        if not rclpy.ok():
            rclpy.init(args=None)

        node = self._ros["Node"]("explore_skill")
        self._node = node

        QoS = self._ros["QoSProfile"]
        Rel = self._ros["ReliabilityPolicy"]
        Dur = self._ros["DurabilityPolicy"]
        Hist = self._ros["HistoryPolicy"]
        # /map is published TRANSIENT_LOCAL by mapping; we must match.
        map_qos = QoS(reliability=Rel.RELIABLE,
                      durability=Dur.TRANSIENT_LOCAL,
                      history=Hist.KEEP_LAST, depth=1)
        node.create_subscription(self._ros["OccupancyGrid"],
                                  self.map_topic, self._on_map, map_qos)
        if self.costmap_topic:
            node.create_subscription(self._ros["OccupancyGrid"],
                                     self.costmap_topic, self._on_costmap,
                                     map_qos)

        # tf2 for map→base_link lookup (so we know where the robot is)
        from tf2_ros import Buffer, TransformListener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

        self._stop_evt.clear()
        self._spin_thread = threading.Thread(target=self._spin_loop,
                                              daemon=True)
        self._spin_thread.start()
        log.info("rclpy spinning; subscribed to map_topic=%s", self.map_topic)

    def stop_runtime(self) -> None:
        if self._task and self._task.state == "exploring":
            self.cancel(None)
        self._stop_evt.set()
        if self._spin_thread:
            self._spin_thread.join(timeout=3.0)
        if self._ros:
            try:
                self._ros["rclpy"].shutdown()
            except Exception:
                pass

    def _spin_loop(self) -> None:
        rclpy = self._ros["rclpy"]
        while not self._stop_evt.is_set():
            try:
                rclpy.spin_once(self._node, timeout_sec=0.2)
                self._refresh_robot_pose()
            except Exception as e:  # noqa: BLE001
                log.exception("spin error: %s", e)
                time.sleep(0.1)

    def _on_map(self, msg) -> None:
        with self._lock:
            self._latest_map = msg

    # A costmap received longer ago than this is not used: navigation
    # publishes the whole grid about once a second, so an older one means it
    # stopped, and its marks may no longer match the map.
    COSTMAP_MAX_AGE_S = 5.0

    def _with_costmap(self, gv, map_msg):
        """`gv` with the costmap's lethal cells, or `gv` unchanged when there
        is no costmap, it is stale, or its frame cannot be put in the map's."""
        from .frontier import overlay_lethal
        with self._lock:
            cm, received = self._latest_costmap, self._costmap_received
        self._costmap_applied = False
        if cm is None:
            return gv
        age = time.monotonic() - received
        if age > self.COSTMAP_MAX_AGE_S:
            self._costmap_note("costmap received %.1f s ago; not used", age)
            return gv
        src, dst = cm.header.frame_id, map_msg.header.frame_id
        tf = (0.0, 0.0, 0.0)
        if src and dst and src != dst:
            try:
                t = self._tf_buffer.lookup_transform(
                    dst, src, self._ros["rclpy"].time.Time()).transform
                q = t.rotation
                tf = (t.translation.x, t.translation.y,
                      math.atan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y * q.y + q.z * q.z)))
            except Exception as e:  # noqa: BLE001
                self._costmap_note("no transform %s <- %s (%s); costmap not "
                                   "used", dst, src, e)
                return gv
        self._costmap_applied = True
        return overlay_lethal(gv, cm, tf)

    def _costmap_note(self, fmt, *args) -> None:
        """Log why the costmap was skipped, once per reason."""
        key = fmt.split(";")[0][:24]
        if key not in self._costmap_notes:
            self._costmap_notes.add(key)
            log.warning(fmt, *args)

    def _on_costmap(self, msg) -> None:
        with self._lock:
            self._latest_costmap, self._costmap_received = msg, time.monotonic()

    def _refresh_robot_pose(self) -> None:
        """Look up map→base_link; record (x,y,yaw) and where it has been."""
        if self._tf_buffer is None:
            return
        try:
            from rclpy.time import Time
            tr = self._tf_buffer.lookup_transform(
                "map", "base_link", Time(),
                timeout=self._ros["Duration"](seconds=0.5))
        except Exception:
            return
        x = float(tr.transform.translation.x)
        y = float(tr.transform.translation.y)
        # Quaternion → yaw (z rotation only — Force3DoF world).
        q = tr.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self._lock:
            self._latest_pose_xyyaw = (x, y, yaw)
            step = self.VISITED_STEP_M
            self._visited_xy.add((round(x / step), round(y / step)))

    def _blind_cells(self, gv, pose) -> set:
        """Cells within BLIND_RADIUS_M of `pose`; empty without a pose."""
        if pose is None:
            return set()
        cx, cy = gv.world_to_cell(pose[0], pose[1])
        r = max(1, int(round(self.BLIND_RADIUS_M / gv.resolution)))
        return {(cx + dx, cy + dy)
                for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                if dx * dx + dy * dy <= r * r}

    def _visited_cells(self, gv, visited_xy: set) -> set:
        """The cells of `gv` within VISITED_RADIUS_M of where the robot has
        been, found on this map from the recorded positions."""
        r = max(1, int(round(self.VISITED_RADIUS_M / gv.resolution)))
        disk = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                if dx * dx + dy * dy <= r * r]
        cells = set()
        for qx, qy in visited_xy:
            cx, cy = gv.world_to_cell(qx * self.VISITED_STEP_M, qy * self.VISITED_STEP_M)
            cells.update((cx + dx, cy + dy) for dx, dy in disk)
        return cells

    def start(self, *, area_hint: str, timeout_s: float,
              max_speed_m_s: float) -> TaskHandle:
        with self._lock:
            if self._task is not None and self._task.state == "exploring":
                raise RuntimeError(
                    f"another exploration task is already running "
                    f"(task_id={self._task.task_id}). Call cancel() "
                    f"before starting a new one.")
            # Reset visited history for the new task so we don't carry
            # over stale cells from a prior incomplete run.
            self._visited_xy = set()
            handle = TaskHandle(
                task_id="exp-" + uuid.uuid4().hex[:8],
                started_at=time.time(),
                timeout_s=float(timeout_s),
                max_speed_m_s=float(max_speed_m_s),
                detail=f"started (area_hint={area_hint!r})",
            )
            self._task = handle
        handle.thread = threading.Thread(
            target=self._run_task, args=(handle,),
            name=f"explore-{handle.task_id}", daemon=True)
        handle.thread.start()
        return handle

    def cancel(self, task_id: Optional[str]) -> Tuple[bool, str]:
        with self._lock:
            t = self._task
            if t is None:
                return True, "no active task (no-op)"
            if task_id and t.task_id != task_id:
                return False, f"task_id mismatch (active={t.task_id})"
            if t.state != "exploring":
                return True, f"task already in terminal state {t.state} (no-op)"
            t.cancel_requested = True
            t.detail = "cancel requested"
        # Best-effort: also tell nav to drop its current goal so the
        # robot stops moving immediately rather than after the leg
        # finishes.
        try:
            self._nav_cancel_rpc(run_id="")
        except Exception as e:  # noqa: BLE001
            log.warning("nav cancel rpc failed: %s", e)
        return True, f"cancel requested for {t.task_id}"

    def status(self, task_id: Optional[str]) -> Optional[dict]:
        with self._lock:
            t = self._task
            if t is None:
                return None
            if task_id and t.task_id != task_id:
                return None
            elapsed = time.time() - t.started_at
            return {
                "state": t.state,
                "area_m2": float(t.last_area_m2),
                "frontiers_left": _frontiers_left_count(self._latest_map),
                "elapsed_s": float(elapsed),
                "eta_s": -1.0,    # we don't currently estimate
                "detail": t.detail,
                "task_id": t.task_id,
            }

    # ── Frontier loop (runs in its own thread per task) ─────────────
    def _run_task(self, handle: TaskHandle) -> None:
        """The task's thread. Whatever goes wrong ends the task with an
        error: a thread that dies quietly leaves it exploring for good, which
        refuses every new task and any cancel."""
        try:
            self._explore(handle)
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] exploration failed", handle.task_id)
            try:
                self._nav_cancel_rpc()
            except Exception:  # noqa: BLE001 — navigation may be what failed
                pass
            self._terminate(handle, "error", f"{type(e).__name__}: {e}")

    def _explore(self, handle: TaskHandle) -> None:
        """Explore until no frontier is worth a visit, then end the task.

        Each round observes the map, chooses the next step and drives there.
        The task ends in `_interrupted` (cancel, timeout), `_next_step`
        (done, stalled, nothing reachable) or `_drive` (navigation keeps
        failing)."""
        from .frontier import GridView, mapped_free_area_m2
        log.info("[%s] exploration task starting", handle.task_id)
        first = self._wait_for_map(timeout_s=15.0)
        if first is None:
            self._terminate(handle, "error",
                            "no /map received within 15s — is mapping running?")
            return
        area = mapped_free_area_m2(GridView.from_msg(first))
        handle.initial_area_m2 = handle.last_area_m2 = area
        handle.area_at_last_progress_m2 = area
        deadline = (handle.started_at + handle.timeout_s
                    if handle.timeout_s > 0 else None)

        while True:
            end = self._interrupted(handle, deadline)
            if end:
                self._terminate(handle, *end)
                return
            view = self._observe(handle)
            if view is None:
                time.sleep(0.5)
                continue
            step = self._next_step(handle, view)
            if step.end:
                self._terminate(handle, *step.end)
                return
            if step.wait:
                time.sleep(0.5)
            elif step.sweep:
                handle.empty_sweeps += 1
                self._sweep_360(handle, deadline=deadline)
            else:
                handle.empty_sweeps = 0
                end = self._drive(handle, step, view, deadline)
                if end:
                    self._terminate(handle, *end)
                    return

    def _interrupted(self, handle: TaskHandle,
                     deadline: Optional[float]) -> Optional[Tuple[str, str]]:
        if handle.cancel_requested:
            return "canceled", "user-requested cancel"
        if deadline is not None and time.time() > deadline:
            return "timeout", f"hit {handle.timeout_s}s ceiling"
        return None

    def _observe(self, handle: TaskHandle) -> Optional["_View"]:
        """The map as frontier search sees it, and the progress it shows.

        Cells the robot has driven over or cannot see under itself count as
        free and the costmap's lethal cells as occupied. mrtsp searches the
        optimized map (map_opt.py) and ignores frontiers shorter than
        `min_frontier_m`."""
        from .frontier import (GridView, fill_traversed, mapped_free_area_m2,
                               total_frontier_count)
        with self._lock:
            latest = self._latest_map
            pose_xyyaw = self._latest_pose_xyyaw
            visited_xy = set(self._visited_xy)
        if latest is None:
            return None
        pose = pose_xyyaw[:2] if pose_xyyaw else None
        gv = GridView.from_msg(latest)
        visited = self._visited_cells(gv, visited_xy)
        area = mapped_free_area_m2(gv)
        # Growth since the last progress, so a map that grows a little every
        # round still counts.
        if area - handle.area_at_last_progress_m2 > self.PROGRESS_AREA_DELTA_M2:
            handle.last_progress_t = time.time()
            handle.area_at_last_progress_m2 = area
        handle.last_area_m2 = area

        gv = fill_traversed(gv, visited | self._blind_cells(gv, pose))
        gv = self._with_costmap(gv, latest)
        if self.min_passage_m > 0:
            source = "map + costmap" if self._costmap_applied else "map only"
            if source != self._gate_source:
                self._gate_source = source
                log.info("[%s] passages of %.2f m judged on %s",
                         handle.task_id, self.min_passage_m, source)
        min_size = self.FRONTIER_MIN_SIZE_CELLS
        if self.strategy == "mrtsp":
            from .map_opt import optimize
            from .strategies import MRTSP_DEFAULTS
            raw = total_frontier_count(gv, min_size=min_size)
            gv = optimize(gv)
            min_frontier_m = {**MRTSP_DEFAULTS, **self.mrtsp}["min_frontier_m"]
            min_size = max(min_size, int(round(min_frontier_m / gv.resolution)))
        if self.min_passage_m > 0 and pose is not None:
            # Rooms off the corridor do not count, so the run ends when the
            # corridor is done.
            from .strategies import passage_frontier_count
            frontiers = passage_frontier_count(
                gv, pose, robot_radius_m=self.robot_radius_m,
                min_passage_m=self.min_passage_m, min_size=min_size)
        else:
            frontiers = total_frontier_count(gv, min_size=min_size)
        if self.strategy == "mrtsp":
            log.info("[%s] frontiers raw=%d optimized=%d",
                     handle.task_id, raw, frontiers)
        return _View(grid=gv, pose=pose,
                     yaw=pose_xyyaw[2] if pose_xyyaw else 0.0, area_m2=area,
                     frontiers=frontiers, min_size=min_size, visited=visited)

    def _next_step(self, handle: TaskHandle, view: "_View") -> "_Step":
        """Where to go next, or why to stop.

        Goals are asked for from the strictest to the most lenient: a near
        frontier not tried yet, any such frontier, then a close look at one
        still there after a visit. With none left, a run that has tried
        every frontier ends. Otherwise mrtsp relaxes the room a goal needs
        and falls back to a greedy goal before it ends, and greedy turns in
        place to see more before it gives up."""
        if view.frontiers == 0:
            return _Step(end=("done",
                              f"no frontiers left (area={view.area_m2:.1f}m²)"))
        if time.time() - handle.last_progress_t > self.NO_PROGRESS_TIMEOUT_S:
            return _Step(end=("error",
                              f"map has not grown for "
                              f"{self.NO_PROGRESS_TIMEOUT_S:.0f}s with "
                              f"{view.frontiers} clusters left "
                              f"(area={view.area_m2:.1f}m²)"))
        if view.pose is None:
            log.info("[%s] no map→base_link tf yet, waiting...", handle.task_id)
            return _Step(wait=True)

        from .strategies import pick
        tried = handle.failed_targets + handle.reached_targets
        keepout = self._keepout_circles()

        def goal(strategy: Optional[str] = None, **overrides):
            args = {"keepout": keepout, "min_size": view.min_size,
                    "visited_cells": view.visited, "blocked_xy": tried,
                    "blocked_radius_m": self.FAILED_TARGET_RADIUS_M,
                    "clearance_m": self.TARGET_CLEARANCE_M or self.robot_radius_m,
                    "max_distance_m": math.inf}
            if self.TARGET_STANDOFF_M is not None:
                args["standoff_m"] = self.TARGET_STANDOFF_M
            args.update(overrides)
            return pick(strategy or self.strategy, view.grid, view.pose,
                        robot_yaw=view.yaw, robot_radius_m=self.robot_radius_m,
                        mrtsp=self.mrtsp, min_passage_m=self.min_passage_m,
                        **args)

        target = goal(max_distance_m=self.MAX_FRONTIER_DISTANCE_M)
        if target is None:
            target = goal()
        if target is not None:
            return _Step(target=target)
        if handle.reached_targets:
            # A frontier long enough to hide a passage and still there after
            # a visit is looked at once more, from as close as the robot fits.
            target = goal(close=True,
                          blocked_xy=handle.failed_targets + handle.close_visits,
                          min_size=self.CLOSE_LOOK_MIN_SIZE_CELLS)
            if target is not None:
                return _Step(target=target, close=True)
        if tried and goal(blocked_xy=None) is not None:
            # Every frontier left has been tried: done if any was reached.
            return _Step(end=("done" if handle.reached_targets else "error",
                              f"{view.frontiers} clusters left, all tried: "
                              f"{len(handle.failed_targets)} unreachable, "
                              f"{len(set(handle.reached_targets))} still there "
                              f"after a visit"))
        if self.strategy != "frontier_greedy":
            # Ask with the chassis' clearance only, then for a greedy goal 1 m
            # short of the frontier: nav2 may get through a doorway the path
            # search reads as closed.
            target = goal(room_margin_m=0.0)
            if target is None:
                target = goal("frontier_greedy",
                              blocked_xy=tried + [view.pose], standoff_m=1.0)
            if target is not None:
                return _Step(target=target)
            return _Step(end=("done",
                              f"{view.frontiers} clusters left, none reachable "
                              f"(area={view.area_m2:.1f}m²)"))
        if handle.empty_sweeps >= self.MAX_EMPTY_SWEEPS:
            return _Step(end=("error",
                              f"no reachable frontier after "
                              f"{handle.empty_sweeps} sweeps ({view.frontiers} "
                              f"raw clusters, {len(handle.failed_targets)} "
                              f"failed targets)"))
        # Frontiers but no safe goal yet: turning in place shows the lidar
        # more free space around the robot, where a goal can then go.
        handle.detail = (f"no safe frontier; spinning to expand FOV "
                         f"({view.frontiers} raw clusters detected)")
        log.info("[%s] %s", handle.task_id, handle.detail)
        return _Step(sweep=True)

    def _drive(self, handle: TaskHandle, step: "_Step", view: "_View",
               deadline: Optional[float]) -> Optional[Tuple[str, str]]:
        """Drive to the step's goal and record the outcome. Returns why the
        task ends, or None to go on."""
        target = step.target
        fx, fy = target.centroid_xy       # the frontier
        tx, ty = target.drive_to          # where the robot stops, short of it
        handle.last_target_xy = (fx, fy)
        handle.detail = (f"driving to ({tx:.2f},{ty:.2f}), short of "
                         f"frontier ({fx:.2f},{fy:.2f}) "
                         f"size={target.size}, {view.frontiers} clusters left")
        log.info("[%s] %s", handle.task_id, handle.detail)

        # The searched path when the strategy has one, else 1.5 times the
        # straight line.
        leg_m = (target.path_m if target.path_m is not None
                 else 1.5 * math.hypot(tx - view.pose[0], ty - view.pose[1]))
        leg_timeout = self._bounded_timeout(
            max(self.NAV_GOAL_TIMEOUT_S, self.LEG_TIMEOUT_PER_M_S * leg_m),
            deadline)
        if leg_timeout <= 0.0:
            return "timeout", f"hit {handle.timeout_s}s ceiling"
        # Arrive facing the frontier.
        hx, hy = (tx, ty) if (fx - tx) ** 2 + (fy - ty) ** 2 > 0.01 else view.pose
        ok, msg = self._nav_navigate_blocking(
            tx, ty, yaw=math.atan2(fy - hy, fx - hx),
            timeout_s=leg_timeout, cancel_evt=handle)
        if handle.cancel_requested:
            return "canceled", "cancel during nav"
        if step.close:
            handle.close_visits.append((fx, fy))
        if ok:
            handle.legs_completed += 1
            handle.consecutive_nav_failures = 0
            handle.reached_targets.append((fx, fy))
        else:
            # Not fatal: remember the frontier so it is not chosen again.
            handle.failed_targets.append((fx, fy))
            handle.consecutive_nav_failures += 1
            log.warning("[%s] nav leg failed: %s (%d in a row)",
                        handle.task_id, msg, handle.consecutive_nav_failures)
            if handle.consecutive_nav_failures >= self.MAX_CONSECUTIVE_NAV_FAILURES:
                return ("error", f"{handle.consecutive_nav_failures} nav legs "
                                 f"failed in a row; last: {msg}")
            handle.detail = f"nav leg failed ({msg}); trying next frontier"
        time.sleep(self.LOOP_QUIET_PERIOD_S)
        return None

    def _terminate(self, handle: TaskHandle, state: str, detail: str) -> None:
        with self._lock:
            handle.state = state
            handle.detail = detail
        log.info("[%s] task %s: %s", handle.task_id, state, detail)

    def _wait_for_map(self, timeout_s: float) -> Any:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._latest_map is not None:
                    return self._latest_map
            time.sleep(0.2)
        return None

    # ── Sweep / coverage helpers ───────────────────────────────────
    @staticmethod
    def _bounded_timeout(requested_s: float,
                         deadline: Optional[float]) -> float:
        if deadline is None:
            return requested_s
        return max(0.0, min(requested_s, deadline - time.time()))

    def _sweep_360(self, handle: TaskHandle, *,
                   deadline: Optional[float]) -> None:
        """Turn in place a full circle, as four goals at the current xy with
        the yaw advanced 90° each time, so the sensors see all around."""
        with self._lock:
            pose = self._latest_pose_xyyaw
        if pose is None:
            return
        x, y, yaw = pose
        for step in (0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi):
            if handle.cancel_requested:
                return
            segment_timeout = self._bounded_timeout(15.0, deadline)
            if segment_timeout <= 0.0:
                return
            target_yaw = yaw + step
            handle.detail = (f"sweep at ({x:.2f},{y:.2f}) "
                             f"yaw={math.degrees(target_yaw)%360:.0f}°")
            log.info("[%s] %s", handle.task_id, handle.detail)
            ok, _ = self._nav_navigate_blocking(
                x, y, yaw=target_yaw, timeout_s=segment_timeout,
                cancel_evt=handle)
            if not ok:
                return
            time.sleep(0.5)

    # ── Nav RPC over MCP HTTP ──────────────────────────────────────
    # simple_nav exposes navigate/status/cancel as MCP tools, so we
    # call them via FastMCP's client. atlas_bridge resolved the MCP
    # endpoint URL for each contract; in practice all three point at
    # the same FastMCP server, but we keep them separate so a future
    # multi-nav setup still works.
    def _keepout_circles(self) -> List[Tuple[float, float, float]]:
        """Where Scene says there is furniture, as (x, y, radius) circles.

        A tabletop is invisible to a lidar that passes under it, so the
        occupancy grid calls that space free and a goal placed there ends
        with the robot against a table leg. Scene has the object from the
        camera; this asks for it.

        Returns an empty list on any failure. The alternative -- refusing
        to explore because an optional input is unavailable -- trades a
        working explorer for a stricter one.
        """

        if not self._scene_endpoint:
            return []
        now = time.time()
        if now - self._keepout_stamp < self.KEEPOUT_TTL_S:
            return self._keepout_cache

        with self._lock:
            latest = self._latest_map
        map_frame = latest.header.frame_id if latest is not None else "map"
        resp = self._scene_mcp_call("list_objects", {})
        circles: List[Tuple[float, float, float]] = []
        for obj in (resp.get("objects") or []):
            label = str(obj.get("label") or "")
            if label == "robot":
                continue                      # itself, not an obstacle
            if obj.get("frame_id") and obj.get("frame_id") != map_frame:
                continue                      # placed in another frame
            size_x = float(obj.get("size_x") or 0.0)
            size_y = float(obj.get("size_y") or 0.0)
            if size_x <= 0.0 or size_y <= 0.0:
                continue                      # no footprint: a room or area entry
            size_z = float(obj.get("size_z") or 0.0)
            z = float(obj.get("z") or 0.0)
            if size_z > 0.0 and not (z - size_z / 2 < self.KEEPOUT_ROBOT_HEIGHT_M
                                     and z + size_z / 2 > self.KEEPOUT_FLOOR_M):
                continue                      # above the robot, or flat on the floor
            # Half the footprint diagonal covers the box whatever its yaw,
            # and the chassis radius is added because the goal is where the
            # robot's centre goes, not where its edge stops.
            radius = (0.5 * math.hypot(size_x, size_y)
                      + self.robot_radius_m)
            circles.append((float(obj.get("x") or 0.0),
                            float(obj.get("y") or 0.0), radius))

        self._keepout_cache = circles
        self._keepout_stamp = now
        if circles:
            log.info("keep-out from scene: %d object(s)", len(circles))
        return circles

    def _scene_mcp_call(self, tool: str, args: dict) -> dict:
        """Scene is a different MCP server from navigation, so it gets its
        own client rather than borrowing the nav one's base URL."""
        try:
            return _run_coroutine_sync(
                lambda: self._scene_mcp_call_async(tool, args))
        except Exception as error:  # noqa: BLE001
            log.warning("scene mcp call %s failed: %s", tool, error)
            return {}

    async def _scene_mcp_call_async(self, tool: str, args: dict) -> dict:
        import json

        from fastmcp import Client

        async with Client(self._scene_endpoint) as c:
            result = await c.call_tool(tool, args)
            if not result.content:
                return {}
            try:
                return json.loads(result.content[0].text)
            except Exception:  # noqa: BLE001
                return {}

    async def _mcp_call(self, tool: str, args: dict) -> dict:
        """Single MCP tool round-trip. Async because fastmcp's client
        is async — we await inside a fresh event loop in the caller.

        A new Client per call: calls come from the task thread and from
        MCP handler threads, each with its own event loop, and every
        call already opens its own MCP session."""
        from fastmcp import Client
        # All three endpoints typically share the same base URL.
        async with Client(self._nav_endpoints["navigate"]) as c:
            result = await c.call_tool(tool, args)
            # FastMCP returns a list of TextContent; the tool returned
            # JSON-serialised dict in its sole entry.
            if not result.content:
                return {}
            import json
            txt = result.content[0].text
            try:
                return json.loads(txt)
            except Exception:
                return {"raw": txt}

    def _mcp_call_sync(self, tool: str, args: dict) -> dict:
        try:
            return _run_coroutine_sync(lambda: self._mcp_call(tool, args))
        except Exception as e:  # noqa: BLE001
            log.warning("mcp call %s failed: %s", tool, e)
            return {}

    def _nav_navigate_blocking(self, x: float, y: float, *,
                                 yaw: Optional[float],
                                 timeout_s: float,
                                 cancel_evt: TaskHandle) -> Tuple[bool, str]:
        """Send a goal via the nav MCP `navigate` tool, then poll the
        sibling `status` tool until terminal or timeout."""
        # Navigate.srv carries a PoseStamped (`goal`) — encode xy in
        # position and yaw as a unit quaternion in orientation. The nav
        # service reads the identity quaternion (yaw=None) as yaw 0, not
        # as "any heading", so callers that care must pass one.
        if yaw is not None:
            qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        else:
            qz, qw = 0.0, 1.0
        goal_pose = {
            "header": {"frame_id": "map",
                       "stamp": {"sec": 0, "nanosec": 0}},
            "pose": {
                "position": {"x": float(x), "y": float(y), "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": qz, "w": qw},
            },
        }
        resp = self._mcp_call_sync("navigate", {"goal": goal_pose})
        if not resp.get("accepted", False):
            return False, f"goal rejected: {resp.get('detail', '')}"
        run_id = resp.get("run_id", "")

        # Poll status. Empty run_id is allowed: navigation falls back to the
        # most recent run.
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if cancel_evt.cancel_requested:
                self._nav_cancel_rpc(run_id=run_id)
                return False, "canceled during nav"
            if self._goal_blocked(x, y):
                # The goal was chosen before the camera saw what is there
                # (a table top the lidar's map lacks). Driving on puts the
                # robot against it; give the leg up and choose again.
                self._nav_cancel_rpc(run_id=run_id)
                return False, "goal now inside an obstacle in the costmap"
            sresp = self._mcp_call_sync("status", {"run_id": run_id})
            if sresp:
                state = str(sresp.get("state", "")).upper()
                if state in ("SUCCEEDED", "FAILED", "CANCELED", "TIMEOUT"):
                    return state == "SUCCEEDED", f"nav terminal: {state}"
            time.sleep(self.NAV_POLL_PERIOD_S)
        self._nav_cancel_rpc(run_id=run_id)
        return False, "leg timeout"

    def _goal_blocked(self, x: float, y: float) -> bool:
        """Whether the costmap now has a lethal cell within the robot's
        radius of the goal. False without a costmap."""
        from .frontier import GridView, is_target_safe
        with self._lock:
            latest = self._latest_map
            have = self._latest_costmap is not None
        if not have or latest is None:
            return False
        gv = self._with_costmap(GridView.from_msg(latest), latest)
        return not is_target_safe(gv, x, y, safe_radius_m=self.robot_radius_m)

    def _nav_cancel_rpc(self, run_id: str = "") -> None:
        self._mcp_call_sync("cancel", {"run_id": run_id})


# ── Helpers ───────────────────────────────────────────────────────────
# A hung nav or Scene server must not hang the task: without a bound the
# task stayed "exploring" forever, cancel could not stop it, and start()
# refused every new task.
MCP_CALL_TIMEOUT_S = 10.0


def _run_coroutine_sync(make_coro) -> Any:
    """Run the coroutine returned by make_coro() to completion, or raise
    TimeoutError after MCP_CALL_TIMEOUT_S.

    asyncio.run() refuses to start inside a running event loop, which
    is where the MCP cancel handler calls cancel() from. In that case
    run it on a short-lived thread with its own loop instead."""
    import asyncio

    def bounded():
        return asyncio.wait_for(make_coro(), MCP_CALL_TIMEOUT_S)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(bounded())

    result: dict = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(bounded())
        except BaseException as e:  # noqa: BLE001
            result["error"] = e

    t = threading.Thread(target=runner, name="explore-mcp-call", daemon=True)
    t.start()
    t.join(MCP_CALL_TIMEOUT_S + 1.0)
    if t.is_alive():
        raise TimeoutError(f"mcp call did not finish in {MCP_CALL_TIMEOUT_S}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _frontiers_left_count(latest_map: Any) -> int:
    if latest_map is None:
        return -1
    try:
        from .frontier import GridView, total_frontier_count
        gv = GridView.from_msg(latest_map)
        return int(total_frontier_count(gv))
    except Exception:
        return -1


def _import_ros() -> dict:
    """Lazy ROS import — keeps unit tests importable without rclpy."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                            DurabilityPolicy, HistoryPolicy)
    from rclpy.duration import Duration
    from nav_msgs.msg import OccupancyGrid
    return {
        "rclpy": rclpy,
        "Node": Node,
        "QoSProfile": QoSProfile,
        "ReliabilityPolicy": ReliabilityPolicy,
        "DurabilityPolicy": DurabilityPolicy,
        "HistoryPolicy": HistoryPolicy,
        "Duration": Duration,
        "OccupancyGrid": OccupancyGrid,
    }
