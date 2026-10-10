# SPDX-License-Identifier: MulanPSL-2.0
"""Drive the real ExploreController through a simulated run."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .nav import SimNav, _dilate
from .robot import RobotParams
from .sensing import FREE_V, Mapper
from .world import World


class SimClock:
    """Stands in for the `time` module inside the controller, so a
    20-minute run takes seconds."""

    def __init__(self):
        self.t = 1_000_000.0

    def time(self) -> float:
        return self.t

    monotonic = time

    def sleep(self, dt: float) -> None:
        self.t += max(0.0, dt)


@dataclass
class Result:
    world: str
    state: str
    detail: str
    sim_time_s: float
    coverage: float                    # known-free / reachable free
    distance_m: float
    rotation_rad: float
    legs: int
    failed_legs: int
    collisions: int
    recoveries: int
    min_clearance_m: float
    goals: List[Tuple[float, float, str]] = field(default_factory=list)
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    collision_points: List[Tuple[float, float]] = field(default_factory=list)
    grid: Optional[np.ndarray] = None  # the final map
    # (simulated s since start, coverage, distance m), about once a second.
    curve: List[Tuple[float, float, float]] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.world:9s} {self.state:8s} t={self.sim_time_s:6.0f}s "
                f"cover={self.coverage * 100:5.1f}% "
                f"dist={self.distance_m:5.1f}m "
                f"turn={math.degrees(self.rotation_rad):6.0f}° "
                f"legs={self.legs:3d} failed={self.failed_legs:3d} "
                f"hits={self.collisions} recov={self.recoveries} "
                f"minclr={self.min_clearance_m:.2f}m  {self.detail}")


def reachable_free(world: World, robot: RobotParams) -> np.ndarray:
    """Free cells the robot's centre can reach from the start."""
    res = world.resolution
    blocked = _dilate(world.blocks_robot(),
                      int(math.ceil(robot.inscribed_m / res)))
    seen = np.zeros_like(blocked)
    sx, sy = world.to_cell(*world.start[:2])
    stack = [(sx, sy)]
    while stack:
        x, y = stack.pop()
        if not (0 <= x < world.width and 0 <= y < world.height):
            continue
        if seen[y, x] or blocked[y, x]:
            continue
        seen[y, x] = True
        stack.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
    # Count the area the lidar can map from there: reachable centres plus
    # the free cells around them.
    return _dilate(seen, int(math.ceil(1.0 / res))) & (world.cells == 0)


def run(world: World, *, robot: Optional[RobotParams] = None,
        timeout_s: float = 1800.0, scene: bool = False,
        strategy: Optional[str] = None, seed: int = 0) -> Result:
    from explore_skill import controller as ctl

    robot = robot or RobotParams()
    clock = SimClock()
    saved_time = ctl.time
    ctl.time = clock
    try:
        c = ctl.ExploreController(
            map_topic="/map",
            nav_navigate_endpoint="sim://nav",
            nav_status_endpoint="sim://nav",
            nav_cancel_endpoint="sim://nav",
            **({"scene_objects_endpoint": "sim://scene"} if scene else {}),
            **({"strategy": strategy,
                "robot_radius_m": robot.inscribed_m} if strategy else {}))
        mapper = Mapper(world, robot, np.random.default_rng(seed))
        target = reachable_free(world, robot)
        n_target = max(1, int(target.sum()))
        t0 = clock.time()
        curve: List[Tuple[float, float, float]] = []

        def sample():
            t = clock.time() - t0
            if curve and t - curve[-1][0] < 1.0:
                return
            known = (mapper.grid == FREE_V) & target
            curve.append((t, float(known.sum()) / n_target,
                          nav.distance_m if nav else 0.0))

        nav = None

        def on_tick(x, y, yaw):
            sample()
            # What the ROS spin thread does on /map and TF.
            with c._lock:
                c._latest_map = mapper.msg()
                c._latest_pose_xyyaw = (x, y, yaw)
                c._mark_visited(x, y, yaw)

        nav = SimNav(world, robot, mapper, clock, on_tick)
        c._nav_navigate_blocking = nav.navigate
        c._nav_cancel_rpc = lambda run_id="": None
        if scene:
            objs = [{"label": "table", "x": x, "y": y,
                     "size_x": sx, "size_y": sy}
                    for x, y, sx, sy in world.objects]
            c._scene_mcp_call = lambda tool, args: {"objects": objs}
        nav.sense()
        handle = ctl.TaskHandle(
            task_id="sim", started_at=clock.time(), timeout_s=timeout_s,
            max_speed_m_s=robot.max_vel_m_s)
        handle.thread = threading.current_thread()
        c._task = handle
        c._visited_cells = set()
        c._run_task(handle)
    finally:
        ctl.time = saved_time

    nav_ = nav
    curve.append((clock.time() - t0,
                  float(((mapper.grid == FREE_V) & target).sum()) / n_target,
                  nav_.distance_m))
    known = (mapper.grid == FREE_V) & target
    failed = [g for g in nav.goals if "SUCCEEDED" not in g[2]]
    return Result(
        world=world.name, state=handle.state, detail=handle.detail,
        sim_time_s=clock.time() - handle.started_at,
        coverage=float(known.sum()) / max(1, int(target.sum())),
        distance_m=nav.distance_m, rotation_rad=nav.rotation_rad,
        legs=len(nav.goals), failed_legs=len(failed),
        collisions=len(nav.collisions), recoveries=nav.recoveries,
        min_clearance_m=nav.min_clearance_m, goals=nav.goals,
        trajectory=nav.trajectory, collision_points=nav.collisions,
        grid=mapper.grid, curve=curve)
