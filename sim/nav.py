# SPDX-License-Identifier: MulanPSL-2.0
"""A stand-in for the navigation service and Nav2 behind it.

What it keeps from Nav2 because explore's behaviour depends on it:

- The global costmap is the map plus an obstacle layer. Lethal cells are
  inflated by the inscribed radius; unknown cells are not traversable
  (`allow_unknown: false`), except where the obstacle layer's raytrace
  (3 m) or the robot's footprint has cleared them.
- NavFn moves a blocked goal to the nearest traversable cell within 0.5 m
  and fails beyond that.
- The path is replanned at 1 Hz on the costmap as the map grows.
- A failed plan runs the default BT's recovery (wait, back up, spin) before
  the goal is reported FAILED.
- At the goal the robot turns to the requested heading. An identity
  quaternion, which is what explore sends when it passes no heading, is
  yaw 0: the nav service does not treat it as "any heading".

The robot moves through the ground-truth world: a tabletop the lidar never
saw still stops it, and that is counted as a collision.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

from .robot import RobotParams
from .sensing import OCC_V, UNKNOWN, Mapper
from .world import World

_LETHAL = 254


def _disc(r_cells: int) -> List[Tuple[int, int]]:
    return [(dx, dy) for dy in range(-r_cells, r_cells + 1)
            for dx in range(-r_cells, r_cells + 1)
            if dx * dx + dy * dy <= r_cells * r_cells]


def _dilate(mask: np.ndarray, r_cells: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dx, dy in _disc(r_cells):
        src = mask[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
        out[max(0, dy):h - max(0, -dy) or h, max(0, dx):w - max(0, -dx) or w] |= src
    return out


def _distance_to(mask: np.ndarray, max_cells: int) -> np.ndarray:
    """Chessboard-ish distance (in cells) to the nearest True, capped."""
    dist = np.full(mask.shape, float(max_cells), dtype=float)
    dist[mask] = 0.0
    for r in range(1, max_cells + 1):
        ring = _dilate(mask, r) & (dist > r)
        dist[ring] = r
    return dist


class SimNav:
    def __init__(self, world: World, robot: RobotParams, mapper: Mapper,
                 clock, on_tick):
        self.world = world
        self.robot = robot
        self.mapper = mapper
        self.clock = clock
        self.on_tick = on_tick              # (x, y, yaw) -> None, each period
        self.pose = list(world.start)
        self.cleared = np.zeros(world.cells.shape, dtype=bool)
        self._truth_block = world.blocks_robot()
        res = world.resolution
        self._truth_dist = _distance_to(
            self._truth_block, int(math.ceil(1.0 / res))) * res
        # Stats.
        self.trajectory: List[Tuple[float, float]] = [tuple(world.start[:2])]
        self.distance_m = 0.0
        self.rotation_rad = 0.0
        self.collisions: List[Tuple[float, float]] = []
        self.min_clearance_m = float("inf")
        self.goals: List[Tuple[float, float, str]] = []
        self.recoveries = 0

    # ── sensing ────────────────────────────────────────────────────
    def sense(self) -> None:
        x, y, _ = self.pose
        free = self.mapper.scan(x, y)
        res = self.world.resolution
        near = ((free[:, 0] + 0.5) * res - x) ** 2 + \
               ((free[:, 1] + 0.5) * res - y) ** 2 <= \
            self.robot.costmap_raytrace_m ** 2
        self.cleared[free[near, 1], free[near, 0]] = True
        cx, cy = self.world.to_cell(x, y)
        r = int(math.ceil(self.robot.circumscribed_m / res))
        for dx, dy in _disc(r):
            if 0 <= cx + dx < self.world.width and 0 <= cy + dy < self.world.height:
                self.cleared[cy + dy, cx + dx] = True
        self.on_tick(*self.pose)

    # ── costmap and planning ───────────────────────────────────────
    def _costmap(self) -> np.ndarray:
        """Cost per map cell: 0 free .. 253 inflated, 254 lethal, 255
        unknown. Planning runs on a 2x coarser grid for speed."""
        res = self.world.resolution
        grid = self.mapper.grid
        occ = grid >= OCC_V
        unknown = (grid == UNKNOWN) & ~self.cleared
        lethal = _dilate(occ, int(math.ceil(self.robot.inscribed_m / res)))
        infl_cells = int(math.ceil(self.robot.inflation_radius_m / res))
        d = _distance_to(occ, infl_cells) * res
        cost = np.where(d < self.robot.inflation_radius_m,
                        252.0 * np.exp(-2.5 * np.maximum(
                            0.0, d - self.robot.inscribed_m)), 0.0)
        cost[lethal] = _LETHAL
        cost[unknown & ~lethal] = 255
        return cost

    def _plan(self, start, goal) -> Optional[List[Tuple[float, float]]]:
        res = self.world.resolution
        f = 2                                          # coarse factor
        cost = self._costmap()
        h, w = cost.shape[0] // f, cost.shape[1] // f
        coarse = cost[:h * f, :w * f].reshape(h, f, w, f).max(axis=(1, 3))
        cres = res * f

        def cell(p):
            return (min(w - 1, max(0, int(p[0] / cres))),
                    min(h - 1, max(0, int(p[1] / cres))))

        s = cell(start)
        g = cell(goal)
        passable = coarse < _LETHAL
        passable[s[1], s[0]] = True        # NavFn clears the robot's cell
        if not passable[g[1], g[0]]:
            tol = int(self.robot.planner_tolerance_m / cres)
            best = None
            for dy in range(-tol, tol + 1):
                for dx in range(-tol, tol + 1):
                    x, y = g[0] + dx, g[1] + dy
                    if 0 <= x < w and 0 <= y < h and passable[y, x] and \
                            dx * dx + dy * dy <= tol * tol:
                        d = dx * dx + dy * dy
                        if best is None or d < best[0]:
                            best = (d, (x, y))
            if best is None:
                return None
            g = best[1]
        dist = {s: 0.0}
        prev = {}
        pq = [(0.0, s)]
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414),
                 (-1, -1, 1.414)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == g:
                break
            if d > dist.get(u, math.inf):
                continue
            for dx, dy, sc in steps:
                v = (u[0] + dx, u[1] + dy)
                if not (0 <= v[0] < w and 0 <= v[1] < h):
                    continue
                if not passable[v[1], v[0]]:
                    continue
                nd = d + sc * (1.0 + coarse[v[1], v[0]] / 50.0)
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if g not in dist:
            return None
        path = [g]
        while path[-1] != s:
            path.append(prev[path[-1]])
        path.reverse()
        return [((cx + 0.5) * cres, (cy + 0.5) * cres) for cx, cy in path]

    # ── motion ─────────────────────────────────────────────────────
    def _turn_to(self, yaw: float) -> None:
        d = (yaw - self.pose[2] + math.pi) % (2 * math.pi) - math.pi
        self.rotation_rad += abs(d)
        self.clock.sleep(abs(d) / self.robot.max_yaw_rate)
        self.pose[2] = yaw

    def _step_to(self, x: float, y: float, heading: float) -> bool:
        """Move to (x, y) through the real world, facing `heading` (the
        path a little ahead, as the controller tracks it). False on
        contact."""
        cx, cy = self.world.to_cell(x, y)
        if not (0 <= cx < self.world.width and 0 <= cy < self.world.height):
            return False
        clearance = self._truth_dist[cy, cx]
        self.min_clearance_m = min(self.min_clearance_m, clearance)
        if clearance < self.robot.inscribed_m:
            self.collisions.append((x, y))
            return False
        d = (heading - self.pose[2] + math.pi) % (2 * math.pi) - math.pi
        if abs(d) > 0.8:
            self._turn_to(heading)          # too sharp: turn in place first
        else:
            self.rotation_rad += abs(d)     # turned while walking
            self.pose[2] = heading
        px, py, _ = self.pose
        step = math.hypot(x - px, y - py)
        self.distance_m += step
        self.clock.sleep(step / self.robot.max_vel_m_s)
        self.pose[0], self.pose[1] = x, y
        self.trajectory.append((x, y))
        return True

    def _recover(self) -> None:
        self.recoveries += 1
        self.rotation_rad += self.robot.recovery_spin_rad
        self.pose[2] += self.robot.recovery_spin_rad
        self.clock.sleep(self.robot.recovery_s)
        self.sense()

    def navigate(self, x: float, y: float, *, yaw: Optional[float],
                 timeout_s: float, cancel_evt) -> Tuple[bool, str]:
        """The controller's _nav_navigate_blocking, on the simulated robot."""
        deadline = self.clock.time() + timeout_s
        goal_yaw = 0.0 if yaw is None else yaw   # identity quaternion = yaw 0
        self.goals.append((x, y, "pending"))
        result = self._navigate(x, y, goal_yaw, deadline, cancel_evt)
        self.goals[-1] = (x, y, result[1])
        return result

    def _navigate(self, x, y, goal_yaw, deadline, cancel_evt):
        recovered = False
        while True:
            if cancel_evt.cancel_requested:
                return False, "canceled during nav"
            if self.clock.time() > deadline:
                return False, "leg timeout"
            px, py, _ = self.pose
            if math.hypot(x - px, y - py) <= self.robot.xy_goal_tolerance:
                self._turn_to(goal_yaw)
                return True, "nav terminal: SUCCEEDED"
            path = self._plan((px, py), (x, y))
            if path is None:
                if recovered:
                    return False, "nav terminal: FAILED"
                self._recover()
                recovered = True
                continue
            # NavFn's tolerance may end the path short of the goal; that
            # still succeeds once the robot is at the end of the path.
            if len(path) <= 1:
                ex, ey = path[-1]
                if math.hypot(ex - x, ey - y) > self.robot.xy_goal_tolerance:
                    self._turn_to(goal_yaw)
                    return True, "nav terminal: SUCCEEDED"
            # Drive for one replan period, sensing as the map updates.
            budget = self.robot.max_vel_m_s * self.robot.replan_period_s
            for i in range(1, len(path)):
                wx, wy = path[i]
                ax, ay = path[min(i + 8, len(path) - 1)]
                px, py, _ = self.pose
                seg = math.hypot(wx - px, wy - py)
                heading = math.atan2(ay - py, ax - px) if (ax, ay) != (px, py) \
                    else self.pose[2]
                if not self._step_to(wx, wy, heading):
                    self.sense()
                    return False, "collision"
                budget -= seg
                if budget <= 0:
                    break
            self.sense()
            if path and math.hypot(path[-1][0] - self.pose[0],
                                   path[-1][1] - self.pose[1]) <= \
                    self.robot.xy_goal_tolerance:
                self._turn_to(goal_yaw)
                return True, "nav terminal: SUCCEEDED"
