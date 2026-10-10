# SPDX-License-Identifier: MulanPSL-2.0
"""Frontier selection strategies, chosen by the `strategy` config key.

    frontier_greedy  the original scorer in frontier.py: frontier size over
                     straight-line distance, goal on the straight line back
                     to the robot.
    mrtsp            the greedy MRTSP choice of frontier_exploration_ros2
                     (https://github.com/mertgulerx/frontier_exploration_ros2,
                     Apache-2.0, Copyright 2026 Mert Güler), after Liu et
                     al., Sci Rep 15, 12261 (2025),
                     doi:10.1038/s41598-025-97231-9.
                     Re-implemented from src/mrtsp_ordering.cpp at 0d61180.

MRTSP, the minimum ratio travelling salesman problem, orders frontiers by
cost over gain. The greedy solver's first target is the frontier with the
lowest start-row cost

    M(0, j) = wd * D(j) / P(j) + t_lb(j) / sqrt(P(j))
    D(j)    = max(dm + du, dn + dv) - sensor_effective_range
    t_lb(j) = path length / vmax + |heading change| / wmax

where P(j) is the frontier's size in cells and

    dm  path length from the robot to the frontier's center cell
        (the frontier cell nearest its centroid)
    dn  straight line from the robot to the centroid
    du  center cell to the frontier's entry cell (the first one the search
        reaches)
    dv  centroid to the entry cell

Only that first target is used: the robot replans after every leg, so the
rest of the order would be thrown away.

Path lengths come from a search over the cells the robot fits in, so a
frontier behind a wall is as far as the way round, and one with no way to
it is dropped. The goal is our own choice, not the package's: a reachable
cell 0.9-2.0 m from the frontier that sees it, with the most room around
it, then the shortest way from the robot.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .frontier import (DEFAULT_PREFERRED_CLEARANCE_M, OCC_THRESH,
                       FrontierCluster, GridView, anchor_cell,
                       cluster_frontiers, find_frontier_cells, in_keepout,
                       pick_target)

STRATEGIES = ("frontier_greedy", "mrtsp")
DEFAULT_STRATEGY = "mrtsp"

# The search runs on cells this many times coarser than the map: 0.1 m on
# a 0.05 m map, a quarter of the cells to visit.
_COARSE = 2
# Where around a frontier the goal may go. The near limit is past the
# lidar's blind zone (controller.BLIND_RADIUS_M): from closer in, the
# frontier is under the robot and the visit cannot clear it.
_GOAL_MIN_M, _GOAL_MAX_M = 0.9, 2.0
# frontier_exploration_ros2's defaults, with the Lite3's speeds.
_WEIGHT_DISTANCE = 1.0
_SENSOR_EFFECTIVE_RANGE_M = 1.5
_VMAX, _WMAX = 0.35, 0.6
# Goals closer than this to the robot do not move exploration on.
_MIN_GOAL_DISTANCE_M = 0.8


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r or (dx == 0 and dy == 0):
                continue
            out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] |= \
                mask[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
    return out


class _Reach:
    """Path distances from the robot over the cells it fits in."""

    def __init__(self, gv: GridView, robot_xy: Tuple[float, float],
                 robot_radius_m: float, keepout=None):
        f = _COARSE
        h, w = gv.height // f, gv.width // f
        d = gv.data[:h * f, :w * f].reshape(h, f, w, f)
        occ = (d >= OCC_THRESH).any(axis=(1, 3))
        unknown = (d == -1).all(axis=(1, 3))
        self.res = gv.resolution * f
        self.gv, self.h, self.w = gv, h, w
        self.occ = occ
        r = max(1, int(math.ceil(robot_radius_m / self.res)))
        self.blocked = _dilate(occ, r) | unknown
        # Scene keep-outs (tabletops the lidar passes under) are walls to
        # the search too, so a frontier reached only under a table is
        # dropped. A circle the robot already stands in is left open.
        for kx, ky, kr in keepout or ():
            if math.hypot(robot_xy[0] - kx, robot_xy[1] - ky) <= kr:
                continue
            ys, xs = np.ogrid[:h, :w]
            wx = gv.origin_x + (xs + 0.5) * self.res
            wy = gv.origin_y + (ys + 0.5) * self.res
            self.blocked |= (wx - kx) ** 2 + (wy - ky) ** 2 <= kr * kr
        # Room around each cell, up to the preferred clearance.
        cap = int(math.ceil(DEFAULT_PREFERRED_CLEARANCE_M / self.res))
        room = np.full(occ.shape, float(cap))
        grown = occ.copy()
        room[grown] = 0.0
        for k in range(1, cap + 1):
            nxt = _dilate(grown, 1)
            room[nxt & ~grown] = k
            grown = nxt
        self.room_m = room * self.res
        self.dist = self._search(self.cell(*robot_xy))

    def cell(self, x: float, y: float) -> Tuple[int, int]:
        gv = self.gv
        return (min(self.w - 1, max(0, int((x - gv.origin_x) / self.res))),
                min(self.h - 1, max(0, int((y - gv.origin_y) / self.res))))

    def world(self, cx: int, cy: int) -> Tuple[float, float]:
        return (self.gv.origin_x + (cx + 0.5) * self.res,
                self.gv.origin_y + (cy + 0.5) * self.res)

    def _search(self, start: Tuple[int, int]) -> np.ndarray:
        dist = np.full((self.h, self.w), np.inf)
        dist[start[1], start[0]] = 0.0
        pq = [(0.0, start)]
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142),
                 (-1, -1, 1.4142))
        blocked, h, w = self.blocked, self.h, self.w
        while pq:
            d, (x, y) = heapq.heappop(pq)
            if d > dist[y, x]:
                continue
            for dx, dy, c in steps:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not blocked[ny, nx]:
                    nd = d + c
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        heapq.heappush(pq, (nd, (nx, ny)))
        return dist * self.res

    def path_m(self, xy: Tuple[float, float]) -> float:
        """Path length to `xy`, or to the nearest cell around it the robot
        fits in: frontier cells sit next to unknown, which is blocked."""
        cx, cy = self.cell(*xy)
        y0, x0 = max(0, cy - 1), max(0, cx - 1)
        near = self.dist[y0:cy + 2, x0:cx + 2]
        return float(near.min()) if near.size else math.inf

    def sees(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)
        for i in range(1, n):
            x = a[0] + round((b[0] - a[0]) * i / n)
            y = a[1] + round((b[1] - a[1]) * i / n)
            if self.occ[y, x]:
                return False
        return True

    def goal_near(self, target: Tuple[float, float], keepout
                  ) -> Optional[Tuple[Tuple[float, float], float]]:
        """A reachable cell 0.9-2.0 m from `target` that sees it: the one
        with the most room, then the shortest way from the robot, so the goal
        sits on the robot's side and the robot arrives facing the frontier.
        Returns (goal, path m)."""
        tx, ty = self.cell(*target)
        r = int(math.ceil(_GOAL_MAX_M / self.res))
        y0, y1 = max(0, ty - r), min(self.h, ty + r + 1)
        x0, x1 = max(0, tx - r), min(self.w, tx + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        sep = np.hypot(xx - tx, yy - ty) * self.res
        ok = (np.isfinite(self.dist[y0:y1, x0:x1])
              & (sep >= _GOAL_MIN_M) & (sep <= _GOAL_MAX_M))
        if not ok.any():
            return None
        room = self.room_m[y0:y1, x0:x1][ok]
        order = np.lexsort((self.dist[y0:y1, x0:x1][ok], -room))
        cx, cy = xx[ok][order], yy[ok][order]
        for i in range(min(len(order), 40)):
            c = (int(cx[i]), int(cy[i]))
            if not self.sees(c, (tx, ty)):
                continue
            goal = self.world(*c)
            if in_keepout(goal[0], goal[1], keepout):
                continue
            return goal, float(self.dist[c[1], c[0]])
        return None


def _candidates(gv: GridView, robot_xy, *, min_size, robot_radius_m,
                blocked_xy, blocked_radius_m, keepout):
    """(cluster with its goal, path length to the goal, MRTSP terms)."""
    cells = find_frontier_cells(gv)
    if cells.size == 0:
        return []
    reach = _Reach(gv, robot_xy, robot_radius_m, keepout)
    out = []
    for c in cluster_frontiers(cells, min_size=min_size):
        center = gv.cell_to_world(*anchor_cell(c))
        if blocked_xy and any(math.hypot(center[0] - bx, center[1] - by)
                              <= blocked_radius_m for bx, by in blocked_xy):
            continue
        found = reach.goal_near(center, keepout)
        if found is None:
            continue
        goal, goal_path_m = found
        res = gv.resolution
        world = [(gv.origin_x + (x + 0.5) * res, gv.origin_y + (y + 0.5) * res)
                 for x, y in c.cell_indices]
        paths = [reach.path_m(p) for p in world]
        entry = world[int(np.argmin(paths))]
        centroid = (sum(p[0] for p in world) / len(world),
                    sum(p[1] for p in world) / len(world))
        dm = reach.path_m(center)
        if not math.isfinite(dm):
            dm = goal_path_m + math.hypot(center[0] - goal[0],
                                          center[1] - goal[1])
        out.append((FrontierCluster(centroid_xy=center, size=c.size,
                                    cell_indices=c.cell_indices,
                                    goal_xy=goal),
                    goal_path_m,
                    {"dm": dm, "centroid": centroid, "entry": entry}))
    return out


def _mrtsp_cost(cand, robot_xy, robot_yaw):
    """compute_mrtsp_start_cost() in mrtsp_ordering.cpp, weights 1."""
    cluster, _, t = cand
    gain = float(cluster.size)
    center, centroid, entry = cluster.centroid_xy, t["centroid"], t["entry"]
    dm = t["dm"]
    dn = math.hypot(centroid[0] - robot_xy[0], centroid[1] - robot_xy[1])
    du = math.hypot(center[0] - entry[0], center[1] - entry[1])
    dv = math.hypot(centroid[0] - entry[0], centroid[1] - entry[1])
    path_cost = max(dm + du, dn + dv) - _SENSOR_EFFECTIVE_RANGE_M
    heading = math.atan2(center[1] - robot_xy[1], center[0] - robot_xy[0])
    turn = abs(math.atan2(math.sin(heading - robot_yaw),
                          math.cos(heading - robot_yaw)))
    time_cost = dm / _VMAX + turn / _WMAX
    return _WEIGHT_DISTANCE * path_cost / gain + time_cost / math.sqrt(gain)


def pick(strategy: str, gv: GridView, robot_xy: Tuple[float, float], *,
         robot_yaw: float = 0.0,
         min_size: int,
         robot_radius_m: float,
         max_distance_m: float = math.inf,
         visited_cells: Optional[set] = None,
         blocked_xy: Optional[Sequence[Tuple[float, float]]] = None,
         blocked_radius_m: float = 1.0,
         keepout: Optional[List[Tuple[float, float, float]]] = None,
         **frontier_kwargs) -> Optional[FrontierCluster]:
    """The next frontier to visit under `strategy`, or None."""
    if strategy == "frontier_greedy":
        return pick_target(gv, robot_xy, min_size=min_size,
                           max_distance_m=max_distance_m,
                           visited_cells=visited_cells, blocked_xy=blocked_xy,
                           blocked_radius_m=blocked_radius_m, keepout=keepout,
                           **frontier_kwargs)
    cands = [c for c in _candidates(
        gv, robot_xy, min_size=min_size, robot_radius_m=robot_radius_m,
        blocked_xy=blocked_xy, blocked_radius_m=blocked_radius_m,
        keepout=keepout) if c[1] <= max_distance_m]
    far = [c for c in cands
           if math.hypot(c[0].goal_xy[0] - robot_xy[0],
                         c[0].goal_xy[1] - robot_xy[1]) >= _MIN_GOAL_DISTANCE_M]
    pool = far or cands
    if not pool:
        return None
    return min(pool, key=lambda c: _mrtsp_cost(c, robot_xy, robot_yaw))[0]


def validate(strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown explore strategy {strategy!r}; "
                         f"expected one of {', '.join(STRATEGIES)}")
    return strategy


