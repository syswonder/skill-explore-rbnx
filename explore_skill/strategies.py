# SPDX-License-Identifier: MulanPSL-2.0
"""Frontier selection strategies, chosen by the `strategy` config key.

    frontier_greedy  frontier.py's scorer: size over straight-line
                     distance, the goal on the line back to the robot.
                     Default.
    mrtsp            Liu et al., Sci Rep 15, 12261 (2025),
                     doi:10.1038/s41598-025-97231-9: frontiers on the
                     optimized map (map_opt.py) put in the visiting order
                     of Eq. 5-8, and the first one taken (_mrtsp_order).

mrtsp measures distance along a path over the cells the robot fits in, so
a frontier behind a wall is as far as the way round and one with no way to
it is dropped. Its goal is our choice, not the paper's: a reachable cell
0.9-2.0 m from the frontier that sees it, with the most room, then the
shortest way from the robot.

`min_passage_m` keeps either strategy to space the robot reaches through
openings at least that wide (a corridor, not the rooms off it): the path
search keeps that half-width from obstacles, and a frontier cell counts
only next to a cell the search reached and outside any opening narrower
than that. Cells seen through a doorway are not reached, so a room's
frontier is dropped however large it is.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .frontier import (DEFAULT_PREFERRED_CLEARANCE_M, OCC_THRESH,
                       FrontierCluster, GridView, anchor_cell,
                       cluster_frontiers, find_frontier_cells, in_keepout,
                       pick_target, score_clusters)

STRATEGIES = ("frontier_greedy", "mrtsp")
DEFAULT_STRATEGY = "frontier_greedy"

# The search runs on cells this many times coarser than the map: 0.1 m on
# a 0.05 m map, a quarter of the cells to visit.
_COARSE = 2
# Where around a frontier the goal may go. The near limit is past the
# lidar's blind zone (controller.BLIND_RADIUS_M): from closer in, the
# frontier is under the robot and the visit cannot clear it.
_GOAL_MIN_M, _GOAL_MAX_M = 0.9, 2.0
# The Lite3's speeds; a deployment sets its own under `mrtsp:`.
_VMAX, _WMAX = 0.35, 0.6
# frontier_greedy's standoff for a second, close look.
_CLOSE_STANDOFF_M = 0.25
# Goals closer than this to the robot do not move exploration on.
_MIN_GOAL_DISTANCE_M = 0.8
# Room a goal needs beyond the robot's radius, so nav2 can still turn and
# back out there; a small room is looked into from its doorway.
GOAL_ROOM_MARGIN_M = 0.2


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
                 robot_radius_m: float, keepout=None,
                 room_margin_m: float = GOAL_ROOM_MARGIN_M,
                 min_passage_m: float = 0.0):
        f = _COARSE
        h, w = gv.height // f, gv.width // f
        d = gv.data[:h * f, :w * f].reshape(h, f, w, f)
        occ = (d >= OCC_THRESH).any(axis=(1, 3))
        unknown = (d == -1).all(axis=(1, 3))
        self.res = gv.resolution * f
        self.gv, self.h, self.w = gv, h, w
        self.occ = occ
        # With min_passage_m, a gap narrower than that is shut as well.
        clear_m = max(robot_radius_m, min_passage_m / 2)
        r = max(1, int(math.ceil(clear_m / self.res)))
        self.blocked = _dilate(occ, r) | unknown
        # Scene keep-outs (tables the lidar sees through) block the search
        # too, except a circle the robot already stands in.
        for kx, ky, kr in keepout or ():
            if math.hypot(robot_xy[0] - kx, robot_xy[1] - ky) <= kr:
                continue
            ys, xs = np.ogrid[:h, :w]
            wx = gv.origin_x + (xs + 0.5) * self.res
            wy = gv.origin_y + (ys + 0.5) * self.res
            self.blocked |= (wx - kx) ** 2 + (wy - ky) ** 2 <= kr * kr
        # A doorway whose far jamb is not mapped yet has unknown for a side,
        # so the gate keeps the same distance from unknown too, specks
        # (sparse beams) aside. Frontier cells sit next to unknown; they
        # count when the search got within that distance of them.
        gate = None
        if min_passage_m > 0:
            solid = ~_dilate(~unknown, 1)
            gate = self.blocked | _dilate(solid, r + 1)
        # The robot stands where it is: the inflation of a wall within its
        # radius must not wall in the search's start, or near a wall every
        # frontier reads as out of reach.
        sx, sy = self.cell(*robot_xy)
        ys, xs = np.ogrid[:h, :w]
        stand = ((xs - sx) ** 2 + (ys - sy) ** 2 <= r * r) & ~occ & ~unknown
        self.blocked &= ~stand
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
        self.min_room_m = robot_radius_m + room_margin_m
        start = self.cell(*robot_xy)
        self.dist = self._search(start)
        self._near = None
        if gate is not None:
            # A frontier cell counts next to a cell reached with walls kept
            # at the gate's distance, and within that distance of one
            # reached with unknown kept there as well. Goals and paths
            # take the second, stricter search.
            self._near = np.isfinite(self.dist)
            self.blocked = gate & ~stand
            self.dist = self._search(start)
            self._cover = _dilate(np.isfinite(self.dist), r + 1)

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

    def reached(self, cells: np.ndarray) -> np.ndarray:
        """Which map cells (cx, cy) the gated search got to (see __init__)."""
        near = self._near.copy()
        near[1:, :] |= self._near[:-1, :]
        near[:-1, :] |= self._near[1:, :]
        near[:, 1:] |= near[:, :-1].copy()
        near[:, :-1] |= near[:, 1:].copy()
        x = np.minimum(cells[:, 0] // _COARSE, self.w - 1)
        y = np.minimum(cells[:, 1] // _COARSE, self.h - 1)
        return near[y, x] & self._cover[y, x]

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

    def goal_near(self, target: Tuple[float, float], keepout,
                  close: bool = False
                  ) -> Optional[Tuple[Tuple[float, float], float]]:
        """A reachable cell 0.9-2.0 m from `target` that sees it: the one
        with the most room, then the shortest way from the robot, so the goal
        sits on the robot's side and the robot arrives facing the frontier.
        With `close`, the reachable cell nearest the frontier instead, for
        a second look at one the first visit could not clear.
        Returns (goal, path m)."""
        tx, ty = self.cell(*target)
        r = int(math.ceil(_GOAL_MAX_M / self.res))
        y0, y1 = max(0, ty - r), min(self.h, ty + r + 1)
        x0, x1 = max(0, tx - r), min(self.w, tx + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        sep = np.hypot(xx - tx, yy - ty) * self.res
        near_m = 0.0 if close else _GOAL_MIN_M
        ok = (np.isfinite(self.dist[y0:y1, x0:x1])
              & (sep >= near_m) & (sep <= _GOAL_MAX_M)
              & (self.room_m[y0:y1, x0:x1] >= self.min_room_m))
        if not ok.any():
            return None
        room = self.room_m[y0:y1, x0:x1][ok]
        if close:
            order = np.argsort(sep[ok], kind="stable")
        else:
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


# Directions across which an opening's width is measured.
_ACROSS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _in_opening(gv: GridView, cells: np.ndarray, width_m: float) -> np.ndarray:
    """Which cells (cx, cy) lie in a gap narrower than `width_m`: obstacles
    on both sides, along one of four directions, closer together than that.
    A doorway is one; so is the inner corner of two walls."""
    occ = gv.data >= OCC_THRESH
    out = np.zeros(len(cells), dtype=bool)
    for dx, dy in _ACROSS:
        step = gv.resolution * math.hypot(dx, dy)
        n = int(math.ceil(width_m / step))
        gap = np.zeros(len(cells))
        for sign in (1, -1):
            first = np.full(len(cells), n)
            for k in range(n, 0, -1):
                x = cells[:, 0] + sign * k * dx
                y = cells[:, 1] + sign * k * dy
                ok = (x >= 0) & (x < gv.width) & (y >= 0) & (y < gv.height)
                hit = np.zeros(len(cells), dtype=bool)
                hit[ok] = occ[y[ok], x[ok]]
                first[hit] = k
            gap += first
        out |= (gap - 1) * step < width_m
    return out


def passage_frontier_cells(gv: GridView, reach: "_Reach",
                           min_passage_m: float) -> np.ndarray:
    """Frontier cells, and with `min_passage_m` only those next to space
    `reach` got to through openings that wide and outside any narrower
    opening. Filtered before clustering: a frontier running from the
    corridor through a doorway into a room keeps its corridor part only."""
    cells = find_frontier_cells(gv)
    if min_passage_m <= 0 or cells.size == 0:
        return cells
    keep = reach.reached(cells) & ~_in_opening(gv, cells, min_passage_m)
    return cells[keep]


def passage_frontier_count(gv: GridView, robot_xy, *, robot_radius_m: float,
                           min_passage_m: float, min_size: int) -> int:
    """How many frontier clusters `passage_frontier_cells` leaves."""
    reach = _Reach(gv, robot_xy, robot_radius_m, min_passage_m=min_passage_m)
    cells = passage_frontier_cells(gv, reach, min_passage_m)
    return len(cluster_frontiers(cells, min_size=min_size)) if cells.size else 0


def _candidates(gv: GridView, robot_xy, *, min_size, robot_radius_m,
                blocked_xy, blocked_radius_m, keepout, close=False,
                room_margin_m=GOAL_ROOM_MARGIN_M, min_passage_m=0.0):
    """(cluster with its goal, path length to the goal, MRTSP terms)."""
    if find_frontier_cells(gv).size == 0:
        return []
    reach = _Reach(gv, robot_xy, robot_radius_m, keepout, room_margin_m,
                   min_passage_m)
    cells = passage_frontier_cells(gv, reach, min_passage_m)
    if cells.size == 0:
        return []
    out = []
    for c in cluster_frontiers(cells, min_size=min_size):
        center = gv.cell_to_world(*anchor_cell(c))
        if blocked_xy and any(math.hypot(center[0] - bx, center[1] - by)
                              <= blocked_radius_m for bx, by in blocked_xy):
            continue
        found = reach.goal_near(center, keepout, close)
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
        # The centroid of a curved frontier can sit in unknown space or in
        # a wall; then go round by the center.
        dn = reach.path_m(centroid)
        if not math.isfinite(dn):
            dn = dm + math.hypot(centroid[0] - center[0],
                                 centroid[1] - center[1])
        out.append((FrontierCluster(centroid_xy=center, size=c.size,
                                    cell_indices=c.cell_indices,
                                    goal_xy=goal, path_m=goal_path_m),
                    goal_path_m,
                    {"dm": dm, "dn": dn, "centroid": centroid,
                     "entry": entry}))
    return out


# Eq. 5-8 parameters. The paper gives no values for the weights; the rest
# are the robot's and are overridden from config (`mrtsp:`).
#
# r_s defaults to 0. With r_s > 0 a frontier nearer than r_s gets a negative
# d, and a negative d over a small P is smaller still, so the cheapest move
# is the smallest frontier in reach: the robot goes for slivers in corners.
# Frontiers shorter than min_frontier_m are not candidates at all.
MRTSP_DEFAULTS = {"wd": 1.0, "ws": 1.0, "sensor_range_m": 0.0,
                  "vmax_m_s": _VMAX, "wmax_rad_s": _WMAX,
                  "min_frontier_m": 1.0}


def _mrtsp_order(cands, robot_xy, robot_yaw, p):
    """Visiting order under Liu et al.'s MRTSP, as candidate indices.

    V_0 is the robot, V_1..V_n the frontiers, each with a center c (the
    frontier cell nearest the mean), a centroid g and a starting cell x
    (the one the robot reaches first). For a move V_i -> V_j:

        d(V_i, V_j) = max(d_m + d_u, d_n + d_v) - r_s            (Eq. 5)
            d_m, d_n  V_i to c_j and to g_j
            d_u, d_v  g_j and c_j to x_j
        P(V_i, V_j) = S_j, the frontier's size in cells           (Eq. 6)
        M[i][j] = Wd d / (Ws P)                  i != 0, j != 0   (Eq. 7a)
                = Wd d / (Ws P) + t_lb(V_i, V_j)  i == 0, j != 0   (Eq. 7b)
                = 0                               j == 0          (Eq. 7c)
        t_lb    = min(L / v_max, min(|Q|, 2 pi - |Q|) / w_max)   (Eq. 8)
            L   straight line from the robot to V_j
            Q   turn from the robot's heading to face V_j

    Eq. 8 is printed with min; FUEL, which the paper follows, bounds the
    time with max. We keep the paper's. From the robot, d_m and d_n are
    path lengths on the map; between two frontiers they are straight
    lines, since a search from every frontier costs seconds per decision.

    The order starts at the robot and is built greedily on M, then
    improved by 2-opt; the matrix is asymmetric, so each move is scored
    by recomputing the path. The return leg costs 0 (Eq. 7c), so the path
    is open."""
    n = len(cands)
    wd, ws, rs = p["wd"], p["ws"], p["sensor_range_m"]
    c = [k[0].centroid_xy for k in cands]
    g = [k[2]["centroid"] for k in cands]
    x = [k[2]["entry"] for k in cands]
    # S as frontier length in metres, not cells: the paper leaves the unit
    # open, and in cells the ratio term shrinks with the map resolution until
    # t_lb (seconds) alone decides.
    size = [float(k[0].size) * p["resolution_m"] for k in cands]
    hyp = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    M = np.zeros((n + 1, n + 1))
    for j in range(n):
        du, dv = hyp(g[j], x[j]), hyp(c[j], x[j])
        dm, dn = cands[j][2]["dm"], cands[j][2]["dn"]
        d = max(dm + du, dn + dv) - rs
        q = math.atan2(c[j][1] - robot_xy[1], c[j][0] - robot_xy[0]) - robot_yaw
        q = abs(math.atan2(math.sin(q), math.cos(q)))
        t_lb = min(hyp(robot_xy, c[j]) / p["vmax_m_s"],
                   min(q, 2 * math.pi - q) / p["wmax_rad_s"])
        M[0, j + 1] = wd * d / (ws * size[j]) + t_lb
        for i in range(n):
            if i != j:
                d = max(hyp(c[i], c[j]) + du, hyp(c[i], g[j]) + dv) - rs
                M[i + 1, j + 1] = wd * d / (ws * size[j])
    order, left = [], set(range(1, n + 1))
    at = 0
    while left:
        at = min(left, key=lambda j: M[at, j])
        order.append(at)
        left.discard(at)

    def cost(seq):
        return M[0, seq[0]] + sum(M[a, b] for a, b in zip(seq, seq[1:]))
    best = cost(order)
    improved = True
    while improved:
        improved = False
        for a in range(n - 1):
            for b in range(a + 1, n):
                trial = order[:a] + order[a:b + 1][::-1] + order[b + 1:]
                tc = cost(trial)
                if tc < best - 1e-9:
                    order, best, improved = trial, tc, True
    return [k - 1 for k in order]


def pick(strategy: str, gv: GridView, robot_xy: Tuple[float, float], *,
         robot_yaw: float = 0.0,
         min_size: int,
         robot_radius_m: float,
         max_distance_m: float = math.inf,
         visited_cells: Optional[set] = None,
         blocked_xy: Optional[Sequence[Tuple[float, float]]] = None,
         blocked_radius_m: float = 1.0,
         keepout: Optional[List[Tuple[float, float, float]]] = None,
         close: bool = False,
         mrtsp: Optional[dict] = None,
         room_margin_m: float = GOAL_ROOM_MARGIN_M,
         min_passage_m: float = 0.0,
         **frontier_kwargs) -> Optional[FrontierCluster]:
    """The next frontier to visit under `strategy`, or None. `close` asks
    for a goal as near the frontier as the robot fits. `min_passage_m`
    keeps frontiers and goals to space reached through openings that wide."""
    if strategy == "frontier_greedy":
        if close:
            frontier_kwargs["standoff_m"] = _CLOSE_STANDOFF_M
        if min_passage_m <= 0:
            return pick_target(gv, robot_xy, min_size=min_size,
                               max_distance_m=max_distance_m,
                               visited_cells=visited_cells,
                               blocked_xy=blocked_xy,
                               blocked_radius_m=blocked_radius_m,
                               keepout=keepout, **frontier_kwargs)
        reach = _Reach(gv, robot_xy, robot_radius_m, keepout,
                       min_passage_m=min_passage_m)
        cells = passage_frontier_cells(gv, reach, min_passage_m)
        if cells.size == 0:
            return None
        scored = score_clusters(
            cluster_frontiers(cells, min_size=min_size), gv, robot_xy,
            max_distance_m=max_distance_m, visited_cells=visited_cells,
            blocked_xy=blocked_xy, blocked_radius_m=blocked_radius_m,
            keepout=keepout, **frontier_kwargs)
        # The goal on the line to the frontier must be in reached space too.
        for _, c in scored:
            if reach.reached(np.array([gv.world_to_cell(*c.goal_xy)]))[0]:
                return c
        return None
    cands = [c for c in _candidates(
        gv, robot_xy, min_size=min_size, robot_radius_m=robot_radius_m,
        blocked_xy=blocked_xy, blocked_radius_m=blocked_radius_m,
        keepout=keepout, close=close, room_margin_m=room_margin_m,
        min_passage_m=min_passage_m)
        if c[1] <= max_distance_m]
    far = [c for c in cands
           if math.hypot(c[0].goal_xy[0] - robot_xy[0],
                         c[0].goal_xy[1] - robot_xy[1]) >= _MIN_GOAL_DISTANCE_M]
    pool = cands if close else (far or cands)
    if not pool:
        return None
    order = _mrtsp_order(pool, robot_xy, robot_yaw,
                         {**MRTSP_DEFAULTS, **(mrtsp or {}),
                          "resolution_m": gv.resolution})
    return pool[order[0]][0]


# mrtsp divides by these, so zero is refused as well as a negative.
_MRTSP_POSITIVE = ("ws", "vmax_m_s", "wmax_rad_s")


def config_problem(cfg: dict):
    """Why the skill's config cannot be used, or None. rbnx checks that a
    config.spec is well formed, not a deployment's values against it, so
    the values are checked here."""
    strategy = cfg.get("strategy") or DEFAULT_STRATEGY
    if strategy not in STRATEGIES:
        return (f"unknown explore strategy {strategy!r}; "
                f"expected one of {', '.join(STRATEGIES)}")
    mrtsp = cfg.get("mrtsp") or {}
    if not isinstance(mrtsp, dict):
        return "mrtsp must be a mapping of parameters"
    for key, value in mrtsp.items():
        if key not in MRTSP_DEFAULTS:
            return f"mrtsp.{key} is not a parameter; expected {', '.join(MRTSP_DEFAULTS)}"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"mrtsp.{key} must be a number, not {value!r}"
        if value < 0 or (value == 0 and key in _MRTSP_POSITIVE):
            bound = "greater than 0" if key in _MRTSP_POSITIVE else "at least 0"
            return f"mrtsp.{key} must be {bound}, not {value}"
    passage = cfg.get("min_passage_m", 0)
    if isinstance(passage, bool) or not isinstance(passage, (int, float)) \
            or passage < 0:
        return f"min_passage_m must be a number at least 0, not {passage!r}"
    return None


def validate(strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown explore strategy {strategy!r}; "
                         f"expected one of {', '.join(STRATEGIES)}")
    return strategy


