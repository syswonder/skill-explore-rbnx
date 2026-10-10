# SPDX-License-Identifier: MulanPSL-2.0
"""Frontier extraction + scoring on nav_msgs/OccupancyGrid.

Standard formulation: a "frontier cell" is a free cell (occupancy=0)
adjacent to at least one unknown cell (occupancy=-1). Cells are
clustered into frontier *groups*, each group represented by its
centroid + size. The controller picks the highest-scoring group as
the next exploration target.

Scoring trades off information gain (cluster size — bigger frontier
= more unknown will become known if visited) against travel cost
(Euclidean distance from robot to centroid as a cheap proxy for the
true planner cost). nav2 / RRT-based explorers use the actual
costmap-aware planner cost, but for the dev demo a Euclidean proxy is
fine and avoids re-implementing A*.

We deliberately don't filter against an inflation halo here — that's
the navigation service's costmap layer's job. If the chosen frontier
is technically unreachable due to obstacle inflation, the nav RPC
will fail with a planning error and the controller picks the next
candidate. Re-doing inflation in this module would duplicate state
and violate the layering rule (frontier finder has no contract
dependency on inflation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


OCC_THRESH = 50  # ≥ this counts as obstacle (matches nav2 convention)

# The chassis the goal has to fit. Kept equal to nav2's `robot_radius` in
# examples/webots/config/nav2_params.yaml -- a clearance smaller than the
# robot is not a safety check, and this one was 0.15.
ROBOT_RADIUS_M = 0.22
# The chassis, and no more. nav2 plans this goal with the same radius and
# runs an inflation layer on top of it; asking for chassis-plus-margin here,
# against the raw map, charges for that margin twice and -- measured on two
# live maps -- turns down every frontier in a furnished room.
DEFAULT_CLEARANCE_M = ROBOT_RADIUS_M

# How far short of the frontier line to aim. A centroid sits on the
# free/unknown boundary by construction, so driving to it means driving to
# the edge of the known world every time; stopping a little back puts the
# robot in space that has been observed, still close enough that the sensors
# cover what lies beyond.
#
# It is where the search starts, not a requirement. If the robot does not
# fit there the line is walked in and out before the frontier is given up:
# "stand exactly this far back" was never the goal, "get near it somewhere
# the robot fits" is.
DEFAULT_STANDOFF_M = 0.45
_STANDOFF_STEP_M = 0.15
# How far back along the line the search may go. The robot only has to see
# the frontier, and the lidar reaches metres, so standing further back costs
# little and buys room.
_MAX_EXTRA_STANDOFF_STEPS = 8

# Clearance worth having, as opposed to the clearance required. Among the
# points that fit, the search takes the one furthest from anything occupied,
# up to this. A chassis-height lidar sees only the legs of a table, so the
# top reaches about half a table depth past the nearest occupied cell; a goal
# at the bare chassis clearance from a leg is a goal under the table.
DEFAULT_PREFERRED_CLEARANCE_M = 0.8


@dataclass
class GridView:
    """Numpy-friendly view of nav_msgs/OccupancyGrid."""
    data: np.ndarray         # shape (h, w), int8 in [-1, 100]
    resolution: float        # m / cell
    origin_x: float
    origin_y: float
    width: int
    height: int

    @classmethod
    def from_msg(cls, msg) -> "GridView":
        h, w = int(msg.info.height), int(msg.info.width)
        arr = np.frombuffer(bytes(msg.data), dtype=np.int8).reshape(h, w)
        return cls(
            data=arr,
            resolution=float(msg.info.resolution),
            origin_x=float(msg.info.origin.position.x),
            origin_y=float(msg.info.origin.position.y),
            width=w, height=h,
        )

    def cell_to_world(self, cx: int, cy: int) -> Tuple[float, float]:
        return (self.origin_x + (cx + 0.5) * self.resolution,
                self.origin_y + (cy + 0.5) * self.resolution)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int((x - self.origin_x) / self.resolution),
                int((y - self.origin_y) / self.resolution))

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.width and 0 <= cy < self.height


@dataclass
class FrontierCluster:
    centroid_xy: Tuple[float, float]   # world coords
    size: int                          # cell count
    cell_indices: np.ndarray           # (N, 2) int — for debugging / viz
    # Where to actually drive. Held back from the centroid toward the robot
    # so the goal is in observed space rather than on the boundary. Reports
    # and overlays keep using centroid_xy: that is what was *found*, and the
    # two being separate is the point.
    goal_xy: Optional[Tuple[float, float]] = None
    # Length of the way to goal_xy when the strategy searched for one.
    path_m: Optional[float] = None

    @property
    def drive_to(self) -> Tuple[float, float]:
        """The pose to navigate to — the standoff point where one exists."""
        return self.goal_xy or self.centroid_xy


def fill_traversed(gv: GridView, cells) -> GridView:
    """`gv` with unknown cells in `cells` marked free.

    A lidar does not see the ground right around itself, so the patch the
    robot stands on can stay unknown in the map however long it stands
    there. That patch rings the robot with frontier cells that no goal can
    ever clear, and the explorer keeps picking them. Cells the robot's own
    body has occupied are free by demonstration. Occupied cells are left
    as they are."""
    data = gv.data.copy()
    for cx, cy in cells:
        if gv.in_bounds(cx, cy) and data[cy, cx] == -1:
            data[cy, cx] = 0
    return GridView(data=data, resolution=gv.resolution,
                    origin_x=gv.origin_x, origin_y=gv.origin_y,
                    width=gv.width, height=gv.height)


def find_frontier_cells(gv: GridView) -> np.ndarray:
    """Return (N, 2) array of (cx, cy) for cells that are free AND
    have at least one unknown 4-neighbour. Vectorised via shifted
    masks to avoid per-cell python loops."""
    g = gv.data
    free    = (g == 0)
    unknown = (g == -1)

    # Pad unknown by 1 in each direction; OR them and intersect with free.
    h, w = g.shape
    has_unknown_neighbour = np.zeros_like(free, dtype=bool)
    has_unknown_neighbour[1:, :]   |= unknown[:-1, :]   # neighbour above
    has_unknown_neighbour[:-1, :]  |= unknown[1:, :]    # below
    has_unknown_neighbour[:, 1:]   |= unknown[:, :-1]   # left
    has_unknown_neighbour[:, :-1]  |= unknown[:, 1:]    # right

    frontier_mask = free & has_unknown_neighbour
    yy, xx = np.where(frontier_mask)
    return np.stack([xx, yy], axis=1)  # (N, 2) as (cx, cy)


def cluster_frontiers(cells: np.ndarray, min_size: int = 3,
                       max_link_cells: int = 2) -> List[FrontierCluster]:
    """Connected-components style clustering with 8-neighbour adjacency
    extended by `max_link_cells` (cells within this Chebyshev distance
    are merged into the same cluster). This is cheaper than a real
    DBSCAN since we already have integer grid coords.

    Drops clusters smaller than `min_size` cells — those are usually
    noise from boundary cells against partially-mapped obstacles.
    """
    if cells.size == 0:
        return []

    # Bucket into a sparse grid for fast neighbour lookup.
    cell_set = {(int(c[0]), int(c[1])): i for i, c in enumerate(cells)}
    parent = list(range(len(cells)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    r = max_link_cells
    for (cx, cy), idx in cell_set.items():
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (cx + dx, cy + dy)
                j = cell_set.get(nb)
                if j is not None:
                    union(idx, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(cells)):
        groups.setdefault(find(i), []).append(i)

    clusters: List[FrontierCluster] = []
    for _, members in groups.items():
        if len(members) < min_size:
            continue
        member_arr = cells[members]
        # Centroid in world frame computed by caller (needs GridView);
        # here we just produce cell-space mean and let the caller
        # convert.
        clusters.append(FrontierCluster(
            centroid_xy=(float(member_arr[:, 0].mean()),
                         float(member_arr[:, 1].mean())),  # cell-space, will convert
            size=len(members),
            cell_indices=member_arr,
        ))
    return clusters


def is_target_safe(gv: GridView, wx: float, wy: float,
                    safe_radius_m: float = DEFAULT_CLEARANCE_M) -> bool:
    """Reject targets that sit inside or near an obstacle. Single
    check: every cell in a `safe_radius_m`-radius patch around the
    target must be NON-OCCUPIED (g < OCC_THRESH). The radius defaults
    to the chassis itself (nav2's `robot_radius`) — it was 0.15, below
    the 0.22 nav2 runs with, so a goal could pass this test with the
    robot's own body overlapping an obstacle. The radius is rounded up
    to whole cells, never down. Unknown cells (-1)
    are allowed — frontier centroids sit on the free/unknown boundary
    by construction, so requiring "mostly known" at the exact centroid
    deadlocks exploration ("no safe frontier" forever even when 7+
    legitimate clusters exist; observed on webots tiago).

    The nav service's costmap layer is the second line of defence
    against inflation-halo violations — if the actual planner can't
    route to the chosen frontier, the goal aborts and the controller
    falls through to the next candidate.
    """
    cx, cy = gv.world_to_cell(wx, wy)
    if not gv.in_bounds(cx, cy):
        return False
    r = max(1, int(math.ceil(safe_radius_m / gv.resolution - 1e-9)))
    y0, y1 = max(0, cy - r), min(gv.height, cy + r + 1)
    x0, x1 = max(0, cx - r), min(gv.width,  cx + r + 1)
    patch = gv.data[y0:y1, x0:x1]
    # A disc, not the bounding square. A square of half-width r reaches
    # r*sqrt(2) into its corners, so the test was quietly demanding 41% more
    # room than it named -- which went unnoticed while the number was 0.15
    # (0.21 at the corners, about the chassis) and turned down every frontier
    # in the room the moment it was raised to the chassis itself. The robot's
    # footprint is a circle and nav2 plans for a circle; so does this.
    yy, xx = np.ogrid[y0 - cy:y1 - cy, x0 - cx:x1 - cx]
    within = (yy * yy + xx * xx) <= r * r
    return bool(np.all(patch[within] < OCC_THRESH))


def clearance_at(gv: GridView, wx: float, wy: float,
                  max_m: float) -> float:
    """Distance from (wx, wy) to the nearest occupied cell, capped at
    `max_m`. Unknown cells do not count: the goal is held back from the
    frontier, and the frontier is unknown on one side by definition."""
    cx, cy = gv.world_to_cell(wx, wy)
    r = max(1, int(math.ceil(max_m / gv.resolution)))
    y0, y1 = max(0, cy - r), min(gv.height, cy + r + 1)
    x0, x1 = max(0, cx - r), min(gv.width,  cx + r + 1)
    oy, ox = np.nonzero(gv.data[y0:y1, x0:x1] >= OCC_THRESH)
    if oy.size == 0:
        return max_m
    d2 = (ox + x0 - cx) ** 2 + (oy + y0 - cy) ** 2
    return min(max_m, float(np.sqrt(d2.min())) * gv.resolution)


def approach_point(gv: GridView,
                    robot_xy: Tuple[float, float],
                    target_xy: Tuple[float, float],
                    *,
                    standoff_m: float,
                    clearance_m: float,
                    preferred_clearance_m: float = DEFAULT_PREFERRED_CLEARANCE_M,
                    keepout: Optional[List[Tuple[float, float, float]]] = None
                    ) -> Optional[Tuple[float, float]]:
    """A point near `target_xy`, on the line back to the robot, that fits.

    Walks the line from `standoff_m` -- further back first, since that is
    more observed ground, then closer in -- and takes the point with the
    most room around it, up to `preferred_clearance_m`. Ties go to the
    earlier point, so in open ground the nominal standoff is kept. Returns
    None only when no point on the segment clears `clearance_m`, which is a
    frontier genuinely unreachable rather than one whose single sampled
    point happened to be against a wall.

    Testing one point and discarding the cluster on failure is what left the
    explorer spinning with two dozen clusters in view; taking the first point
    that fits is what parked it beside table legs.
    """
    offsets = [standoff_m]
    step = _STANDOFF_STEP_M
    for i in range(1, _MAX_EXTRA_STANDOFF_STEPS + 1):
        offsets.append(standoff_m + i * step)
        # Closer in, but never onto the frontier itself: offset 0 is the
        # centroid, which sits next to unknown space.
        if i <= 4 and standoff_m - i * step > 0.05:
            offsets.append(standoff_m - i * step)
    seen: set[int] = set()
    best, best_room = None, -1.0
    for offset in offsets:
        key = int(round(offset * 100))
        if key in seen:
            continue
        seen.add(key)
        px, py = standoff_point(robot_xy, target_xy, offset)
        if not is_target_safe(gv, px, py, safe_radius_m=clearance_m):
            continue
        if in_keepout(px, py, keepout):
            continue
        room = clearance_at(gv, px, py, preferred_clearance_m)
        if room > best_room:
            best, best_room = (px, py), room
        if room >= preferred_clearance_m:
            break
    return best


def standoff_point(robot_xy: Tuple[float, float],
                    target_xy: Tuple[float, float],
                    standoff_m: float) -> Tuple[float, float]:
    """Pull `target_xy` back toward the robot by `standoff_m`.

    Returns the target unchanged when the robot is already closer than the
    standoff: there is nothing to hold back from, and retracting past the
    robot would send it backwards away from the frontier it is meant to
    observe.
    """
    dx, dy = target_xy[0] - robot_xy[0], target_xy[1] - robot_xy[1]
    distance = (dx * dx + dy * dy) ** 0.5
    if distance <= standoff_m or distance <= 1e-6:
        return target_xy
    scale = (distance - standoff_m) / distance
    return (robot_xy[0] + dx * scale, robot_xy[1] + dy * scale)


def in_keepout(x: float, y: float,
                keepout: Optional[List[Tuple[float, float, float]]]) -> bool:
    """Whether (x, y) falls inside any (cx, cy, radius) circle.

    These come from whatever can see what the grid cannot. A lidar at
    chassis height does not see a tabletop, so a table is free space in the
    occupancy grid and the robot drives into it; the only way to avoid one
    is for something with a different view to name it. This module takes
    that as data and asks no questions about where it came from.
    """
    if not keepout:
        return False
    for cx, cy, radius in keepout:
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
            return True
    return False


def anchor_cell(c: FrontierCluster) -> Tuple[int, int]:
    """Return the frontier cell of `c` closest to its cell-space mean.

    The mean of a curved or ring-shaped cluster can fall inside the
    unknown region or a wall it wraps around. Every member cell is
    known free by construction, so a frontier anchored on one is never
    in unknown space."""
    cells = c.cell_indices
    mx, my = c.centroid_xy
    d2 = (cells[:, 0] - mx) ** 2 + (cells[:, 1] - my) ** 2
    best = cells[int(np.argmin(d2))]
    return int(best[0]), int(best[1])


def score_clusters(clusters: List[FrontierCluster], gv: GridView,
                    robot_xy: Tuple[float, float], *,
                    max_distance_m: float = 8.0,
                    visited_cells: Optional[set] = None,
                    visited_penalty_m: float = 1.5,
                    clearance_m: float = DEFAULT_CLEARANCE_M,
                    preferred_clearance_m: float = DEFAULT_PREFERRED_CLEARANCE_M,
                    standoff_m: float = DEFAULT_STANDOFF_M,
                    keepout: Optional[List[Tuple[float, float, float]]] = None,
                    blocked_xy: Optional[Sequence[Tuple[float, float]]] = None,
                    blocked_radius_m: float = 1.0
                    ) -> List[Tuple[float, FrontierCluster]]:
    """Score frontiers and rank descending. Score formula:

        score = info_gain / (travel + visited_penalty + 1)
                * min(clearance, preferred_clearance) / preferred_clearance

    With these guards:
      - travel > max_distance_m → cluster dropped entirely (local
        preference: don't try to teleport across a multi-room map).
      - frontier within blocked_radius_m of a blocked_xy point →
        dropped. The controller records frontiers navigation failed to
        reach; the cluster moves a few cm between map updates, so an
        exact match would never fire.
      - the standoff point, not the centroid, is what gets checked and
        driven to: the centroid is on the free/unknown boundary by
        construction, and stopping there is how the robot ends up with
        its nose in whatever the boundary was hiding.
      - standoff point inside lethal halo → dropped (is_target_safe()).
      - standoff point inside a keep-out circle → dropped. Those name
        obstacles the grid does not contain, a table under a
        chassis-height lidar being the case this was written for.
      - centroid in/near a visited cell → travel penalty added so
        re-visiting unexplored fringes is preferred.

    The frontier position of a cluster is its anchor_cell(), returned
    as centroid_xy of the world-frame cluster.

    visited_cells is a set of (cx, cy) cell-space coordinates the
    skill has already driven through; the controller maintains it.
    """
    blocked_r2 = blocked_radius_m * blocked_radius_m
    scored = []
    for c in clusters:
        wx, wy = gv.cell_to_world(*anchor_cell(c))
        c_world = FrontierCluster(centroid_xy=(wx, wy),
                                  size=c.size,
                                  cell_indices=c.cell_indices)
        travel = ((wx - robot_xy[0]) ** 2 + (wy - robot_xy[1]) ** 2) ** 0.5
        if travel > max_distance_m:
            continue                             # too far — skip
        if blocked_xy and any((wx - bx) ** 2 + (wy - by) ** 2 <= blocked_r2
                              for bx, by in blocked_xy):
            continue                             # nav already failed here
        # Checked where the robot will stand, not where the frontier is,
        # and along the whole line rather than at one point on it.
        approach = approach_point(gv, robot_xy, (wx, wy),
                                   standoff_m=standoff_m,
                                   clearance_m=clearance_m,
                                   preferred_clearance_m=preferred_clearance_m,
                                   keepout=keepout)
        if approach is None:
            continue                 # nowhere on the line fits — skip
        c_world.goal_xy = approach

        penalty = 0.0
        if visited_cells:
            tcx, tcy = gv.world_to_cell(wx, wy)
            radius = max(1, int(round(visited_penalty_m / gv.resolution)))
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if (tcx + dx, tcy + dy) in visited_cells:
                        penalty = visited_penalty_m
                        break
                if penalty:
                    break

        # A cramped goal is worth less than a roomy one: between two
        # frontiers of equal promise, go where the robot keeps its distance.
        room = clearance_at(gv, approach[0], approach[1],
                            preferred_clearance_m)
        score = (c_world.size / (travel + penalty + 1.0)
                 * room / preferred_clearance_m)
        scored.append((score, c_world))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def pick_target(gv: GridView, robot_xy: Tuple[float, float], *,
                 min_size: int = 3,
                 max_distance_m: float = 8.0,
                 visited_cells: Optional[set] = None,
                 clearance_m: float = DEFAULT_CLEARANCE_M,
                 preferred_clearance_m: float = DEFAULT_PREFERRED_CLEARANCE_M,
                 standoff_m: float = DEFAULT_STANDOFF_M,
                 keepout: Optional[List[Tuple[float, float, float]]] = None,
                 blocked_xy: Optional[Sequence[Tuple[float, float]]] = None,
                 blocked_radius_m: float = 1.0
                 ) -> Optional[FrontierCluster]:
    """End-to-end convenience. Returns None if no SAFE frontier in
    range — caller may declare done."""
    cells = find_frontier_cells(gv)
    if cells.size == 0:
        return None
    clusters = cluster_frontiers(cells, min_size=min_size)
    if not clusters:
        return None
    scored = score_clusters(clusters, gv, robot_xy,
                             max_distance_m=max_distance_m,
                             visited_cells=visited_cells,
                             clearance_m=clearance_m,
                             preferred_clearance_m=preferred_clearance_m,
                             standoff_m=standoff_m,
                             keepout=keepout,
                             blocked_xy=blocked_xy,
                             blocked_radius_m=blocked_radius_m)
    return scored[0][1] if scored else None


def total_frontier_count(gv: GridView, min_size: int = 3) -> int:
    """For status reporting: how many frontier clusters remain that
    are large enough to warrant another exploration round."""
    cells = find_frontier_cells(gv)
    if cells.size == 0:
        return 0
    return len(cluster_frontiers(cells, min_size=min_size))


def mapped_free_area_m2(gv: GridView) -> float:
    """How much area (m²) is currently mapped as free."""
    free_cells = int(np.sum(gv.data == 0))
    return free_cells * (gv.resolution ** 2)
