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

from dataclasses import dataclass
from typing import List, Optional, Tuple

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

    @property
    def drive_to(self) -> Tuple[float, float]:
        """The pose to navigate to — the standoff point where one exists."""
        return self.goal_xy or self.centroid_xy


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
    to the chassis plus a margin — it was 0.15, below the 0.22
    `robot_radius` nav2 runs with, so a goal could pass this test with
    the robot's own body overlapping an obstacle. Unknown cells (-1)
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
    r = max(1, int(round(safe_radius_m / gv.resolution)))
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


def approach_point(gv: GridView,
                    robot_xy: Tuple[float, float],
                    target_xy: Tuple[float, float],
                    *,
                    standoff_m: float,
                    clearance_m: float,
                    keepout: Optional[List[Tuple[float, float, float]]] = None
                    ) -> Optional[Tuple[float, float]]:
    """A point near `target_xy`, on the line back to the robot, that fits.

    Starts at `standoff_m` and walks the line -- further back first, since
    that is more observed ground, then closer in. Returns None only when no
    point on the segment clears, which is a frontier genuinely unreachable
    rather than one whose single sampled point happened to be against a wall.

    Testing one point and discarding the cluster on failure is what left the
    explorer spinning with two dozen clusters in view.
    """
    offsets = [standoff_m]
    step = _STANDOFF_STEP_M
    for i in range(1, 5):
        offsets.append(standoff_m + i * step)
        offsets.append(max(0.0, standoff_m - i * step))
    seen: set[int] = set()
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
        return (px, py)
    return None


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


def score_clusters(clusters: List[FrontierCluster], gv: GridView,
                    robot_xy: Tuple[float, float], *,
                    max_distance_m: float = 8.0,
                    visited_cells: Optional[set] = None,
                    visited_penalty_m: float = 1.5,
                    clearance_m: float = DEFAULT_CLEARANCE_M,
                    standoff_m: float = DEFAULT_STANDOFF_M,
                    keepout: Optional[List[Tuple[float, float, float]]] = None
                    ) -> List[Tuple[float, FrontierCluster]]:
    """Score frontiers and rank descending. Score formula:

        score = info_gain / (travel + visited_penalty + 1)

    With these guards:
      - travel > max_distance_m → cluster dropped entirely (local
        preference: don't try to teleport across a multi-room map).
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

    visited_cells is a set of (cx, cy) cell-space coordinates the
    skill has already driven through; the controller maintains it.
    """
    scored = []
    for c in clusters:
        cx, cy = c.centroid_xy
        wx, wy = gv.cell_to_world(int(round(cx)), int(round(cy)))
        c_world = FrontierCluster(centroid_xy=(wx, wy),
                                  size=c.size,
                                  cell_indices=c.cell_indices)
        travel = ((wx - robot_xy[0]) ** 2 + (wy - robot_xy[1]) ** 2) ** 0.5
        if travel > max_distance_m:
            continue                             # too far — skip
        # Checked where the robot will stand, not where the frontier is,
        # and along the whole line rather than at one point on it.
        approach = approach_point(gv, robot_xy, (wx, wy),
                                   standoff_m=standoff_m,
                                   clearance_m=clearance_m,
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

        score = c_world.size / (travel + penalty + 1.0)
        scored.append((score, c_world))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def pick_target(gv: GridView, robot_xy: Tuple[float, float], *,
                 min_size: int = 3,
                 max_distance_m: float = 8.0,
                 visited_cells: Optional[set] = None,
                 clearance_m: float = DEFAULT_CLEARANCE_M,
                 standoff_m: float = DEFAULT_STANDOFF_M,
                 keepout: Optional[List[Tuple[float, float, float]]] = None
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
                             standoff_m=standoff_m,
                             keepout=keepout)
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
