# SPDX-License-Identifier: MulanPSL-2.0
"""Lidar ray casting and the occupancy map built from it."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .robot import RobotParams
from .world import World

UNKNOWN, FREE_V, OCC_V = -1, 0, 100


class Mapper:
    """What RTAB-Map publishes on /map: a 2D grid at the world resolution,
    free along each ray up to its return, occupied at the return. Rays
    that hit nothing within range mark nothing, and nothing inside the
    blind radius is marked free."""

    def __init__(self, world: World, robot: RobotParams):
        self.world = world
        self.robot = robot
        self.grid = np.full(world.cells.shape, UNKNOWN, dtype=np.int8)
        self._seen = world.seen_by_lidar()
        angles = np.linspace(0, 2 * np.pi, robot.lidar_rays, endpoint=False)
        step = world.resolution / 2
        self._dists = np.arange(step, robot.lidar_range_m, step)
        self._cos = np.cos(angles)[:, None]
        self._sin = np.sin(angles)[:, None]

    def scan(self, x: float, y: float) -> np.ndarray:
        """Integrate one scan at (x, y). Returns the (cx, cy) cells this
        scan saw as free, for the costmap's raytrace clearing."""
        w = self.world
        res = w.resolution
        px = x + self._cos * self._dists
        py = y + self._sin * self._dists
        cx = np.floor(px / res).astype(int)
        cy = np.floor(py / res).astype(int)
        inside = (cx >= 0) & (cx < w.width) & (cy >= 0) & (cy < w.height)
        cxc = np.clip(cx, 0, w.width - 1)
        cyc = np.clip(cy, 0, w.height - 1)
        hit = self._seen[cyc, cxc] & inside
        n = self._dists.size
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), n)
        idx = np.arange(n)[None, :]
        before = (idx < first[:, None]) & inside & (
            self._dists[None, :] >= self.robot.blind_radius_m)
        free_cy, free_cx = cyc[before], cxc[before]
        has_hit = first < n
        rows = np.nonzero(has_hit)[0]
        hit_cy = cyc[rows, first[rows]]
        hit_cx = cxc[rows, first[rows]]
        self.grid[free_cy, free_cx] = FREE_V
        self.grid[hit_cy, hit_cx] = OCC_V
        return np.stack([free_cx, free_cy], axis=1)

    def msg(self) -> SimpleNamespace:
        """An OccupancyGrid-shaped message, as the controller receives it."""
        origin = SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))
        info = SimpleNamespace(width=self.world.width,
                               height=self.world.height,
                               resolution=self.world.resolution,
                               origin=origin)
        return SimpleNamespace(info=info, data=self.grid.tobytes())
