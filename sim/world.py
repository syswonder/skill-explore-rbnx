# SPDX-License-Identifier: MulanPSL-2.0
"""Ground-truth worlds.

A world is a grid of what is really there. Each kind of cell answers two
questions differently: does the lidar see it, and does it stop the robot.

    FREE      seen through, passable
    WALL      seen, blocks            walls, furniture down to the floor
    OVERHANG  not seen, blocks        a tabletop above the lidar's scan plane
    GLASS     not seen, blocks        the lidar passes through it

The tables are the case the robot keeps hitting: the lidar returns only
the four legs, the top blocks the body.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

FREE, WALL, OVERHANG, GLASS = 0, 1, 2, 3


@dataclass
class World:
    name: str
    cells: np.ndarray                 # (h, w) uint8, one of the kinds above
    resolution: float
    start: Tuple[float, float, float]  # x, y, yaw
    # Scene's view of the furniture, as (x, y, size_x, size_y) boxes.
    objects: List[Tuple[float, float, float, float]] = field(
        default_factory=list)

    @property
    def height(self) -> int:
        return self.cells.shape[0]

    @property
    def width(self) -> int:
        return self.cells.shape[1]

    def to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(np.floor(x / self.resolution)),
                int(np.floor(y / self.resolution)))

    def blocks_robot(self) -> np.ndarray:
        return self.cells != FREE

    def seen_by_lidar(self) -> np.ndarray:
        return self.cells == WALL


class _Builder:
    def __init__(self, w_m: float, h_m: float, res: float = 0.05):
        self.res = res
        self.cells = np.zeros((int(round(h_m / res)), int(round(w_m / res))),
                              dtype=np.uint8)
        self.objects: List[Tuple[float, float, float, float]] = []

    def _c(self, v: float) -> int:
        return int(round(v / self.res))

    def box(self, x0, y0, x1, y1, kind=WALL):
        self.cells[self._c(y0):self._c(y1), self._c(x0):self._c(x1)] = kind

    def hwall(self, y, x0, x1, kind=WALL, t=0.1):
        self.box(x0, y, x1, y + t, kind)

    def vwall(self, x, y0, y1, kind=WALL, t=0.1):
        self.box(x, y0, x + t, y1, kind)

    def border(self):
        h, w = self.cells.shape
        self.hwall(0, 0, w * self.res)
        self.hwall(h * self.res - 0.1, 0, w * self.res)
        self.vwall(0, 0, h * self.res)
        self.vwall(w * self.res - 0.1, 0, h * self.res)

    def table(self, x, y, sx=1.2, sy=0.7, leg=0.05):
        """Top (invisible, blocks) with four legs (visible) at the corners."""
        self.box(x - sx / 2, y - sy / 2, x + sx / 2, y + sy / 2, OVERHANG)
        for lx in (x - sx / 2, x + sx / 2 - leg):
            for ly in (y - sy / 2, y + sy / 2 - leg):
                self.box(lx, ly, lx + leg, ly + leg, WALL)
        self.objects.append((x, y, sx, sy))

    def build(self, name, start) -> World:
        return World(name=name, cells=self.cells, resolution=self.res,
                     start=start, objects=self.objects)


def office() -> World:
    """Three rooms off a corridor, doors 0.9 m, tables and a cabinet."""
    b = _Builder(16.0, 12.0)
    b.border()
    b.hwall(5.0, 0, 16)                       # corridor below, rooms above
    for x0, x1 in ((1.5, 2.4), (7.0, 7.9), (12.5, 13.4)):
        b.box(x0, 5.0, x1, 5.1, FREE)         # doors
    b.vwall(5.5, 5.0, 12.0)
    b.vwall(10.5, 5.0, 12.0)
    b.table(2.8, 8.5)
    b.table(8.0, 9.5, sx=1.6, sy=0.8)
    b.table(13.0, 8.0)
    b.table(13.0, 10.5, sx=0.8, sy=0.8)
    b.box(6.0, 11.2, 7.5, 11.9)               # cabinet against the wall
    b.box(3.0, 1.0, 3.6, 1.6)                 # pillar in the corridor
    return b.build("office", (1.0, 2.5, 0.0))


def corridor() -> World:
    """A 25 m corridor: the far end is beyond the 6 m local radius."""
    b = _Builder(26.0, 4.0)
    b.border()
    b.box(8.0, 0.1, 8.4, 1.2)                 # alcove posts
    b.box(16.0, 2.8, 16.4, 3.9)
    return b.build("corridor", (1.0, 2.0, 0.0))


def tables() -> World:
    """One room packed with tables: goals near legs are under tabletops."""
    b = _Builder(10.0, 8.0)
    b.border()
    for i, x in enumerate((2.5, 5.0, 7.5)):
        for y in (2.2, 5.6):
            b.table(x, y + (0.3 if i == 1 else 0.0))
    return b.build("tables", (0.8, 4.0, 0.0))


def glass() -> World:
    """A glass wall the lidar sees through splits the room."""
    b = _Builder(10.0, 8.0)
    b.border()
    b.vwall(5.0, 0.0, 8.0, GLASS)
    b.box(5.0, 3.6, 5.1, 4.5, FREE)           # opening in the glass
    return b.build("glass", (1.5, 2.0, 0.0))


def house() -> World:
    """Eight rooms on two sides of a hall, some reached only through
    another: greedy choices leave rooms behind and walk back for them."""
    b = _Builder(20.0, 14.0)
    b.border()
    b.hwall(5.0, 0, 20)
    b.hwall(9.0, 0, 20)
    for x in (5.0, 10.0, 15.0):
        b.vwall(x, 0.0, 5.0)
        b.vwall(x, 9.0, 14.0)
    for x0 in (2.0, 12.0, 17.0):                # doors from the hall, south
        b.box(x0, 5.0, x0 + 0.9, 5.1, FREE)
    b.box(7.0, 2.0, 7.1, 2.0, FREE)
    b.box(10.0, 2.0, 10.1, 2.9, FREE)            # room 2 only through room 3
    for x0 in (1.5, 6.5, 11.5, 16.5):           # doors from the hall, north
        b.box(x0, 9.0, x0 + 0.9, 9.1, FREE)
    b.table(2.5, 12.0)
    b.table(12.5, 11.5, sx=0.8, sy=0.8)
    b.table(7.5, 1.5)
    b.box(18.0, 1.0, 19.5, 1.6)                  # sofa
    return b.build("house", (1.0, 7.0, 0.0))


def warehouse() -> World:
    """Rows of shelving in a 30 x 20 m hall."""
    b = _Builder(30.0, 20.0)
    b.border()
    for y in (4.0, 8.0, 12.0, 16.0):
        b.box(4.0, y, 14.0, y + 0.8)
        b.box(16.0, y, 26.0, y + 0.8)
    b.box(14.8, 9.0, 15.2, 11.0)                  # pillar in the cross aisle
    return b.build("warehouse", (2.0, 2.0, 0.0))


WORLDS: Dict[str, Callable[[], World]] = {
    "office": office, "corridor": corridor, "tables": tables, "glass": glass,
    "house": house, "warehouse": warehouse,
}
