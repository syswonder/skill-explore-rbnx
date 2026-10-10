# SPDX-License-Identifier: MulanPSL-2.0
"""Write a run as a PNG with only numpy and zlib."""
from __future__ import annotations

import struct
import zlib

import numpy as np

from .run import Result
from .sensing import FREE_V, OCC_V
from .world import GLASS, OVERHANG, World


def _png(path: str, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes()
                   for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def render(path: str, world: World, grid: np.ndarray, result: Result,
           scale: int = 3) -> None:
    """Grey unknown, white free, black mapped obstacle; orange tabletops
    and cyan glass from the ground truth; blue trajectory; green goals
    that succeeded, red goals that did not; magenta collisions."""
    img = np.full(grid.shape + (3,), 160, dtype=np.uint8)
    img[grid == FREE_V] = (255, 255, 255)
    img[grid >= OCC_V] = (0, 0, 0)
    img[world.cells == OVERHANG] = (255, 170, 60)
    img[world.cells == GLASS] = (60, 200, 230)
    res = world.resolution

    def dot(x, y, color, r=1):
        cx, cy = int(x / res), int(y / res)
        img[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1] = color

    for x, y in result.trajectory:
        dot(x, y, (40, 90, 230), 0)
    for x, y, outcome in result.goals:
        dot(x, y, (30, 170, 60) if "SUCCEEDED" in outcome else (220, 40, 40), 2)
    for x, y in result.collision_points:
        dot(x, y, (230, 0, 230), 3)
    img = img[::-1]                     # y up
    img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    _png(path, img)
