# SPDX-License-Identifier: MulanPSL-2.0
"""Where a frontier target is allowed to be.

Written after the robot drove into a wall on a map where nothing was wrong:
the clearance demanded of a goal was 0.15 m against nav2 configured with
`robot_radius: 0.22`, so a goal could pass with the chassis overlapping an
obstacle. The other half is not a constant at all -- a frontier centroid is
on the free/unknown boundary by construction, so "drive to the centroid" is
"drive to the edge of what has been seen", every single time.

These pin the clearance against the chassis, the standoff that keeps the
goal inside observed space, and the keep-out circles that exist because a
chassis-height lidar cannot see a tabletop and so the grid never will.
"""
import numpy as np
import pytest

from explore_skill.frontier import (DEFAULT_CLEARANCE_M, GridView,
                                    ROBOT_RADIUS_M, in_keepout,
                                    is_target_safe, pick_target,
                                    standoff_point)

FREE, UNKNOWN, WALL = 0, -1, 100


def _grid(rows, resolution=0.05):
    data = np.array(rows, dtype=np.int8)
    h, w = data.shape
    return GridView(data=data, resolution=resolution, origin_x=0.0,
                    origin_y=0.0, width=w, height=h)


def test_the_clearance_is_at_least_the_robot():
    """A clearance below `robot_radius` is not a safety check. This is the
    regression itself: the default was 0.15 against a 0.22 chassis."""
    assert DEFAULT_CLEARANCE_M >= ROBOT_RADIUS_M


def test_a_goal_the_chassis_does_not_fit_is_rejected():
    """40x40 of free space with a wall down one column: a point one cell
    from that wall clears 0.15 m but not the robot."""
    rows = [[FREE] * 40 for _ in range(40)]
    for y in range(40):
        rows[y][20] = WALL
    gv = _grid(rows)
    # The old radius spans 3 cells, the chassis 6. Cell 16 reaches cell 19
    # under the first and cell 22 under the second, so the wall at 20 is
    # invisible to one check and fatal to the other.
    near_wall_x = (16 + 0.5) * 0.05
    assert is_target_safe(gv, near_wall_x, 1.0, safe_radius_m=0.15)
    assert not is_target_safe(gv, near_wall_x, 1.0)


def test_the_standoff_pulls_the_goal_back_toward_the_robot():
    """Not past it: retracting further than the robot would send it away
    from the frontier it is there to observe."""
    assert standoff_point((0.0, 0.0), (2.0, 0.0), 0.45) == pytest.approx(
        (1.55, 0.0))
    # Already closer than the standoff — nothing to hold back from.
    assert standoff_point((0.0, 0.0), (0.3, 0.0), 0.45) == (0.3, 0.0)
    # Degenerate: robot already on the target.
    assert standoff_point((1.0, 1.0), (1.0, 1.0), 0.45) == (1.0, 1.0)


def test_the_goal_lands_short_of_the_frontier_it_names():
    """The whole point: `centroid_xy` still reports what was found, and
    `drive_to` is somewhere the robot has already seen."""
    # Left half observed free, right half unknown: the frontier runs down
    # the middle, which is exactly where the robot used to be sent.
    rows = [[FREE] * 30 + [UNKNOWN] * 30 for _ in range(40)]
    gv = _grid(rows)
    target = pick_target(gv, robot_xy=(0.2, 1.0))
    assert target is not None
    assert target.drive_to != target.centroid_xy
    # Held back along the line from the robot, so nearer the robot than the
    # frontier is.
    assert target.drive_to[0] < target.centroid_xy[0]


def test_a_keepout_circle_drops_a_target_the_grid_calls_free():
    """A table is free space to a lidar that passes under it. Nothing in
    the grid will ever say otherwise, so something else has to."""
    rows = [[FREE] * 30 + [UNKNOWN] * 30 for _ in range(40)]
    gv = _grid(rows)
    clear = pick_target(gv, robot_xy=(0.2, 1.0))
    assert clear is not None

    # A circle covering where that goal landed.
    gx, gy = clear.drive_to
    assert in_keepout(gx, gy, [(gx, gy, 0.5)])
    blocked = pick_target(gv, robot_xy=(0.2, 1.0),
                          keepout=[(gx, gy, 0.5)])
    assert blocked is None or blocked.drive_to != clear.drive_to


def test_keepout_is_inert_when_nothing_is_named():
    """The common case pays nothing for the uncommon one."""
    assert not in_keepout(1.0, 1.0, None)
    assert not in_keepout(1.0, 1.0, [])
