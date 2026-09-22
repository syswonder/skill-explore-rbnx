# SPDX-License-Identifier: MulanPSL-2.0
"""Finding somewhere near a frontier that the robot actually fits.

Written after the first version of the standoff left the explorer spinning
in place with two dozen frontier clusters in view. It computed one point on
the line back to the robot, tested it, and discarded the whole cluster when
that single point was against a wall. Measured on a live map: every nearby
frontier turned down, zero targets, and the loop fell through to its
spin-and-retry fallback forever.

The goal was never "stand exactly 0.45 m back". It is "get near this
frontier somewhere the robot fits", and the line from the frontier to the
robot is a row of candidates, not one.
"""
import numpy as np
import pytest

from explore_skill.frontier import (DEFAULT_CLEARANCE_M, GridView,
                                    ROBOT_RADIUS_M, approach_point)

FREE, UNKNOWN, WALL = 0, -1, 100


def _grid(rows, resolution=0.05):
    data = np.array(rows, dtype=np.int8)
    h, w = data.shape
    return GridView(data=data, resolution=resolution, origin_x=0.0,
                    origin_y=0.0, width=w, height=h)


def test_the_clearance_is_the_chassis_and_not_more():
    """nav2 plans this goal with the same radius and inflates on top of it.
    Asking for chassis-plus-margin here bills that margin twice, and on a
    furnished map it rejected everything."""
    assert DEFAULT_CLEARANCE_M == ROBOT_RADIUS_M


def test_a_blocked_standoff_walks_the_line_instead_of_giving_up():
    """The regression. A wall exactly where the nominal standoff lands used
    to lose the frontier; now the search steps along the line and finds the
    room that is there."""
    rows = [[FREE] * 60 for _ in range(60)]
    # A one-cell wall crossing the line between robot and frontier, at the
    # column the nominal 0.45 m standoff lands on.
    for y in range(60):
        rows[y][29] = WALL
    gv = _grid(rows)
    robot = (0.30, 1.50)
    frontier = (2.20, 1.50)

    found = approach_point(gv, robot, frontier, standoff_m=0.45,
                           clearance_m=DEFAULT_CLEARANCE_M)
    assert found is not None
    # Wherever it landed, the chassis has to fit there.
    from explore_skill.frontier import is_target_safe
    assert is_target_safe(gv, found[0], found[1],
                          safe_radius_m=DEFAULT_CLEARANCE_M)


def test_it_still_prefers_standing_back_from_the_boundary():
    """Open ground: the nominal standoff is taken, because the point of the
    standoff is to be in space that has already been observed."""
    rows = [[FREE] * 80 for _ in range(80)]
    gv = _grid(rows)
    found = approach_point(gv, (0.30, 1.50), (2.20, 1.50), standoff_m=0.45,
                           clearance_m=DEFAULT_CLEARANCE_M)
    assert found == pytest.approx((2.20 - 0.45, 1.50))


def test_a_frontier_with_no_room_anywhere_is_still_refused():
    """The search is not a way to say yes eventually. A frontier the robot
    cannot reach safely at any offset is one to skip."""
    rows = [[WALL] * 40 for _ in range(40)]
    rows[20][20] = FREE
    gv = _grid(rows)
    assert approach_point(gv, (0.50, 1.00), (1.05, 1.00), standoff_m=0.45,
                          clearance_m=DEFAULT_CLEARANCE_M) is None


def test_keepout_applies_at_every_offset_not_just_the_first():
    """Otherwise walking the line becomes a way around the circles that
    exist because the lidar cannot see the furniture."""
    rows = [[FREE] * 80 for _ in range(80)]
    gv = _grid(rows)
    robot, frontier = (0.30, 1.50), (2.20, 1.50)
    # A circle covering the whole segment the search walks.
    blanket = [((robot[0] + frontier[0]) / 2, 1.50, 1.60)]
    assert approach_point(gv, robot, frontier, standoff_m=0.45,
                          clearance_m=DEFAULT_CLEARANCE_M,
                          keepout=blanket) is None
