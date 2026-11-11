# SPDX-License-Identifier: MulanPSL-2.0
"""Frontier search and the goals it hands to navigation (frontier.py)."""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from explore_skill.frontier import (DEFAULT_CLEARANCE_M, GridView,
                                    ROBOT_RADIUS_M, approach_point,
                                    clearance_at, fill_traversed,
                                    find_frontier_cells, is_target_safe,
                                    overlay_lethal, pick_target)

FREE, UNKNOWN, WALL = 0, -1, 100


def _grid(rows, resolution=0.05):
    data = np.array(rows, dtype=np.int8)
    h, w = data.shape
    return GridView(data=data, resolution=resolution, origin_x=0.0,
                    origin_y=0.0, width=w, height=h)


def _half_unknown():
    """Left half free, right half unknown: one frontier down the middle."""
    return _grid([[FREE] * 30 + [UNKNOWN] * 30 for _ in range(40)])


def _two_pockets():
    """A 10 m room at 0.1 m/cell with two unknown pockets."""
    data = np.zeros((100, 100), dtype=np.int8)
    data[20:30, 60:70] = UNKNOWN               # centre (6.5, 2.5)
    data[70:80, 60:70] = UNKNOWN               # centre (6.5, 7.5)
    return GridView(data=data, resolution=0.1, origin_x=0.0, origin_y=0.0,
                    width=100, height=100)


# Goals

def test_the_clearance_is_the_chassis():
    """nav2 inflates on top of its own robot radius, so asking for more
    here would bill the margin twice."""
    assert DEFAULT_CLEARANCE_M == ROBOT_RADIUS_M


def test_a_goal_the_chassis_does_not_fit_is_rejected():
    rows = [[FREE] * 40 for _ in range(40)]
    for y in range(40):
        rows[y][20] = WALL
    gv = _grid(rows)
    near_wall_x = (16 + 0.5) * 0.05           # 0.15 m clear, not a chassis
    assert is_target_safe(gv, near_wall_x, 1.0, safe_radius_m=0.15)
    assert not is_target_safe(gv, near_wall_x, 1.0)


def test_clearance_is_the_distance_to_the_nearest_occupied_cell():
    rows = [[FREE] * 40 for _ in range(40)]
    rows[20][30] = WALL
    x, y = _grid(rows).cell_to_world(20, 20)
    assert clearance_at(_grid(rows), x, y, 2.0) == pytest.approx(0.50)
    assert clearance_at(_grid(rows), x, y, 0.3) == 0.3
    rows[20][30] = UNKNOWN
    assert clearance_at(_grid(rows), x, y, 2.0) == 2.0


def test_in_the_open_the_goal_stands_back_from_the_frontier():
    gv = _grid([[FREE] * 80 for _ in range(80)])
    found = approach_point(gv, (0.30, 1.50), (2.20, 1.50), standoff_m=0.45,
                           clearance_m=DEFAULT_CLEARANCE_M)
    assert found == pytest.approx((2.20 - 0.45, 1.50))


def test_a_blocked_standoff_moves_along_the_line_to_room():
    rows = [[FREE] * 60 for _ in range(60)]
    for y in range(60):
        rows[y][29] = WALL                     # where the 0.45 m standoff lands
    gv = _grid(rows)
    found = approach_point(gv, (0.30, 1.50), (2.20, 1.50), standoff_m=0.45,
                           clearance_m=DEFAULT_CLEARANCE_M)
    assert found is not None
    assert is_target_safe(gv, *found, safe_radius_m=DEFAULT_CLEARANCE_M)


def test_the_goal_moves_back_from_a_table_leg_beside_it():
    """The leg passes the chassis check, but the tabletop is over it."""
    rows = [[FREE] * 100 for _ in range(60)]
    robot, frontier = (0.30, 1.50), (4.00, 1.50)
    nominal = approach_point(_grid(rows), robot, frontier, standoff_m=0.45,
                             clearance_m=DEFAULT_CLEARANCE_M)
    lx, ly = _grid(rows).world_to_cell(nominal[0], nominal[1] + 0.30)
    rows[ly][lx] = WALL
    gv = _grid(rows)
    goal = approach_point(gv, robot, frontier, standoff_m=0.45,
                          clearance_m=DEFAULT_CLEARANCE_M)
    assert goal[0] < nominal[0]
    assert clearance_at(gv, *goal, 2.0) > 0.30


def test_a_frontier_with_no_room_anywhere_is_refused():
    rows = [[WALL] * 40 for _ in range(40)]
    rows[20][20] = FREE
    assert approach_point(_grid(rows), (0.50, 1.00), (1.05, 1.00),
                          standoff_m=0.45,
                          clearance_m=DEFAULT_CLEARANCE_M) is None


def test_keepout_applies_at_every_offset():
    gv = _grid([[FREE] * 80 for _ in range(80)])
    robot, frontier = (0.30, 1.50), (2.20, 1.50)
    blanket = [((robot[0] + frontier[0]) / 2, 1.50, 1.60)]
    assert approach_point(gv, robot, frontier, standoff_m=0.45,
                          clearance_m=DEFAULT_CLEARANCE_M,
                          keepout=blanket) is None


# Choosing a frontier

def test_the_goal_is_short_of_the_frontier_it_names():
    target = pick_target(_half_unknown(), robot_xy=(0.2, 1.0))
    assert target.drive_to[0] < target.centroid_xy[0]


def test_a_keepout_circle_moves_the_goal():
    clear = pick_target(_half_unknown(), robot_xy=(0.2, 1.0))
    gx, gy = clear.drive_to
    blocked = pick_target(_half_unknown(), robot_xy=(0.2, 1.0),
                          keepout=[(gx, gy, 0.5)])
    assert blocked is None or blocked.drive_to != clear.drive_to


def test_a_frontier_near_a_failed_goal_is_skipped():
    gv = _two_pockets()
    first = pick_target(gv, (5.0, 2.5), max_distance_m=8.0)
    assert first.centroid_xy[1] < 5.0          # pocket A
    fx, fy = first.centroid_xy
    # The cluster drifts a few cm between map updates.
    second = pick_target(gv, (5.0, 2.5), max_distance_m=8.0,
                         blocked_xy=[(fx + 0.05, fy - 0.05)])
    assert second.centroid_xy[1] > 5.0         # pocket B
    assert pick_target(gv, (5.0, 2.5), max_distance_m=8.0,
                       blocked_xy=[first.centroid_xy,
                                   second.centroid_xy]) is None


def test_between_equal_frontiers_the_roomier_one_wins():
    rows = [[FREE] * 120 for _ in range(60)]
    rows[0] = [UNKNOWN] * 120
    rows[59] = [UNKNOWN] * 120
    for y in range(36, 56, 4):                 # clutter before the upper one
        for x in range(50, 71, 4):
            rows[y][x] = WALL
    gv = _grid(rows)
    robot = gv.cell_to_world(60, 30)
    target = pick_target(gv, robot, min_size=3, max_distance_m=8.0)
    assert target.centroid_xy[1] < robot[1]


def test_unknown_under_the_robot_is_not_a_frontier():
    rows = [[FREE] * 40 for _ in range(40)]
    for y in range(17, 24):
        for x in range(17, 24):
            rows[y][x] = UNKNOWN
    rows[5][5] = WALL
    gv = _grid(rows)
    under = {(x, y) for y in range(17, 24) for x in range(17, 24)}
    filled = fill_traversed(gv, under | {(5, 5)})
    assert len(find_frontier_cells(gv)) > 0
    assert len(find_frontier_cells(filled)) == 0
    assert filled.data[5, 5] == WALL           # occupied stays occupied
    assert gv.data[20, 20] == UNKNOWN          # the input is not modified


# The costmap's lethal cells

def _costmap(cost, res, ox=0.0, oy=0.0, oyaw=0.0):
    h, w = cost.shape
    q = NS(x=0.0, y=0.0, z=math.sin(oyaw / 2), w=math.cos(oyaw / 2))
    return NS(data=cost.tobytes(), info=NS(
        width=w, height=h, resolution=res,
        origin=NS(position=NS(x=ox, y=oy), orientation=q)))


def test_lethal_cells_become_obstacles_on_free_cells_only():
    data = np.zeros((40, 40), dtype=np.int8)
    data[:, 30:] = UNKNOWN
    gv = GridView(data=data, resolution=0.05, origin_x=0.0, origin_y=0.0,
                  width=40, height=40)
    cost = np.zeros((20, 20), dtype=np.int8)   # 0.1 m cells
    cost[5:8, 5:8] = 100                       # a table top at 0.5-0.8 m
    cost[5:8, 15:18] = 100                     # over unknown
    cost[10:12, 10:12] = 99                    # inscribed, not lethal
    out = overlay_lethal(gv, _costmap(cost, 0.1)).data
    assert (out[10:16, 10:16] == WALL).all()
    assert (out[10:16, 30:36] == UNKNOWN).all()
    assert (out[20:24, 20:24] == FREE).all()


def test_a_costmap_in_another_frame_is_transformed():
    gv = GridView(data=np.zeros((60, 60), dtype=np.int8), resolution=0.05,
                  origin_x=0.0, origin_y=0.0, width=60, height=60)
    # A table top at map (2.0..2.2, 0.5..0.7), seen from a costmap frame at
    # map (1, 0) turned +90 degrees: map (x, y) -> frame (y, -(x - 1)).
    cost = np.zeros((40, 40), dtype=np.int8)
    for fx in np.arange(0.5, 0.7, 0.05):
        for fy in np.arange(-1.2, -1.0, 0.05):
            cost[int((fy + 2.0) / 0.05), int(fx / 0.05)] = 100
    out = overlay_lethal(gv, _costmap(cost, 0.05, oy=-2.0),
                         (1.0, 0.0, math.pi / 2)).data
    assert (out[11:13, 41:43] == WALL).all()  # around map (2.1, 0.6)
    assert (out == WALL).sum() <= 20
    # A grid origin turned 90 degrees: grid column i runs along frame +y.
    cost = np.zeros((40, 40), dtype=np.int8)
    cost[2:4, 12:14] = 100
    out = overlay_lethal(gv, _costmap(cost, 0.05, ox=1.0,
                                      oyaw=math.pi / 2)).data
    assert (out[12:14, 16:18] == WALL).any()  # map x 0.8..0.9, y 0.6..0.7


def test_unknown_standoff_searches_back_to_known_ground():
    # A curved frontier's representative point can be on its far side.
    # The nominal approach crosses unknown, but a farther standoff is free.
    rows = [[FREE] * 60 for _ in range(40)]
    for row in rows:
        row[30:40] = [UNKNOWN] * 10
    gv = _grid(rows)
    goal = approach_point(gv, (.5, 1.), (2.05, 1.), standoff_m=.45,
                          clearance_m=DEFAULT_CLEARANCE_M)
    assert goal is not None
    cx, cy = gv.world_to_cell(*goal)
    assert gv.data[cy, cx] == FREE


def test_unknown_only_approaches_do_not_become_navigation_goals():
    gv = _grid([[UNKNOWN] * 60 for _ in range(40)])
    assert approach_point(gv, (.5, 1.), (2.05, 1.), standoff_m=.45,
                          clearance_m=DEFAULT_CLEARANCE_M) is None


def test_known_goal_can_still_border_unknown_space():
    gv = _half_unknown()
    goal = approach_point(gv, (.5, 1.), (1.5, 1.), standoff_m=.1,
                          clearance_m=DEFAULT_CLEARANCE_M)
    assert goal == pytest.approx((1.4, 1.))


def test_approach_just_below_map_origin_is_outside():
    gv = _grid([[FREE] * 20 for _ in range(20)])
    assert approach_point(gv, (-.02, .5), (-.01, .5), standoff_m=.45,
                          clearance_m=DEFAULT_CLEARANCE_M) is None
