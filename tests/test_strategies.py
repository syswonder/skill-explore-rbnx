# SPDX-License-Identifier: MulanPSL-2.0
"""The `strategy` config key and the path-aware strategies behind it."""
import math

import numpy as np
import pytest

from explore_skill.frontier import GridView
from explore_skill.strategies import STRATEGIES, pick, validate

FREE, UNKNOWN, WALL = 0, -1, 100
RADIUS = 0.235


def _grid(data, resolution=0.05):
    h, w = data.shape
    return GridView(data=data.astype(np.int8), resolution=resolution,
                    origin_x=0.0, origin_y=0.0, width=w, height=h)


def _room(w, h):
    data = np.full((h, w), FREE, dtype=np.int8)
    data[0, :] = data[-1, :] = data[:, 0] = data[:, -1] = WALL
    return data


def test_a_frontier_behind_a_wall_is_as_far_as_the_way_round():
    """Straight-line distance puts the frontier just behind the wall first;
    the way round is longer than the walk to the one in the open."""
    data = _room(120, 60)
    data[0:46, 60] = WALL                      # gap at the top only
    data[5:16, 64:71] = UNKNOWN                # 0.85 m away through the wall
    data[40:56, 5:9] = UNKNOWN                 # 2.9 m away in the open
    robot = (2.5, 0.5)
    target = pick("mrtsp", _grid(data), robot, min_size=8,
                  robot_radius_m=RADIUS)
    assert target is not None
    assert target.centroid_xy[0] < 3.0
    assert target.goal_xy[0] < 3.0             # goal on the robot's side


def test_a_frontier_with_no_way_to_it_is_dropped():
    data = _room(100, 60)
    data[20:41, 60:81] = WALL
    data[21:40, 61:80] = FREE
    data[25:36, 65:76] = UNKNOWN               # walled in, robot outside
    assert pick("mrtsp", _grid(data), (1.0, 1.0), min_size=8,
                robot_radius_m=RADIUS) is None


def test_the_goal_is_short_of_the_frontier_and_faces_it():
    """Down a corridor the goal sits between the robot and the frontier,
    so arriving facing the frontier needs no turn."""
    data = _room(120, 60)
    data[1:-1, 100:-1] = UNKNOWN
    gv = _grid(data)
    robot = (0.5, 1.5)
    target = pick("mrtsp", gv, robot, min_size=8, robot_radius_m=RADIUS)
    gx, gy = target.goal_xy
    fx, fy = target.centroid_xy
    assert gx < fx - 0.4
    heading = math.atan2(fy - gy, fx - gx)
    assert abs(heading) < math.radians(30)


def test_goals_keep_the_robot_radius_from_walls():
    data = _room(120, 60)
    data[1:-1, 100:-1] = UNKNOWN
    data[30, 20:90] = WALL                     # a wall down the middle
    gv = _grid(data)
    target = pick("mrtsp", gv, (0.5, 0.5), min_size=8,
                  robot_radius_m=RADIUS)
    gy = target.goal_xy[1]
    assert abs(gy - 1.5) > RADIUS and gy > 0.05 + RADIUS


def test_unknown_strategies_are_rejected():
    for s in STRATEGIES:
        assert validate(s) == s
    with pytest.raises(ValueError, match="unknown explore strategy"):
        validate("tare")

