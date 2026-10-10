# SPDX-License-Identifier: MulanPSL-2.0
"""Goals keep their distance, and the ground under the robot is not a frontier.

On the Lite3 the explorer parked beside table legs, the only part of a table
a chassis-height lidar sees, and kept picking frontiers in the unknown disc
the lidar's blind zone leaves around the robot.
"""
import numpy as np

from explore_skill.frontier import (DEFAULT_CLEARANCE_M, GridView,
                                    approach_point, clearance_at,
                                    fill_traversed, find_frontier_cells,
                                    pick_target)

FREE, UNKNOWN, WALL = 0, -1, 100


def _grid(rows, resolution=0.05):
    data = np.array(rows, dtype=np.int8)
    h, w = data.shape
    return GridView(data=data, resolution=resolution, origin_x=0.0,
                    origin_y=0.0, width=w, height=h)


def test_clearance_is_the_distance_to_the_nearest_occupied_cell():
    rows = [[FREE] * 40 for _ in range(40)]
    rows[20][30] = WALL
    gv = _grid(rows)
    x, y = gv.cell_to_world(20, 20)
    assert abs(clearance_at(gv, x, y, 2.0) - 0.50) < 1e-6
    assert clearance_at(gv, x, y, 0.3) == 0.3
    rows[20][30] = UNKNOWN
    assert clearance_at(_grid(rows), x, y, 2.0) == 2.0


def test_the_goal_moves_back_from_a_table_leg_beside_the_nominal_standoff():
    """A leg 0.3 m to the side of the nominal standoff passes the chassis
    check, but the tabletop is above it. Further back along the line there
    is room, and that is where the goal goes."""
    rows = [[FREE] * 100 for _ in range(60)]
    robot, frontier = (0.30, 1.50), (4.00, 1.50)
    gv = _grid(rows)
    nominal = approach_point(gv, robot, frontier, standoff_m=0.45,
                             clearance_m=DEFAULT_CLEARANCE_M)
    lx, ly = gv.world_to_cell(nominal[0], nominal[1] + 0.30)
    rows[ly][lx] = WALL
    gv = _grid(rows)
    goal = approach_point(gv, robot, frontier, standoff_m=0.45,
                          clearance_m=DEFAULT_CLEARANCE_M)
    assert goal[0] < nominal[0]
    assert clearance_at(gv, goal[0], goal[1], 2.0) > 0.30


def test_between_equal_frontiers_the_roomier_one_wins():
    rows = [[FREE] * 120 for _ in range(60)]
    for x in range(120):
        rows[0][x] = UNKNOWN
        rows[59][x] = UNKNOWN
    # Clutter along the approach to the upper frontier only.
    for y in range(36, 56, 4):
        for x in range(50, 71, 4):
            rows[y][x] = WALL
    gv = _grid(rows)
    robot = gv.cell_to_world(60, 30)
    target = pick_target(gv, robot, min_size=3, max_distance_m=8.0)
    assert target is not None
    assert target.centroid_xy[1] < robot[1]


def test_unknown_under_the_robot_is_not_a_frontier():
    rows = [[FREE] * 40 for _ in range(40)]
    for y in range(17, 24):
        for x in range(17, 24):
            rows[y][x] = UNKNOWN
    rows[5][5] = WALL
    gv = _grid(rows)
    assert len(find_frontier_cells(gv)) > 0
    under = {(x, y) for y in range(17, 24) for x in range(17, 24)}
    filled = fill_traversed(gv, under | {(5, 5)})
    assert len(find_frontier_cells(filled)) == 0
    assert filled.data[5, 5] == WALL          # occupied stays occupied
    assert gv.data[20, 20] == UNKNOWN         # the input is not modified
