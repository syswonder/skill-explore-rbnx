# SPDX-License-Identifier: MulanPSL-2.0
"""`min_passage_m`: explore a corridor and not the rooms off it.

The floor below is a 2.0 m corridor along y 5.0-7.0 m, its far end still
unknown, with:
  room A  below, a 0.9 m door at x 4.0-4.9; the lidar saw a wide fan of
          the room through it, so the room has the longest frontier;
  room B  above, a 0.9 m door at x 11.0-11.9, the room unseen: a frontier
          right in the doorway;
  lounge  below, through a 1.6 m archway at x 8.0-9.6, partly seen.
"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from explore_skill.frontier import (GridView, cluster_frontiers, pad_unknown,
                                    find_frontier_cells, overlay_lethal)
from explore_skill.map_opt import optimize
from explore_skill.strategies import (_Reach, config_problem, pick,
                                      passage_frontiers,
                                      passage_frontier_count)

FREE, UNKNOWN, WALL = 0, -1, 100
RES = 0.05
RADIUS = 0.32
PASSAGE = 1.2
START = (2.0, 6.0)
STRATEGIES = ("mrtsp", "frontier_greedy")
DOORS = ((4.0, 4.9, 4.95), (11.0, 11.9, 7.05))   # x from, x to, wall y


def _c(m):
    return int(round(m / RES))


def _grid(data):
    h, w = data.shape
    return GridView(data=data.astype(np.int8), resolution=RES,
                    origin_x=0.0, origin_y=0.0, width=w, height=h)


def _box(data, x0, x1, y0, y1, value):
    data[_c(y0):_c(y1), _c(x0):_c(x1)] = value


def _floor(room_b_door=True):
    data = np.full((_c(12.0), _c(20.0)), UNKNOWN, dtype=np.int8)
    _box(data, 0.5, 17.0, 4.9, 7.1, WALL)        # corridor walls ...
    _box(data, 0.6, 17.0, 5.0, 7.0, FREE)        # ... and the corridor
    # Room A, a fan seen through its door.
    _box(data, 4.0, 4.9, 4.9, 5.0, FREE)
    for y in np.arange(1.5, 4.9, RES):
        half = 0.45 + (4.9 - y) * 0.9
        _box(data, max(2.0, 4.45 - half), min(7.0, 4.45 + half),
             y, y + RES, FREE)
    # Room B, unseen behind its door.
    if room_b_door:
        _box(data, 11.0, 11.9, 7.0, 7.1, FREE)
    # The lounge through the archway.
    _box(data, 8.0, 9.6, 4.9, 5.0, FREE)
    _box(data, 7.6, 10.0, 2.5, 4.9, FREE)
    _box(data, 7.5, 7.6, 2.5, 4.9, WALL)
    _box(data, 10.0, 10.1, 2.5, 4.9, WALL)
    return data


def _world(cells):
    return [((x + 0.5) * RES, (y + 0.5) * RES) for x, y in cells]


def _in_room(x, y):
    """Behind a door: room A below x 2-7, room B above."""
    return (y < 5.0 and 2.0 <= x <= 7.0) or y > 7.0


def _pick(strategy, gv, robot=START, passage=PASSAGE, **kwargs):
    return pick(strategy, gv, robot, min_size=8, robot_radius_m=RADIUS,
                min_passage_m=passage, **kwargs)


def _all_picks(strategy, gv, robot=START, passage=PASSAGE):
    """Every target the strategy offers, blocking each one in turn."""
    out, blocked = [], []
    for _ in range(20):
        t = _pick(strategy, gv, robot, passage, blocked_xy=blocked)
        if t is None:
            break
        out.append(t)
        blocked.append(t.centroid_xy)
    return out


def _cells(gv, passage=PASSAGE, robot=START):
    reach = _Reach(gv, robot, RADIUS, min_passage_m=passage)
    return _world(np.concatenate([c.cell_indices for c in
                                  passage_frontiers(gv, reach, passage, 1)]
                                 or [np.zeros((0, 2), int)]))


def test_min_passage_m_is_checked_at_init():
    assert config_problem({"min_passage_m": 1.2}) is None
    assert config_problem({"min_passage_m": 0}) is None
    for bad in (-0.5, "wide", True):
        assert "min_passage_m" in config_problem({"min_passage_m": bad})


def test_the_room_seen_through_a_door_has_the_largest_frontier():
    clusters = cluster_frontiers(find_frontier_cells(_grid(_floor())), 8)
    largest = max(clusters, key=lambda c: c.size)
    assert all(_in_room(*p) for p in _world(largest.cell_indices))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_nothing_behind_or_in_a_door_is_chosen(strategy):
    gv = _grid(_floor())
    assert not any(_in_room(*p) for p in _cells(gv))
    picks = _all_picks(strategy, gv)
    assert picks
    for t in picks:
        assert not _in_room(*t.centroid_xy), t.centroid_xy
        assert not _in_room(*t.drive_to), t.drive_to
        assert all(not _in_room(*p) for p in _world(t.cell_indices))
        # Not in a doorway either.
        for x0, x1, wall_y in DOORS:
            x, y = t.drive_to
            assert not (x0 - 0.3 <= x <= x1 + 0.3 and abs(y - wall_y) < 0.5)


def test_the_optimized_map_mrtsp_searches_is_gated_too():
    gv = optimize(_grid(_floor()))
    for t in _all_picks("mrtsp", gv):
        assert not _in_room(*t.centroid_xy)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_off_the_rooms_are_reachable_as_before(strategy):
    gv = _grid(_floor())
    with_zero = _all_picks(strategy, gv, passage=0.0)
    assert any(_in_room(*t.centroid_xy) for t in with_zero)
    plain = pick(strategy, gv, START, min_size=8, robot_radius_m=RADIUS)
    assert plain.centroid_xy == with_zero[0].centroid_xy


def test_the_corridor_end_and_the_archway_still_count():
    cells = _cells(_grid(_floor()))
    assert any(x > 16.5 and 5.0 <= y <= 7.0 for x, y in cells)  # corridor end
    assert any(7.6 <= x <= 10.0 and y < 3.0 for x, y in cells)  # lounge


def test_a_frontier_running_through_a_doorway_keeps_its_corridor_part():
    """Room C at the corridor's end: its door at x 16.1-17.0 lines up with
    the corridor's unknown end and its far jamb is not mapped yet, so one
    frontier runs from the corridor through the doorway into the room."""
    data = _floor()
    _box(data, 16.1, 17.0, 4.9, 5.0, FREE)
    _box(data, 15.0, 17.0, 2.5, 4.9, FREE)
    gv = _grid(data)
    raw = cluster_frontiers(find_frontier_cells(gv), 8)
    line = [c for c in raw
            if any(p[0] > 16.8 and p[1] > 5.0 for p in _world(c.cell_indices))]
    assert len(line) == 1
    assert any(p[1] < 4.5 for p in _world(line[0].cell_indices))
    kept = [p for p in _cells(gv) if p[0] > 16.8]
    assert kept and all(5.0 <= y <= 7.0 for _, y in kept)


def test_a_door_only_the_costmap_shows():
    """The map has a 1.6 m opening; the camera sees a half-open door leaf
    in it, which only the costmap holds, leaving 0.7 m."""
    gv = _grid(_floor())
    cost = np.zeros(gv.data.shape, dtype=np.int8)
    _box(cost, 8.0, 8.9, 4.9, 5.0, 100)
    costmap = NS(data=cost.tobytes(), info=NS(
        width=gv.width, height=gv.height, resolution=RES,
        origin=NS(position=NS(x=0.0, y=0.0),
                  orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))))
    lounge = lambda cells: [p for p in cells if 7.6 <= p[0] <= 10.0 and p[1] < 4.9]
    assert lounge(_cells(gv))
    assert not lounge(_cells(overlay_lethal(gv, costmap)))


def test_a_corridor_narrower_than_the_setting_ends_the_run():
    data = np.full((_c(4.0), _c(12.0)), UNKNOWN, dtype=np.int8)
    _box(data, 0.5, 10.0, 1.4, 2.7, WALL)
    _box(data, 0.6, 10.0, 1.5, 2.6, FREE)        # 1.1 m wide
    gv = _grid(data)
    for strategy in STRATEGIES:
        assert _pick(strategy, gv, (1.5, 2.05)) is None
        assert _pick(strategy, gv, (1.5, 2.05), passage=0.0) is not None
    assert passage_frontier_count(gv, (1.5, 2.05), robot_radius_m=RADIUS,
                                  min_passage_m=PASSAGE, min_size=8) == 0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_robot_beside_the_wall_still_finds_the_corridor(strategy):
    gv = _grid(_floor(room_b_door=False))
    robot = (14.0, 5.0 + 0.3)
    t = _pick(strategy, gv, robot)
    assert t is not None and not _in_room(*t.centroid_xy)


def _start_between_doors(door_south, door_north):
    """The sim run's start: a 2.2 m corridor along y 0-2.2, the robot just
    east of a door on each side, everything west of it unseen and the map
    cropped there, as SLAM crops it. Rooms behind the doors are unseen."""
    data = np.full((_c(2.4), _c(12.0)), UNKNOWN, dtype=np.int8)
    _box(data, 0.0, 12.0, 0.0, 0.1, WALL)
    _box(data, 0.0, 12.0, 2.3, 2.4, WALL)
    _box(data, 0.0, 12.0, 0.1, 2.3, FREE)
    for (x0, x1), (y0, y1) in ((door_south, (0.0, 0.1)), (door_north, (2.3, 2.4))):
        if x0 is not None:
            _box(data, x0, x1, y0, y1, FREE)
    gv = _grid(data)
    # Shift so the grid starts at x = 3.65, the robot at 4.0.
    return GridView(data=gv.data, resolution=RES, origin_x=3.65,
                    origin_y=-0.1, width=gv.width, height=gv.height)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("doors", [((3.65, 3.9), (3.65, 3.9)),    # both sides
                                   ((3.65, 3.9), (None, None))])  # one side
def test_the_corridor_behind_the_start_is_explored(strategy, doors):
    """Regression: the map ended at the robot's back, free to its edge, so
    the corridor behind the start had no frontier and was never seen."""
    gv = pad_unknown(_start_between_doors(*doors))
    robot = (4.0, 1.1)
    picks = _all_picks(strategy, gv, robot)
    behind = [t for t in picks if t.centroid_xy[0] < 3.8]
    assert behind, [t.centroid_xy for t in picks]
    for t in behind:
        assert 0.0 <= t.centroid_xy[1] <= 2.2


def test_a_corridor_end_keeps_a_frontier_long_enough_for_mrtsp():
    """The gate trims the frontier off the walls; what is left of a 1.9 m
    corridor's end must still pass mrtsp's 1 m minimum, judged before."""
    data = np.full((_c(4.0), _c(12.0)), UNKNOWN, dtype=np.int8)
    _box(data, 0.5, 10.0, 0.9, 3.0, WALL)
    _box(data, 0.6, 10.0, 1.0, 2.9, FREE)                # 1.9 m
    t = pick("mrtsp", _grid(data), (2.0, 2.0), min_size=20,
             robot_radius_m=RADIUS, min_passage_m=PASSAGE)
    assert t is not None and t.centroid_xy[0] > 9.5


def _wedge_corridor():
    """Regression (sim run tiago-on-east2): a 2.6 m corridor x 0-20, the
    robot at x 2 looking along it. Beams near the axis found nothing in
    range, so a wedge of unknown opens ahead of the robot; the strips of
    free space beside it are known up to x 12. A 0.9 m door at x 6
    opens into a room the lidar saw a little of."""
    data = np.full((_c(8.0), _c(20.0)), UNKNOWN, dtype=np.int8)
    _box(data, 0.0, 20.0, 2.0, 2.1, WALL)
    _box(data, 0.0, 20.0, 4.7, 4.8, WALL)
    _box(data, 0.0, 12.0, 2.1, 4.7, FREE)
    for x in np.arange(2.6, 12.0, RES):          # the wedge, 10 degrees each side
        half = (x - 2.4) * math.tan(math.radians(10))
        _box(data, x, x + RES, 3.4 - half, 3.4 + half, UNKNOWN)
    _box(data, 6.0, 6.9, 2.0, 2.1, FREE)          # the door, both jambs mapped
    _box(data, 5.5, 7.4, 0.5, 2.0, FREE)
    return _grid(data)


def test_a_wedge_ahead_does_not_end_the_corridor():
    gv, robot = _wedge_corridor(), (2.0, 3.4)
    apex = (2.8, 3.4)
    strict = _pick("mrtsp", gv, robot, blocked_xy=[apex])
    assert strict is None or strict.centroid_xy[0] < 4.0
    t = pick("mrtsp", gv, robot, min_size=8, robot_radius_m=RADIUS,
             min_passage_m=PASSAGE, blocked_xy=[apex], keep_off_unknown=False)
    assert t is not None and t.centroid_xy[0] > 4.0
    assert 2.1 < t.centroid_xy[1] < 4.7 and 2.1 < t.drive_to[1] < 4.7


def test_without_keeping_off_unknown_the_door_still_shuts():
    gv, robot = _wedge_corridor(), (2.0, 3.4)
    blocked = []
    for _ in range(10):
        t = pick("mrtsp", gv, robot, min_size=8, robot_radius_m=RADIUS,
                 min_passage_m=PASSAGE, blocked_xy=blocked, keep_off_unknown=False)
        if t is None:
            break
        assert t.centroid_xy[1] > 2.1 and t.drive_to[1] > 2.1, t.centroid_xy
        blocked.append(t.centroid_xy)


@pytest.mark.parametrize("passage", (0.0, PASSAGE))
def test_a_frontier_still_there_after_a_visit_is_tried_at_its_far_part(passage):
    """Only the cells near a tried frontier are given up: a 6 m frontier
    along the corridor's edge, tried at its east end, is tried again west."""
    data = np.full((_c(5.0), _c(14.0)), UNKNOWN, dtype=np.int8)
    _box(data, 0.0, 14.0, 0.9, 1.0, WALL)
    _box(data, 0.0, 14.0, 1.0, 3.6, FREE)          # corridor 2.6 m, north side unseen
    _box(data, 0.0, 0.1, 0.9, 3.6, WALL)
    _box(data, 13.9, 14.0, 0.9, 3.6, WALL)
    _box(data, 0.1, 13.9, 3.6, 3.7, UNKNOWN)
    gv = _grid(data)
    first = pick("mrtsp", gv, (12.0, 2.3), min_size=20, robot_radius_m=RADIUS,
                 min_passage_m=passage)
    assert first is not None
    again = pick("mrtsp", gv, (12.0, 2.3), min_size=20, robot_radius_m=RADIUS,
                 min_passage_m=passage, blocked_xy=[first.centroid_xy])
    assert again is not None
    assert math.hypot(again.centroid_xy[0] - first.centroid_xy[0],
                      again.centroid_xy[1] - first.centroid_xy[1]) > 1.0
