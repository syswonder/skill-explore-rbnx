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

from explore_skill.frontier import (GridView, cluster_frontiers,
                                    find_frontier_cells, overlay_lethal)
from explore_skill.map_opt import optimize
from explore_skill.strategies import (_Reach, config_problem, pick,
                                      passage_frontier_cells,
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
    return _world(passage_frontier_cells(gv, reach, passage))


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
