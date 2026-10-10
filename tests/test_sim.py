# SPDX-License-Identifier: MulanPSL-2.0
"""Whole runs of the real controller in the 2D simulator (sim/).

Each world is a situation the Lite3 got stuck in. Before these fixes the
robot never left the unknown disc its lidar leaves under it: hundreds of
goals, no distance travelled, and a full turn every few seconds.
"""
import math

import pytest

from sim.run import run
from sim.world import WORLDS


@pytest.fixture(scope="module")
def corridor():
    return run(WORLDS["corridor"](), timeout_s=900)


def test_a_long_corridor_is_explored_to_the_far_end(corridor):
    assert corridor.state == "done"
    assert corridor.coverage > 0.95
    assert corridor.distance_m > 15.0          # the far end is 24 m away


def test_the_robot_does_not_turn_in_place_much(corridor):
    """About one turn per 10 m of straight corridor is plenty."""
    assert math.degrees(corridor.rotation_rad) < 36 * corridor.distance_m
    assert corridor.recoveries == 0


def test_a_room_with_tables_is_explored_without_touching_them():
    result = run(WORLDS["tables"](), timeout_s=900)
    assert result.state == "done"
    assert result.collisions == 0
    assert result.coverage > 0.95


def test_a_glass_wall_does_not_trap_the_run():
    """The lidar sees through glass, so the robot will bump it; what
    matters is that the run still ends and maps the room."""
    result = run(WORLDS["glass"](), timeout_s=900)
    assert result.state in ("done", "error")
    assert result.coverage > 0.9
    assert result.sim_time_s < 900
