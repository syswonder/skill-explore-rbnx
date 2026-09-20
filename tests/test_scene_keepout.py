# SPDX-License-Identifier: MulanPSL-2.0
"""Turning what Scene sees into places not to drive.

A lidar at chassis height passes under a tabletop, so the occupancy grid
calls that square metre free and always will. Scene has the table from the
camera. These pin the translation between the two, and the two ways it can
go wrong: swallowing the map because list_objects still returns room
entries for v1 compatibility, and treating the robot as an obstacle.
"""
import math

import pytest

from explore_skill.controller import ExploreController
from explore_skill.frontier import ROBOT_RADIUS_M


def _controller(scene_endpoint="http://127.0.0.1:1/mcp/"):
    """Constructed, not started: __init__ opens nothing."""
    return ExploreController(
        map_topic="/map",
        nav_navigate_endpoint="http://127.0.0.1:1/mcp/",
        nav_status_endpoint="http://127.0.0.1:1/mcp/",
        nav_cancel_endpoint="http://127.0.0.1:1/mcp/",
        scene_objects_endpoint=scene_endpoint,
    )


def _objects(*entries):
    return {"objects": list(entries)}


def _obj(label, x, y, sx, sy):
    return {"label": label, "x": x, "y": y,
            "size_x": sx, "size_y": sy}


def test_a_table_becomes_a_circle_the_chassis_fits_outside_of():
    """Half the footprint diagonal covers the box at any yaw, and the
    chassis radius is added because the goal is where the robot centre
    goes, not where its edge stops."""
    ctrl = _controller()
    ctrl._scene_mcp_call = lambda tool, args: _objects(
        _obj("table", 2.0, 3.0, 1.2, 0.8))
    circles = ctrl._keepout_circles()
    assert len(circles) == 1
    x, y, r = circles[0]
    assert (x, y) == (2.0, 3.0)
    assert r == pytest.approx(0.5 * math.hypot(1.2, 0.8) + ROBOT_RADIUS_M)


def test_the_robot_is_not_an_obstacle_to_itself():
    ctrl = _controller()
    ctrl._scene_mcp_call = lambda tool, args: _objects(
        _obj("robot", 0.0, 0.0, 0.54, 0.54),
        _obj("chair", 1.0, 1.0, 0.5, 0.5))
    assert len(ctrl._keepout_circles()) == 1


def test_a_room_entry_does_not_blanket_the_map():
    """list_objects still returns room entries for v1 compatibility. One
    of those as a keep-out circle would cover everything and end the run
    with \x27no safe frontier\x27."""
    ctrl = _controller()
    ctrl._scene_mcp_call = lambda tool, args: _objects(
        _obj("living_room", 0.0, 0.0, 6.0, 5.0),
        _obj("chair", 1.0, 1.0, 0.5, 0.5))
    circles = ctrl._keepout_circles()
    assert len(circles) == 1
    assert circles[0][:2] == (1.0, 1.0)


def test_an_entry_with_no_footprint_is_skipped():
    """Annotations carry a pose but no box; a zero-size circle is either
    meaningless or, with the chassis added, a phantom obstacle."""
    ctrl = _controller()
    ctrl._scene_mcp_call = lambda tool, args: _objects(
        _obj("note", 1.0, 1.0, 0.0, 0.0))
    assert ctrl._keepout_circles() == []


def test_no_scene_means_no_circles_and_no_call():
    """A deployment without Scene explores exactly as it did before."""
    ctrl = _controller(scene_endpoint=None)
    called = []
    ctrl._scene_mcp_call = lambda tool, args: called.append(tool) or {}
    assert ctrl._keepout_circles() == []
    assert called == []


def test_a_failed_lookup_yields_no_circles_rather_than_an_error():
    """Refusing to explore because an optional input is unavailable trades
    a working explorer for a stricter one."""
    ctrl = _controller()
    ctrl._scene_mcp_call = lambda tool, args: {}
    assert ctrl._keepout_circles() == []


def test_the_snapshot_is_reused_within_its_ttl():
    """Furniture does not move on the timescale of an exploration leg."""
    ctrl = _controller()
    calls = []

    def once(tool, args):
        calls.append(tool)
        return _objects(_obj("chair", 1.0, 1.0, 0.5, 0.5))

    ctrl._scene_mcp_call = once
    ctrl._keepout_circles()
    ctrl._keepout_circles()
    assert len(calls) == 1
