# SPDX-License-Identifier: MulanPSL-2.0
"""Scene objects as keep-out circles (`ExploreController._keepout_circles`).

The lidar map sees a table as free space; Scene has it from the camera."""
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


def _obj(label, x, y, sx, sy, z=0.4, sz=0.8, frame="map"):
    return {"label": label, "x": x, "y": y, "z": z, "frame_id": frame,
            "size_x": sx, "size_y": sy, "size_z": sz}


def test_a_table_becomes_a_circle_the_chassis_fits_outside_of():
    """Half the footprint diagonal covers the box at any yaw; the chassis
    radius is added because a goal is where the robot's centre goes."""
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
    """list_objects also returns rooms and areas, with no footprint."""
    ctrl = _controller()
    ctrl._scene_mcp_call = lambda tool, args: _objects(
        _obj("living_room", 0.0, 0.0, 0.0, 0.0),
        _obj("chair", 1.0, 1.0, 0.5, 0.5))
    circles = ctrl._keepout_circles()
    assert len(circles) == 1
    assert circles[0][:2] == (1.0, 1.0)


def test_an_entry_with_no_footprint_is_skipped():
    ctrl = _controller()
    ctrl._scene_mcp_call = lambda tool, args: _objects(
        _obj("note", 1.0, 1.0, 0.0, 0.0))
    assert ctrl._keepout_circles() == []


def test_no_scene_means_no_circles_and_no_call():
    ctrl = _controller(scene_endpoint=None)
    called = []
    ctrl._scene_mcp_call = lambda tool, args: called.append(tool) or {}
    assert ctrl._keepout_circles() == []
    assert called == []


def test_a_failed_lookup_yields_no_circles_rather_than_an_error():
    ctrl = _controller()
    ctrl._scene_mcp_call = lambda tool, args: {}
    assert ctrl._keepout_circles() == []


def test_the_snapshot_is_reused_within_its_ttl():
    ctrl = _controller()
    calls = []

    def once(tool, args):
        calls.append(tool)
        return _objects(_obj("chair", 1.0, 1.0, 0.5, 0.5))

    ctrl._scene_mcp_call = once
    ctrl._keepout_circles()
    ctrl._keepout_circles()
    assert len(calls) == 1


def test_large_furniture_is_kept():
    """A dining table or a bed is exactly what the lidar passes under."""
    ctrl = _controller()
    ctrl._scene_mcp_call = lambda tool, args: _objects(
        _obj("dining_table", 1.0, 1.0, 2.4, 1.0), _obj("bed", 4.0, 1.0, 2.0, 1.6, z=0.3, sz=0.6))
    assert len(ctrl._keepout_circles()) == 2


def test_only_what_the_robot_could_hit_is_kept():
    ctrl = _controller()
    ctrl._scene_mcp_call = lambda tool, args: _objects(
        _obj("ceiling_lamp", 1.0, 1.0, 0.5, 0.5, z=2.6, sz=0.3),
        _obj("picture", 2.0, 1.0, 0.8, 0.1, z=1.8, sz=0.5),
        _obj("rug", 3.0, 1.0, 2.0, 1.5, z=0.01, sz=0.02),
        _obj("table", 4.0, 1.0, 1.2, 0.8, z=0.38, sz=0.76))
    assert [c[:2] for c in ctrl._keepout_circles()] == [(4.0, 1.0)]


def test_an_object_in_another_frame_is_skipped():
    ctrl = _controller()
    ctrl._scene_mcp_call = lambda tool, args: _objects(
        _obj("chair", 1.0, 1.0, 0.5, 0.5, frame="odom"),
        _obj("table", 2.0, 1.0, 1.2, 0.8))
    assert [c[:2] for c in ctrl._keepout_circles()] == [(2.0, 1.0)]


def test_paused_zero_reference_keeps_snapshot_but_new_generation_rejects_pending_result():
    from unittest.mock import Mock
    ctrl = _controller()
    reading = [(0, 1, 1)]
    ctrl._reference = Mock(read=lambda: reading[0])
    calls = []
    ctrl._scene_mcp_call = lambda *args: calls.append(args) or _objects(_obj('chair', 1, 1, .5, .5))
    assert ctrl._keepout_circles()
    assert ctrl._keepout_circles()
    assert len(calls) == 1
    reading[0] = (11_000_000_000, 1, 1)
    def interrupted(*args):
        reading[0] = (0, 2, 1)
        return _objects(_obj('table', 2, 2, 1, 1))
    ctrl._scene_mcp_call = interrupted
    with pytest.raises(RuntimeError, match='reference changed'):
        ctrl._keepout_circles()
    assert ctrl._keepout_cache[0][:2] == (1, 1)
    ctrl._scene_mcp_call = lambda *args: {}
    assert ctrl._keepout_circles() == []
