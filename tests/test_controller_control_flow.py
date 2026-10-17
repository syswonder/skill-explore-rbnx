import asyncio
import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from explore_skill.controller import ExploreController, TaskHandle


def _map_msg(pockets=((20, 60), (70, 60))):
    """OccupancyGrid-like message: a walled 10 m room at 0.1 m/cell with a
    1 m unknown pocket, one frontier cluster, at each (row, column)."""
    data = np.zeros((100, 100), dtype=np.int8)
    data[0, :] = data[-1, :] = data[:, 0] = data[:, -1] = 100
    for y, x in pockets:
        data[y:y + 10, x:x + 10] = -1
    origin = SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))
    info = SimpleNamespace(width=100, height=100, resolution=0.1,
                           origin=origin)
    return SimpleNamespace(info=info, data=data.tobytes())


class ExploreControlFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = ExploreController(
            map_topic="/map",
            nav_navigate_endpoint="http://127.0.0.1:1/mcp/",
            nav_status_endpoint="http://127.0.0.1:1/mcp/",
            nav_cancel_endpoint="http://127.0.0.1:1/mcp/",
        )
        self.controller._latest_pose_xyyaw = (1.0, 2.0, 0.0)
        self.controller._latest_map = object()

    def _handle(self, legs: int = 0) -> TaskHandle:
        return TaskHandle(
            task_id="test", started_at=time.time(), timeout_s=60.0,
            max_speed_m_s=0.1, legs_completed=legs,
        )

    def test_a_failure_in_the_task_ends_it_with_an_error(self) -> None:
        """A dead thread would leave the task exploring for good."""
        handle = self._handle()
        cancels = []
        self.controller._explore = lambda h: 1 / 0
        self.controller._nav_cancel_rpc = lambda run_id="": cancels.append(run_id)
        self.controller._run_task(handle)
        self.assertEqual(handle.state, "error")
        self.assertIn("ZeroDivisionError", handle.detail)
        self.assertEqual(len(cancels), 1)

    def test_where_it_has_been_follows_the_map_when_its_origin_moves(self) -> None:
        """A growing SLAM map moves its origin; the cells marked visited
        must stay where the robot was, not shift with the old indices."""
        from explore_skill.frontier import GridView
        self.controller._visited_xy = {(5, 10)}            # (1.0, 2.0) m
        for ox in (0.0, -4.0):
            gv = GridView(data=np.zeros((100, 160), dtype=np.int8), resolution=0.05,
                          origin_x=ox, origin_y=0.0, width=160, height=100)
            cells = self.controller._visited_cells(gv, self.controller._visited_xy)
            xs = [gv.cell_to_world(cx, cy)[0] for cx, cy in cells]
            self.assertLess(abs(min(xs) - 1.0 + self.controller.VISITED_RADIUS_M), 0.1)
            self.assertLess(abs(max(xs) - 1.0 - self.controller.VISITED_RADIUS_M), 0.1)

    def test_a_costmap_is_used_only_while_it_keeps_coming(self) -> None:
        gv, sent = object(), []
        cm = SimpleNamespace(header=SimpleNamespace(frame_id="map"))
        self.controller._latest_costmap = cm
        with patch("explore_skill.frontier.overlay_lethal", lambda g, c, tf: sent.append(c) or "with"):
            self.controller._costmap_received = time.monotonic()
            self.assertEqual(self.controller._with_costmap(gv, SimpleNamespace(header=SimpleNamespace(frame_id="map"))), "with")
            self.controller._costmap_received = time.monotonic() - 10
            self.assertIs(self.controller._with_costmap(gv, SimpleNamespace(header=SimpleNamespace(frame_id="map"))), gv)

    def test_navigation_timeout_cancels_the_accepted_run(self) -> None:
        calls = []

        def rpc(tool, args):
            calls.append((tool, args))
            if tool == "navigate":
                return {"accepted": True, "run_id": "run-42"}
            if tool == "status":
                return {"state": "RUNNING"}
            return {"accepted": True}

        self.controller._mcp_call_sync = rpc
        self.controller.NAV_POLL_PERIOD_S = 0.001
        ok, detail = self.controller._nav_navigate_blocking(
            1.0, 2.0, yaw=None, timeout_s=0.003,
            cancel_evt=self._handle(),
        )

        self.assertFalse(ok)
        self.assertEqual(detail, "leg timeout")
        self.assertIn(("cancel", {"run_id": "run-42"}), calls)

    def test_navigation_cancel_request_cancels_the_accepted_run(self) -> None:
        calls = []

        def rpc(tool, args):
            calls.append((tool, args))
            if tool == "navigate":
                return {"accepted": True, "run_id": "run-7"}
            return {"state": "RUNNING"}

        handle = self._handle()
        handle.cancel_requested = True
        self.controller._mcp_call_sync = rpc
        ok, detail = self.controller._nav_navigate_blocking(
            1.0, 2.0, yaw=None, timeout_s=1.0, cancel_evt=handle,
        )

        self.assertFalse(ok)
        self.assertEqual(detail, "canceled during nav")
        self.assertIn(("cancel", {"run_id": "run-7"}), calls)

    def _run_with_nav(self, nav, map_msg=None, pose=(5.0, 5.0, 0.0),
                      timeout_s=None):
        self.controller._latest_map = map_msg or _map_msg()
        self.controller._latest_pose_xyyaw = pose
        self.controller.LOOP_QUIET_PERIOD_S = 0.0
        self.controller._nav_navigate_blocking = nav
        handle = self._handle()
        if timeout_s is not None:
            handle.timeout_s = timeout_s
        self.controller._task = handle
        self.controller._run_task(handle)
        return handle

    def test_failed_frontier_is_not_retried(self) -> None:
        goals = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            goals.append((x, y, yaw))
            return False, "nav terminal: FAILED"

        handle = self._run_with_nav(nav)

        self.assertEqual(len(goals), 2)
        (ax, ay, _), (bx, by, _) = goals
        self.assertGreater(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5, 1.0)
        # Both frontiers failed: end now, without turning in place.
        self.assertEqual(handle.state, "error")
        self.assertIn("all tried: 2 unreachable", handle.detail)

    def test_a_run_that_reached_frontiers_is_done_despite_one_failure(self) -> None:
        goals = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            goals.append((x, y))
            if len(goals) == 1:
                return False, "nav terminal: FAILED"
            return True, "nav terminal: SUCCEEDED"

        handle = self._run_with_nav(nav)

        self.assertEqual(len(goals), 3)       # fail, visit, close look
        self.assertEqual(handle.state, "done")
        self.assertIn("1 unreachable, 1 still there after a visit",
                      handle.detail)

    def test_consecutive_nav_failures_end_the_task(self) -> None:
        legs = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            # Every frontier is unreachable, and there are always new
            # ones: forget earlier failures so a fresh target exists.
            cancel_evt.failed_targets.clear()
            if (x, y) != (5.0, 5.0):          # not a sweep at the pose
                legs.append((x, y))
            return False, "nav terminal: FAILED"

        handle = self._run_with_nav(nav)

        self.assertEqual(len(legs),
                         self.controller.MAX_CONSECUTIVE_NAV_FAILURES)
        self.assertEqual(handle.state, "error")
        self.assertIn("nav legs failed in a row", handle.detail)

    def test_a_frontier_still_there_after_a_visit_is_not_revisited(self) -> None:
        """Navigation succeeds but the map never changes (glass, or open
        space past the lidar's range)."""
        goals = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            goals.append((x, y))
            return True, "nav terminal: SUCCEEDED"

        handle = self._run_with_nav(nav)

        # Each frontier once, then once more from as close as the robot fits.
        self.assertEqual(len(goals), 4)
        self.assertEqual(handle.state, "done")
        self.assertIn("2 still there after a visit", handle.detail)

    def test_a_map_without_frontiers_is_done_without_moving(self) -> None:
        goals = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            goals.append((x, y))
            return True, "nav terminal: SUCCEEDED"

        handle = self._run_with_nav(nav, _map_msg(pockets=()))

        self.assertEqual(goals, [])
        self.assertEqual(handle.state, "done")

    def test_legs_face_the_frontier(self) -> None:
        sent = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            sent.append((x, y, yaw, cancel_evt.last_target_xy))
            return True, "nav terminal: SUCCEEDED"

        self._run_with_nav(nav)

        self.assertTrue(sent)
        for x, y, yaw, (fx, fy) in sent:
            self.assertIsNotNone(yaw)
            if math.hypot(fx - x, fy - y) > 0.1:   # a close look sits on it
                self.assertAlmostEqual(yaw, math.atan2(fy - y, fx - x),
                                       places=6)

    def test_a_frontier_out_of_local_range_is_still_explored(self) -> None:
        """The local radius is a preference: with nothing nearer, a far
        frontier is the next goal."""
        goals = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            goals.append((x, y))
            return True, "nav terminal: SUCCEEDED"

        self._run_with_nav(nav, _map_msg(pockets=((85, 85),)),
                           pose=(1.0, 1.0, 0.0))

        self.assertGreaterEqual(len(goals), 1)
        self.assertGreater(math.hypot(goals[0][0] - 1.0, goals[0][1] - 1.0),
                           self.controller.MAX_FRONTIER_DISTANCE_M)

    def test_a_long_leg_gets_time_for_its_length(self) -> None:
        legs = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            legs.append((x, y, timeout_s))
            return True, "nav terminal: SUCCEEDED"

        self.controller.NAV_GOAL_TIMEOUT_S = 10.0
        self._run_with_nav(nav, _map_msg(pockets=((85, 85),)),
                           pose=(1.0, 1.0, 0.0), timeout_s=3600.0)

        x, y, timeout_s = legs[0]
        # The path is searched from the robot's cell, so it can be a little
        # shorter than the straight line from the exact pose.
        self.assertGreater(timeout_s, 0.95 *
                           self.controller.LEG_TIMEOUT_PER_M_S *
                           math.hypot(x - 1.0, y - 1.0))

    def test_a_hung_mcp_call_times_out(self) -> None:
        from explore_skill import controller as ctl

        async def hang(tool, args):
            await asyncio.sleep(60)

        self.controller._mcp_call = hang
        with patch.object(ctl, "MCP_CALL_TIMEOUT_S", 0.05):
            started = time.time()
            self.assertEqual(self.controller._mcp_call_sync("status", {}), {})

            async def handler():
                return self.controller._mcp_call_sync("cancel", {})

            self.assertEqual(asyncio.run(handler()), {})
        self.assertLess(time.time() - started, 5.0)

    def test_mcp_call_works_inside_a_running_event_loop(self) -> None:
        async def fake_call(tool, args):
            return {"tool": tool, "args": args}

        self.controller._mcp_call = fake_call

        async def handler():
            return self.controller._mcp_call_sync("cancel", {"run_id": ""})

        self.assertEqual(asyncio.run(handler()),
                         {"tool": "cancel", "args": {"run_id": ""}})


if __name__ == "__main__":
    unittest.main()
