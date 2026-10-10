import asyncio
import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from explore_skill.controller import ExploreController, TaskHandle


def _map_msg():
    """OccupancyGrid-like message: free 10 m x 10 m room with two
    unknown pockets, i.e. two separate frontier clusters."""
    data = np.zeros((100, 100), dtype=np.int8)
    data[20:30, 60:70] = -1
    data[70:80, 60:70] = -1
    origin = SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))
    info = SimpleNamespace(width=100, height=100, resolution=0.1,
                           origin=origin)
    return SimpleNamespace(info=info, data=data.tobytes())


class _Grid:
    @staticmethod
    def world_to_cell(_x, _y):
        return (4, 7)


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

    @patch("explore_skill.frontier.GridView.from_msg", return_value=_Grid())
    def test_sweep_requires_periodic_leg_and_missing_coverage(self, _grid) -> None:
        self.controller.SWEEP_EVERY_N_LEGS = 3
        self.controller._viewed_sectors[(4, 7)] = set()
        self.assertFalse(self.controller._should_sweep(self._handle(1)))
        self.assertFalse(self.controller._should_sweep(self._handle(2)))
        self.assertTrue(self.controller._should_sweep(self._handle(3)))

        self.controller._viewed_sectors[(4, 7)] = set(range(6))
        self.assertFalse(self.controller._should_sweep(self._handle(3)))

    @patch("explore_skill.frontier.GridView.from_msg", return_value=_Grid())
    def test_periodic_sweep_is_off_by_default(self, _grid) -> None:
        """A 360° lidar gains nothing from turning in place."""
        self.controller._viewed_sectors[(4, 7)] = set()
        for legs in range(1, 10):
            self.assertFalse(self.controller._should_sweep(self._handle(legs)))

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


    def _run_with_nav(self, nav, map_msg=None):
        self.controller._latest_map = map_msg or _map_msg()
        self.controller._latest_pose_xyyaw = (5.0, 5.0, 0.0)
        self.controller.LOOP_QUIET_PERIOD_S = 0.0
        self.controller._nav_navigate_blocking = nav
        handle = self._handle()
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
        # Both frontiers failed: give up at once instead of turning in
        # place or looping until the global timeout.
        self.assertEqual(handle.state, "error")
        self.assertIn("all tried: 2 unreachable", handle.detail)

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
        """Nav succeeds but the map never changes: glass, or open space
        past the lidar's range. Each frontier is visited once, then the
        task ends instead of creeping toward it 10 cm at a time."""
        goals = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            goals.append((x, y))
            return True, "nav terminal: SUCCEEDED"

        handle = self._run_with_nav(nav)

        self.assertEqual(len(goals), 2)
        self.assertEqual(handle.state, "done")
        self.assertIn("2 still there after a visit", handle.detail)

    def test_a_map_without_frontiers_is_done_without_moving(self) -> None:
        data = np.zeros((100, 100), dtype=np.int8)
        msg = _map_msg()
        msg.data = data.tobytes()
        goals = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            goals.append((x, y))
            return True, "nav terminal: SUCCEEDED"

        handle = self._run_with_nav(nav, msg)

        self.assertEqual(goals, [])
        self.assertEqual(handle.state, "done")

    def test_legs_face_the_frontier(self) -> None:
        """No heading meant yaw 0, and the robot turned to face map +X at
        the end of every leg."""
        sent = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            sent.append((x, y, yaw, cancel_evt.last_target_xy))
            return True, "nav terminal: SUCCEEDED"

        self._run_with_nav(nav)

        self.assertTrue(sent)
        for x, y, yaw, (fx, fy) in sent:
            self.assertIsNotNone(yaw)
            self.assertAlmostEqual(yaw, math.atan2(fy - y, fx - x), places=6)

    def test_a_frontier_out_of_local_range_is_still_explored(self) -> None:
        """The 6 m local radius is a preference. With nothing nearer, the
        far end of a corridor is the next goal, not a reason to stop."""
        data = np.zeros((100, 100), dtype=np.int8)
        data[85:95, 85:95] = -1
        msg = _map_msg()
        msg.data = data.tobytes()
        goals = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            goals.append((x, y))
            return True, "nav terminal: SUCCEEDED"

        self.controller._latest_pose_xyyaw = (1.0, 1.0, 0.0)
        self.controller._latest_map = msg
        self.controller.LOOP_QUIET_PERIOD_S = 0.0
        self.controller._nav_navigate_blocking = nav
        handle = self._handle()
        self.controller._task = handle
        self.controller._run_task(handle)

        self.assertEqual(len(goals), 1)
        self.assertGreater(math.hypot(goals[0][0] - 1.0, goals[0][1] - 1.0),
                           self.controller.MAX_FRONTIER_DISTANCE_M)

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
