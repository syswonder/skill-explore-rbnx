import asyncio
import math
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np

from explore_skill.controller import ExploreController, TaskHandle


def _map_msg(pockets=((20, 60), (70, 60))):
    """OccupancyGrid-like message: a free 10 m room at 0.1 m/cell with a
    1 m unknown pocket, one frontier cluster, at each (row, column)."""
    data = np.zeros((100, 100), dtype=np.int8)
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
        self.controller.NAV_CANCEL_POLL_TIMEOUT_S = 0.0

    def _handle(self, legs: int = 0) -> TaskHandle:
        return TaskHandle(
            task_id="test", started_at=time.time(), timeout_s=60.0,
            max_speed_m_s=0.1, legs_completed=legs,
        )

    def test_a_failure_in_the_task_ends_it_with_an_error(self) -> None:
        """A dead thread would leave the task exploring for good."""
        handle = self._handle()
        handle.nav_run_id = "nav-exception"
        cancels = []
        self.controller._explore = lambda h: 1 / 0
        self.controller._nav_cancel_rpc = lambda run_id: (cancels.append(run_id) or True, "stopped")
        self.controller._run_task(handle)
        self.assertEqual(handle.state, "error")
        self.assertIn("ZeroDivisionError", handle.detail)
        self.assertEqual(cancels, ["nav-exception"])
        self.assertEqual(handle.nav_run_id, "")

    def test_finished_status_does_not_keep_increasing_elapsed_time(self) -> None:
        handle = self._handle()
        handle.started_at = 90.0
        self.controller._task = handle
        with patch("explore_skill.controller.time.time", return_value=100.0):
            self.controller._terminate(handle, "done", "finished")
        for queried_at in (110.0, 1000.0):
            with patch("explore_skill.controller.time.time", return_value=queried_at):
                self.assertEqual(self.controller.status(handle.task_id)["elapsed_s"], 10.0)

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
        cancel_sent = False

        def rpc(tool, args):
            nonlocal cancel_sent
            calls.append((tool, args))
            if tool == "navigate":
                return {"accepted": True, "run_id": "run-42"}
            if tool == "status":
                return {
                    "known": True,
                    "state": "CANCELED" if cancel_sent else "RUNNING",
                }
            cancel_sent = True
            return {"accepted": True}

        handle = self._handle()
        self.controller._mcp_call_sync = rpc
        self.controller.NAV_POLL_PERIOD_S = 0.001
        ok, detail = self.controller._nav_navigate_blocking(
            1.0, 2.0, yaw=None, timeout_s=0.003,
            cancel_evt=handle,
        )

        self.assertFalse(ok)
        self.assertEqual(detail, "leg timeout")
        self.assertEqual(calls[-2:], [
            ("cancel", {"run_id": "run-42"}),
            ("status", {"run_id": "run-42"}),
        ])
        self.assertEqual(handle.nav_run_id, "")
        self.assertEqual(handle.nav_cancel_error, "")

    def test_unverified_terminal_navigation_status_keeps_the_owned_run(self) -> None:
        for response in (
            {"state": "CANCELED"},
            {"known": False, "state": "SUCCEEDED"},
            "malformed status",
            ["CANCELED"],
        ):
            with self.subTest(response=response):
                handle = self._handle()
                self.controller._task = handle
                self.controller._latest_map = None
                now = [0.0]

                def rpc(tool, args):
                    if tool == "navigate":
                        return {"accepted": True, "run_id": "nav-unverified"}
                    if tool == "cancel":
                        return {"accepted": True}
                    self.assertEqual(args, {"run_id": "nav-unverified"})
                    return response

                def advance(seconds):
                    now[0] += seconds
                    self.assertLessEqual(now[0], 2.0)

                with (
                    patch.object(self.controller, "_mcp_call_sync", side_effect=rpc) as calls,
                    patch("explore_skill.controller.time.time", side_effect=lambda: now[0]),
                    patch("explore_skill.controller.time.sleep", side_effect=advance),
                ):
                    accepted, detail = self.controller._nav_navigate_blocking(
                        1.0, 2.0, yaw=None, timeout_s=2.0, cancel_evt=handle,
                    )

                self.assertFalse(accepted)
                self.assertEqual(detail, "leg timeout")
                self.assertEqual(handle.nav_run_id, "nav-unverified")
                self.assertEqual(handle.state, "exploring")
                self.assertIn(
                    "stop unconfirmed", self.controller.status(handle.task_id)["detail"],
                )
                calls.assert_any_call("cancel", {"run_id": "nav-unverified"})
                with patch("explore_skill.controller.threading.Thread") as worker:
                    with self.assertRaises(RuntimeError):
                        self.controller.start(
                            area_hint="", timeout_s=60.0, max_speed_m_s=0.1,
                        )
                    worker.assert_not_called()

    def test_pre_canceled_task_does_not_start_navigation(self) -> None:
        handle = self._handle()
        handle.cancel_requested = True
        with patch.object(self.controller, "_mcp_call_sync") as rpc:
            ok, detail = self.controller._nav_navigate_blocking(
                1.0, 2.0, yaw=None, timeout_s=1.0, cancel_evt=handle,
            )

        self.assertFalse(ok)
        self.assertIn("cancel", detail.lower())
        rpc.assert_not_called()

    def test_navigation_cancel_request_cancels_the_accepted_run(self) -> None:
        handle = self._handle()

        def rpc(tool, args):
            if tool == "navigate":
                handle.cancel_requested = True
                return {"accepted": True, "run_id": "run-7"}
            if tool == "status":
                return {"known": True, "state": "CANCELED"}
            return {"accepted": True}

        with patch.object(self.controller, "_mcp_call_sync", side_effect=rpc) as calls:
            ok, detail = self.controller._nav_navigate_blocking(
                1.0, 2.0, yaw=None, timeout_s=1.0, cancel_evt=handle,
            )

        self.assertFalse(ok)
        self.assertEqual(detail, "canceled during nav")
        self.assertEqual(calls.call_args_list[-2:], [
            call("cancel", {"run_id": "run-7"}),
            call("status", {"run_id": "run-7"}),
        ])
        self.assertFalse(handle.nav_starting)
        self.assertEqual(handle.nav_run_id, "")

    def test_transient_stop_resumes_upstream_leg_and_sweep_handling(self) -> None:
        from explore_skill.strategies import pick

        for trigger in ("costmap", "leg timeout", "sweep timeout", "sweep cancel"):
            with self.subTest(trigger=trigger):
                handle = self._handle()
                handle.started_at = handle.last_progress_t = 0.0
                handle.timeout_s = 180.0
                self.controller._task = handle
                self.controller._latest_map = _map_msg()
                self.controller._latest_pose_xyyaw = (5.0, 5.0, 0.0)
                self.controller.strategy = "frontier_greedy"
                now = [0.0]
                cancel_count = [0]
                starts = []
                reason = {
                    "costmap": "goal now inside an obstacle in the costmap",
                    "sweep cancel": "user-requested cancel",
                }.get(trigger, "leg timeout")

                def rpc(tool, args):
                    if tool == "navigate":
                        if starts:
                            self.assertEqual(cancel_count[0], 3)
                            self.assertFalse(handle.nav_cancel_error)
                            self.assertEqual(
                                handle.consecutive_nav_failures,
                                0 if trigger.startswith("sweep") else 1,
                            )
                        starts.append(args["goal"]["pose"]["position"])
                        self.assertLessEqual(len(starts), 2)
                        return {
                            "accepted": True,
                            "run_id": "nav-unresolved" if len(starts) == 1 else "nav-resumed",
                        }
                    if args["run_id"] == "nav-resumed":
                        self.assertEqual(tool, "status")
                        self.controller._latest_map = _map_msg(pockets=())
                        return {"known": True, "state": "SUCCEEDED"}
                    self.assertEqual(args, {"run_id": "nav-unresolved"})
                    if tool == "cancel":
                        cancel_count[0] += 1
                        return {"accepted": True}
                    if trigger == "sweep cancel" and not cancel_count[0]:
                        handle.cancel_requested = True
                    return {
                        "known": True,
                        "state": "CANCELED" if cancel_count[0] >= 3 else "RUNNING",
                    }

                def advance(seconds):
                    now[0] += seconds
                    self.assertLess(now[0], 70.0, "stop retry did not terminate")

                def observe_retry(seconds):
                    status = self.controller.status(handle.task_id)
                    self.assertEqual(status["state"], "exploring")
                    self.assertIn(reason, status["detail"])
                    self.assertIn("stop unconfirmed", status["detail"])
                    self.assertEqual(handle.nav_run_id, "nav-unresolved")
                    self.assertIsNone(handle.finished_at)
                    self.assertEqual(len(starts), 1)
                    self.assertEqual(handle.failed_targets, [])
                    with self.assertRaises(RuntimeError):
                        self.controller.start(
                            area_hint="", timeout_s=60.0, max_speed_m_s=0.1,
                        )
                    advance(seconds)
                    return False

                def choose_target(*args, **kwargs):
                    if trigger.startswith("sweep") and cancel_count[0] < 3:
                        return None
                    return pick(*args, **kwargs)

                with (
                    patch("explore_skill.controller.time.time", side_effect=lambda: now[0]),
                    patch("explore_skill.controller.time.sleep", side_effect=advance),
                    patch("explore_skill.strategies.pick", side_effect=choose_target),
                    patch.object(
                        self.controller, "_goal_blocked",
                        side_effect=lambda *_args: trigger == "costmap" and len(starts) == 1,
                    ),
                    patch.object(self.controller, "_mcp_call_sync", side_effect=rpc) as calls,
                    patch.object(
                        self.controller._stop_evt, "wait", side_effect=observe_retry,
                    ) as retry,
                ):
                    self.controller._run_task(handle)

                self.assertEqual(
                    handle.state, "canceled" if trigger == "sweep cancel" else "done",
                    handle.detail,
                )
                if trigger == "sweep cancel":
                    self.assertEqual(handle.detail, reason)
                else:
                    self.assertIn("no frontiers left", handle.detail)
                self.assertNotIn("stop unconfirmed", handle.detail)
                self.assertEqual(handle.nav_run_id, "")
                self.assertFalse(handle.nav_cancel_error)
                self.assertEqual(handle.finished_at, now[0])
                self.assertEqual(cancel_count[0], 3)
                self.assertEqual(
                    sum(rpc_call.args[0] == "navigate" for rpc_call in calls.call_args_list),
                    1 if trigger == "sweep cancel" else 2,
                )
                self.assertEqual(
                    len(handle.failed_targets), 0 if trigger.startswith("sweep") else 1,
                )
                self.assertEqual(handle.legs_completed, 0 if trigger == "sweep cancel" else 1)
                self.assertEqual(handle.consecutive_nav_failures, 0)
                if not trigger.startswith("sweep"):
                    self.assertNotEqual(starts[0], starts[1])
                retry.assert_called_once_with(self.controller.NAV_POLL_PERIOD_S)

    def test_delayed_stop_retains_the_upstream_consecutive_failure_limit(self) -> None:
        handle = self._handle()
        handle.consecutive_nav_failures = self.controller.MAX_CONSECUTIVE_NAV_FAILURES - 1
        self.controller._task = handle
        self.controller._latest_map = _map_msg()
        self.controller._latest_pose_xyyaw = (5.0, 5.0, 0.0)

        def observe_retry(_seconds):
            self.assertEqual(handle.state, "exploring")
            self.assertIsNone(handle.finished_at)
            self.assertEqual(handle.nav_run_id, "nav-last-failure")
            return False

        with (
            patch.object(self.controller, "_goal_blocked", return_value=True),
            patch.object(self.controller, "_mcp_call_sync", side_effect=[
                {"accepted": True, "run_id": "nav-last-failure"},
                {"accepted": True},
                {"known": True, "state": "RUNNING"},
                {"accepted": True},
                {"known": True, "state": "RUNNING"},
                {"accepted": True},
                {"known": True, "state": "CANCELED"},
            ]) as calls,
            patch.object(self.controller._stop_evt, "wait", side_effect=observe_retry),
        ):
            self.controller._run_task(handle)

        self.assertEqual(handle.state, "error")
        self.assertIn("5 nav legs failed in a row", handle.detail)
        self.assertIn("goal now inside an obstacle in the costmap", handle.detail)
        self.assertNotIn("stop unconfirmed", handle.detail)
        self.assertEqual(len(handle.failed_targets), 1)
        self.assertEqual(
            handle.consecutive_nav_failures, self.controller.MAX_CONSECUTIVE_NAV_FAILURES,
        )
        self.assertEqual(handle.nav_run_id, "")
        self.assertEqual(calls.call_count, 7)

    def test_task_interruption_during_stop_wait_prevents_replacement_motion(self) -> None:
        from explore_skill.strategies import pick

        for sweep in (False, True):
            for interruption in ("cancel", "timeout"):
                with self.subTest(sweep=sweep, interruption=interruption):
                    handle = self._handle()
                    handle.started_at = handle.last_progress_t = 0.0
                    handle.timeout_s = 180.0
                    self.controller._task = handle
                    self.controller._latest_map = _map_msg()
                    self.controller._latest_pose_xyyaw = (5.0, 5.0, 0.0)
                    self.controller.strategy = "frontier_greedy"
                    now = [0.0]
                    cancel_count = [0]

                    def rpc(tool, args):
                        if tool == "navigate":
                            return {"accepted": True, "run_id": "nav-interrupted"}
                        self.assertEqual(args, {"run_id": "nav-interrupted"})
                        if tool == "cancel":
                            cancel_count[0] += 1
                            return {"accepted": True}
                        return {
                            "known": True,
                            "state": "CANCELED" if cancel_count[0] >= 3 else "RUNNING",
                        }

                    def advance(seconds):
                        now[0] += seconds
                        self.assertLess(now[0], 185.0)

                    def interrupt(_seconds):
                        self.assertEqual(handle.state, "exploring")
                        self.assertIsNone(handle.finished_at)
                        self.assertEqual(handle.nav_run_id, "nav-interrupted")
                        if interruption == "cancel":
                            ok, detail = self.controller.cancel(handle.task_id)
                            self.assertTrue(ok, detail)
                        else:
                            now[0] = handle.timeout_s + 1.0
                        return False

                    def choose_target(*args, **kwargs):
                        return None if sweep else pick(*args, **kwargs)

                    with (
                        patch("explore_skill.controller.time.time", side_effect=lambda: now[0]),
                        patch("explore_skill.controller.time.sleep", side_effect=advance),
                        patch("explore_skill.strategies.pick", side_effect=choose_target),
                        patch.object(self.controller, "_goal_blocked", return_value=not sweep),
                        patch.object(self.controller, "_mcp_call_sync", side_effect=rpc) as calls,
                        patch.object(self.controller._stop_evt, "wait", side_effect=interrupt),
                    ):
                        self.controller._run_task(handle)

                    self.assertEqual(
                        handle.state, "canceled" if interruption == "cancel" else "timeout",
                    )
                    self.assertEqual(
                        handle.detail, "user-requested cancel" if interruption == "cancel"
                        else f"hit {handle.timeout_s}s ceiling",
                    )
                    self.assertEqual(handle.failed_targets, [])
                    self.assertEqual(handle.nav_run_id, "")
                    self.assertIsNotNone(handle.finished_at)
                    self.assertEqual(
                        sum(rpc_call.args[0] == "navigate" for rpc_call in calls.call_args_list),
                        1,
                    )

    def test_shutdown_during_leg_or_sweep_stop_retains_unconfirmed_ownership(self) -> None:
        from explore_skill.strategies import pick

        for sweep in (False, True):
            with self.subTest(sweep=sweep):
                handle = self._handle()
                handle.started_at = handle.last_progress_t = 0.0
                handle.timeout_s = 180.0
                self.controller._task = handle
                self.controller._stop_evt.clear()
                self.controller._latest_map = _map_msg()
                self.controller._latest_pose_xyyaw = (5.0, 5.0, 0.0)
                self.controller.strategy = "frontier_greedy"
                now = [0.0]

                def rpc(tool, args):
                    if tool == "navigate":
                        return {"accepted": True, "run_id": "nav-shutdown"}
                    self.assertEqual(args, {"run_id": "nav-shutdown"})
                    if tool == "cancel":
                        return {"accepted": True}
                    return {"known": True, "state": "RUNNING"}

                def advance(seconds):
                    now[0] += seconds
                    self.assertLess(now[0], 65.0, "shutdown did not exit stop waiting")

                def shut_down(_seconds):
                    self.controller.stop_runtime()
                    return self.controller._stop_evt.is_set()

                def choose_target(*args, **kwargs):
                    return None if sweep else pick(*args, **kwargs)

                with (
                    patch("explore_skill.controller.time.time", side_effect=lambda: now[0]),
                    patch("explore_skill.controller.time.sleep", side_effect=advance),
                    patch("explore_skill.strategies.pick", side_effect=choose_target),
                    patch.object(self.controller, "_mcp_call_sync", side_effect=rpc) as calls,
                    patch.object(self.controller._stop_evt, "wait", side_effect=shut_down),
                ):
                    self.controller._run_task(handle)

                self.assertEqual(handle.state, "exploring")
                self.assertIsNone(handle.finished_at)
                self.assertEqual(handle.nav_run_id, "nav-shutdown")
                self.assertTrue(handle.cancel_requested)
                self.assertIn("stop unconfirmed", self.controller.status(handle.task_id)["detail"])
                self.assertEqual(
                    sum(rpc_call.args[0] == "navigate" for rpc_call in calls.call_args_list), 1,
                )
                with self.assertRaises(RuntimeError):
                    self.controller.start(
                        area_hint="", timeout_s=60.0, max_speed_m_s=0.1,
                    )

    def test_failed_navigation_start_ends_locally_and_allows_a_new_exploration(self) -> None:
        failures = (
            RuntimeError("transport unavailable"),
            "timeout",
            {},
            None,
            ["malformed reply"],
            {"accepted": True},
            {"accepted": True, "run_id": ""},
            {"accepted": True, "run_id": " "},
            {"accepted": True, "run_id": 42},
        )
        self.controller.NAV_RPC_TIMEOUT_S = 0.02
        self.controller.LOOP_QUIET_PERIOD_S = 0.0
        for failure in failures:
            with self.subTest(failure=failure):
                self.controller._task = None
                self.controller._latest_map = _map_msg()
                self.controller._latest_pose_xyyaw = (5.0, 5.0, 0.0)

                async def failed_start(tool, args):
                    self.assertEqual(tool, "navigate")
                    if isinstance(failure, Exception):
                        raise failure
                    if failure == "timeout":
                        await asyncio.sleep(60.0)
                    return failure

                async def successful_navigation(tool, args):
                    if tool == "navigate":
                        return {"accepted": True, "run_id": "nav-restarted"}
                    self.assertEqual(tool, "status")
                    self.assertEqual(args, {"run_id": "nav-restarted"})
                    self.controller._latest_map = _map_msg(pockets=())
                    return {"known": True, "state": "SUCCEEDED"}

                with patch.object(
                    self.controller, "_mcp_call", new=AsyncMock(side_effect=failed_start),
                ) as rpc:
                    handle = self.controller.start(
                        area_hint="", timeout_s=60.0, max_speed_m_s=0.1,
                    )
                    handle.thread.join(timeout=2.0)
                    self.assertFalse(handle.thread.is_alive())
                    self.assertEqual(handle.state, "error", handle.detail)
                    self.assertFalse(handle.nav_starting)
                    self.assertEqual(handle.nav_run_id, "")
                    self.assertIn("outcome unknown", handle.detail)
                    self.assertIn("stop unconfirmed", handle.detail)
                    self.assertIsNotNone(handle.finished_at)
                    self.assertEqual(handle.failed_targets, [])
                    self.assertEqual(rpc.await_count, 1)

                    ok, detail = self.controller.cancel(handle.task_id)
                    self.assertFalse(ok, detail)
                    self.assertIn("stop unconfirmed", detail)
                    self.assertEqual(rpc.await_count, 1)

                    rpc.side_effect = successful_navigation
                    resumed = self.controller.start(
                        area_hint="", timeout_s=60.0, max_speed_m_s=0.1,
                    )
                    resumed.thread.join(timeout=2.0)
                    self.assertFalse(resumed.thread.is_alive())
                    self.assertEqual(resumed.state, "done", resumed.detail)
                    self.assertNotEqual(resumed.task_id, handle.task_id)
                    self.assertEqual(resumed.legs_completed, 1)
                    self.assertFalse(resumed.nav_start_error)
                    self.assertFalse(resumed.nav_starting)
                    self.assertEqual(resumed.nav_run_id, "")
                    self.assertEqual(rpc.await_count, 3)

    def test_navigation_start_exception_releases_the_pending_flag(self) -> None:
        handle = self._handle()
        self.controller._task = handle
        with patch.object(
            self.controller, "_mcp_call_sync", side_effect=RuntimeError("start failed"),
        ) as rpc:
            ok, detail = self.controller._nav_navigate_blocking(
                1.0, 2.0, yaw=None, timeout_s=1.0, cancel_evt=handle,
            )
            self.assertFalse(ok, detail)
            self.assertFalse(handle.nav_starting)
            self.assertIn("start failed", detail)
            self.controller._terminate(handle, "error", detail)
            ok, detail = self.controller.cancel(handle.task_id)
            self.assertFalse(ok, detail)
            self.assertIn("stop unconfirmed", detail)
            self.assertEqual(handle.state, "error")
            self.assertIsNotNone(handle.finished_at)
            self.assertEqual(rpc.call_count, 1)

    def test_cancel_during_a_failed_start_does_not_leave_a_pending_worker(self) -> None:
        for response in ({}, {"accepted": True}):
            with self.subTest(response=response):
                self.controller._task = None
                self.controller._latest_map = _map_msg()
                self.controller._latest_pose_xyyaw = (5.0, 5.0, 0.0)
                entered = threading.Event()
                release = threading.Event()

                def rpc_response(tool, args):
                    self.assertEqual(tool, "navigate")
                    entered.set()
                    if not release.wait(timeout=2.0):
                        raise TimeoutError("test did not release failed start")
                    return response

                with patch.object(
                    self.controller, "_mcp_call_sync", side_effect=rpc_response,
                ) as rpc:
                    handle = self.controller.start(
                        area_hint="", timeout_s=60.0, max_speed_m_s=0.1,
                    )
                    try:
                        self.assertTrue(entered.wait(timeout=2.0))
                        ok, detail = self.controller.cancel(handle.task_id)
                        self.assertFalse(ok, detail)
                        self.assertTrue(handle.nav_starting)
                    finally:
                        release.set()
                        handle.thread.join(timeout=2.0)
                    self.assertFalse(handle.thread.is_alive())
                    self.assertFalse(handle.nav_starting)
                    self.assertEqual(handle.state, "error", handle.detail)
                    self.assertIn("stop unconfirmed", handle.detail)
                    self.assertIsNotNone(handle.finished_at)
                    self.assertEqual(rpc.call_count, 1)

    def test_failed_sweep_start_does_not_launch_another_rotation(self) -> None:
        handle = self._handle()
        self.controller._task = handle
        with patch.object(
            self.controller, "_mcp_call_sync", return_value={"accepted": True},
        ) as rpc:
            self.controller._sweep_360(handle, deadline=None)
            self.controller._terminate(handle, "canceled", "user-requested cancel")

        self.assertFalse(handle.nav_starting)
        self.assertEqual(handle.state, "error")
        self.assertIn("without a valid run_id", handle.detail)
        self.assertIn("stop unconfirmed", handle.detail)
        self.assertEqual(rpc.call_count, 1)

    def test_uncertain_navigation_start_prevents_more_goals_in_the_same_task(self) -> None:
        handle = self._handle()
        with patch.object(self.controller, "_mcp_call_sync", return_value={}) as rpc:
            for position in (1.0, 2.0):
                ok, detail = self.controller._nav_navigate_blocking(
                    position, 2.0, yaw=None, timeout_s=1.0, cancel_evt=handle,
                )
                self.assertFalse(ok, detail)
                self.assertIn("stop unconfirmed", detail)
        self.assertFalse(handle.nav_starting)
        self.assertEqual(rpc.call_count, 1)

    def test_rejected_navigation_start_releases_ownership(self) -> None:
        handle = self._handle()
        with patch.object(self.controller, "_mcp_call_sync", side_effect=[
            {"accepted": False, "detail": "goal rejected"},
            {"accepted": True, "run_id": "nav-next"},
            {"known": True, "state": "SUCCEEDED"},
        ]):
            ok, detail = self.controller._nav_navigate_blocking(
                1.0, 2.0, yaw=None, timeout_s=1.0, cancel_evt=handle,
            )
            self.assertFalse(ok, detail)
            self.assertFalse(handle.nav_starting)
            self.assertEqual(handle.nav_run_id, "")
            ok, detail = self.controller._nav_navigate_blocking(
                3.0, 4.0, yaw=None, timeout_s=1.0, cancel_evt=handle,
            )

        self.assertTrue(ok, detail)
        self.assertFalse(handle.nav_starting)
        self.assertEqual(handle.nav_run_id, "")

    def test_cancel_forwards_only_the_owned_navigation_run_id(self) -> None:
        handle = self._handle()
        handle.nav_run_id = "nav-owned-42"
        self.controller._task = handle
        with patch.object(
            self.controller, "_mcp_call_sync",
            side_effect=[
                {"accepted": True, "detail": "cancel requested"},
                {"known": True, "state": "CANCELED"},
            ],
        ) as rpc:
            ok, detail = self.controller.cancel(handle.task_id)

        self.assertTrue(ok, detail)
        self.assertTrue(handle.cancel_requested)
        self.assertEqual(handle.nav_run_id, "")
        self.assertEqual(handle.nav_cancel_error, "")
        self.assertEqual(rpc.call_args_list, [
            call("cancel", {"run_id": "nav-owned-42"}),
            call("status", {"run_id": "nav-owned-42"}),
        ])

    def test_accepted_cancel_requires_known_terminal_status_before_clearing_id(self) -> None:
        for status in (
            {"known": True, "state": "RUNNING"},
            {"known": False, "state": "CANCELED"},
            {},
        ):
            with self.subTest(status=status):
                handle = self._handle()
                handle.nav_run_id = "nav-stop-unconfirmed"
                self.controller._task = handle
                with patch.object(
                    self.controller, "_mcp_call_sync", side_effect=[
                        {"accepted": True, "detail": "cancel requested"},
                        status,
                        {"accepted": True, "detail": "cancel requested"},
                        {"known": True, "state": "CANCELED"},
                    ],
                ) as rpc:
                    ok, detail = self.controller.cancel(handle.task_id)

                    self.assertFalse(ok, detail)
                    self.assertIn("unconfirmed", detail.lower())
                    self.assertTrue(handle.cancel_requested)
                    self.assertEqual(handle.nav_run_id, "nav-stop-unconfirmed")
                    self.assertIn("unconfirmed", handle.nav_cancel_error.lower())

                    ok, detail = self.controller.cancel(handle.task_id)

                self.assertTrue(ok, detail)
                self.assertEqual(handle.nav_run_id, "")
                self.assertEqual(handle.nav_cancel_error, "")
                self.assertEqual(rpc.call_args_list, [
                    call("cancel", {"run_id": "nav-stop-unconfirmed"}),
                    call("status", {"run_id": "nav-stop-unconfirmed"}),
                    call("cancel", {"run_id": "nav-stop-unconfirmed"}),
                    call("status", {"run_id": "nav-stop-unconfirmed"}),
                ])

    def test_cancel_polls_until_the_owned_navigation_is_known_terminal(self) -> None:
        self.controller.NAV_CANCEL_POLL_TIMEOUT_S = 0.6
        handle = self._handle()
        handle.nav_run_id = "nav-settling"
        self.controller._task = handle
        now = [0.0]

        def advance(seconds):
            self.assertEqual(handle.nav_run_id, "nav-settling")
            self.assertIsNone(handle.finished_at)
            now[0] += seconds

        with (
            patch.object(self.controller, "_mcp_call_sync", side_effect=[
                {"accepted": True},
                {"known": True, "state": "CANCELING"},
                {"known": False, "state": "CANCELED"},
                {"known": True, "state": "CANCELED"},
            ]) as rpc,
            patch("explore_skill.controller.time.monotonic", side_effect=lambda: now[0]),
            patch("explore_skill.controller.time.sleep", side_effect=advance),
        ):
            ok, detail = self.controller.cancel(handle.task_id)

        self.assertTrue(ok, detail)
        self.assertEqual(handle.nav_run_id, "")
        self.assertEqual(handle.nav_cancel_error, "")
        self.assertAlmostEqual(now[0], 0.4)
        self.assertEqual(rpc.call_args_list, [
            call("cancel", {"run_id": "nav-settling"}),
            *[call("status", {"run_id": "nav-settling"})] * 3,
        ])

    def test_cancel_polling_has_a_deadline_and_never_confirms_unknown_status(self) -> None:
        for response in (
            {"known": True, "state": "CANCELING"},
            {"known": True, "state": "RUNNING"},
            {"known": False, "state": "CANCELED"},
            {"state": "CANCELED"},
            {},
            None,
            ["CANCELED"],
        ):
            with self.subTest(response=response):
                self.controller.NAV_CANCEL_POLL_TIMEOUT_S = 0.4
                handle = self._handle()
                handle.nav_run_id = "nav-never-stopped"
                self.controller._task = handle
                now = [0.0]

                def advance(seconds):
                    now[0] += seconds
                    self.assertLessEqual(now[0], 0.4)

                def rpc_response(tool, args):
                    self.assertEqual(args, {"run_id": "nav-never-stopped"})
                    return {"accepted": True} if tool == "cancel" else response

                with (
                    patch.object(self.controller, "_mcp_call_sync", side_effect=rpc_response) as rpc,
                    patch("explore_skill.controller.time.monotonic", side_effect=lambda: now[0]),
                    patch("explore_skill.controller.time.sleep", side_effect=advance),
                ):
                    ok, detail = self.controller.cancel(handle.task_id)

                self.assertFalse(ok, detail)
                self.assertIn("unconfirmed", detail)
                self.assertEqual(handle.nav_run_id, "nav-never-stopped")
                self.assertIsNone(handle.finished_at)
                self.assertAlmostEqual(now[0], 0.4)
                self.assertEqual(rpc.call_count, 4)

    def test_worker_and_user_cancellation_are_serialized(self) -> None:
        handle = self._handle()
        handle.nav_run_id = "nav-shared"
        self.controller._task = handle
        cancel_entered = threading.Event()
        user_waiting = threading.Event()
        release_cancel = threading.Event()
        original_lock = self.controller._cancel_lock
        outcomes = []

        def rpc(tool, args):
            self.assertEqual(args, {"run_id": "nav-shared"})
            if tool == "cancel":
                cancel_entered.set()
                if not release_cancel.wait(timeout=2.0):
                    raise TimeoutError("test did not release cancellation")
                return {"accepted": True}
            return {"known": True, "state": "CANCELED"}

        class ObservedLock:
            def __enter__(self):
                if threading.current_thread() is caller:
                    user_waiting.set()
                return original_lock.__enter__()

            def __exit__(self, *args):
                return original_lock.__exit__(*args)

        worker = threading.Thread(
            target=lambda: outcomes.append(self.controller._cancel_navigation(handle)),
            daemon=True,
        )
        caller = threading.Thread(
            target=lambda: outcomes.append(self.controller.cancel(handle.task_id)),
            daemon=True,
        )
        with (
            patch.object(self.controller, "_cancel_lock", ObservedLock()),
            patch.object(self.controller, "_mcp_call_sync", side_effect=rpc) as calls,
        ):
            worker.start()
            try:
                self.assertTrue(cancel_entered.wait(timeout=2.0))
                caller.start()
                self.assertTrue(user_waiting.wait(timeout=2.0))
                self.assertEqual(handle.nav_run_id, "nav-shared")
            finally:
                release_cancel.set()
                worker.join(timeout=2.0)
                if caller.ident is not None:
                    caller.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertFalse(caller.is_alive())
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(all(accepted for accepted, _detail in outcomes))
        self.assertEqual(handle.nav_run_id, "")
        self.assertEqual(calls.call_args_list, [
            call("cancel", {"run_id": "nav-shared"}),
            call("status", {"run_id": "nav-shared"}),
        ])

    def test_cancel_with_mismatched_task_id_does_not_cancel_navigation(self) -> None:
        handle = self._handle()
        handle.nav_run_id = "nav-owned-42"
        self.controller._task = handle
        with patch.object(self.controller, "_mcp_call_sync") as rpc:
            ok, detail = self.controller.cancel("another-exploration")

        self.assertFalse(ok)
        self.assertTrue(detail)
        self.assertFalse(handle.cancel_requested)
        self.assertEqual(handle.nav_run_id, "nav-owned-42")
        rpc.assert_not_called()

    def test_cancel_failure_preserves_navigation_id_and_reports_error(self) -> None:
        failures = (
            {"accepted": False, "detail": "stop rejected"},
            {},
            None,
            RuntimeError("transport unavailable"),
            TimeoutError("cancel RPC timed out"),
        )
        for failure in failures:
            with self.subTest(failure=failure):
                handle = self._handle()
                handle.nav_run_id = "nav-unconfirmed"
                self.controller._task = handle

                def rpc_response(tool, args):
                    if tool == "status":
                        return {"known": True, "state": "RUNNING"}
                    if isinstance(failure, Exception):
                        raise failure
                    return failure

                with patch.object(
                    self.controller, "_mcp_call_sync", side_effect=rpc_response,
                ) as rpc:
                    ok, detail = self.controller.cancel(handle.task_id)

                self.assertFalse(ok, detail)
                self.assertTrue(detail)
                self.assertTrue(handle.cancel_requested)
                self.assertEqual(handle.nav_run_id, "nav-unconfirmed")
                self.assertTrue(handle.nav_cancel_error)
                rpc.assert_any_call("cancel", {"run_id": "nav-unconfirmed"})
                for rpc_call in rpc.call_args_list:
                    self.assertEqual(rpc_call.args[1], {"run_id": "nav-unconfirmed"})

    def test_failed_cancel_can_retry_after_local_terminal_state(self) -> None:
        for state in ("canceled", "error"):
            with self.subTest(state=state):
                handle = self._handle()
                handle.nav_run_id = "nav-retry"
                self.controller._task = handle
                with patch.object(
                    self.controller, "_mcp_call_sync",
                    return_value={"accepted": False, "detail": "not stopped"},
                ) as rpc:
                    ok, detail = self.controller.cancel(handle.task_id)
                    self.assertFalse(ok, detail)
                    handle.state = state
                    with patch("explore_skill.controller.threading.Thread") as worker:
                        with self.assertRaises(RuntimeError):
                            self.controller.start(
                                area_hint="", timeout_s=60.0, max_speed_m_s=0.1,
                            )
                        worker.assert_not_called()
                    self.assertIs(self.controller._task, handle)
                    self.assertEqual(handle.nav_run_id, "nav-retry")

                    rpc.side_effect = [
                        {"accepted": True, "detail": "cancel requested"},
                        {"known": True, "state": "CANCELED"},
                    ]
                    ok, detail = self.controller.cancel(handle.task_id)

                self.assertTrue(ok, detail)
                self.assertEqual(handle.nav_run_id, "")
                self.assertEqual(handle.nav_cancel_error, "")
                self.assertEqual(rpc.call_args_list, [
                    call("cancel", {"run_id": "nav-retry"}),
                    call("status", {"run_id": "nav-retry"}),
                    call("cancel", {"run_id": "nav-retry"}),
                    call("status", {"run_id": "nav-retry"}),
                ])

    def test_termination_retries_unresolved_stop_before_publishing_original_result(self) -> None:
        for state, reason in (
            ("done", "no frontiers left"),
            ("timeout", "hit 60s ceiling"),
            ("error", "map has not grown for 180s"),
            ("canceled", "user-requested cancel"),
        ):
            with self.subTest(state=state):
                handle = self._handle()
                handle.nav_run_id = "nav-not-stopped"
                self.controller._task = handle

                def observe_retry(_seconds):
                    status = self.controller.status(handle.task_id)
                    self.assertEqual(status["state"], "exploring")
                    self.assertIn(reason, status["detail"])
                    self.assertIn("stop unconfirmed", status["detail"])
                    self.assertIsNone(handle.finished_at)
                    self.assertEqual(handle.nav_run_id, "nav-not-stopped")
                    with patch("explore_skill.controller.threading.Thread") as worker:
                        with self.assertRaises(RuntimeError):
                            self.controller.start(
                                area_hint="", timeout_s=60.0, max_speed_m_s=0.1,
                            )
                        worker.assert_not_called()
                    return False

                with (
                    patch.object(self.controller, "_mcp_call_sync", side_effect=[
                        {"accepted": False, "detail": "stop rejected"},
                        {"known": True, "state": "RUNNING"},
                        {"accepted": True},
                        {"known": False, "state": "CANCELED"},
                        {"accepted": False, "detail": "run is already canceled"},
                        {"known": True, "state": "CANCELED"},
                    ]) as rpc,
                    patch.object(
                        self.controller._stop_evt, "wait", side_effect=observe_retry,
                    ) as retry,
                ):
                    self.controller._terminate(handle, state, reason)

                self.assertEqual(handle.state, state)
                self.assertEqual(handle.detail, reason)
                self.assertEqual(handle.nav_run_id, "")
                self.assertFalse(handle.nav_cancel_error)
                self.assertIsNotNone(handle.finished_at)
                self.assertEqual(
                    retry.call_args_list,
                    [call(self.controller.NAV_POLL_PERIOD_S)] * 2,
                )
                self.assertEqual(rpc.call_args_list, [
                    rpc_call for _ in range(3) for rpc_call in (
                        call("cancel", {"run_id": "nav-not-stopped"}),
                        call("status", {"run_id": "nav-not-stopped"}),
                    )
                ])

    def test_confirmed_cancel_during_stop_retry_preserves_failure_reason(self) -> None:
        handle = self._handle()
        handle.nav_run_id = "nav-timed-out"
        self.controller._task = handle

        def confirm_cancel(_seconds):
            accepted, detail = self.controller.cancel(handle.task_id)
            self.assertTrue(accepted, detail)
            self.assertEqual(handle.state, "exploring")
            return False

        with (
            patch.object(self.controller, "_mcp_call_sync", side_effect=[
                {"accepted": False},
                {"known": True, "state": "RUNNING"},
                {"accepted": True},
                {"known": True, "state": "CANCELED"},
            ]),
            patch.object(self.controller._stop_evt, "wait", side_effect=confirm_cancel),
        ):
            self.controller._terminate(handle, "timeout", "hit 60s ceiling")

        self.assertEqual(handle.state, "timeout")
        self.assertEqual(handle.detail, "hit 60s ceiling")
        self.assertEqual(handle.nav_run_id, "")

    def test_rejected_cancel_succeeds_only_for_known_terminal_navigation(self) -> None:
        for state in ("SUCCEEDED", "FAILED", "CANCELED", "TIMEOUT"):
            for known in (True, False):
                with self.subTest(state=state, known=known):
                    handle = self._handle()
                    handle.nav_run_id = "nav-terminal"
                    self.controller._task = handle
                    with patch.object(
                        self.controller, "_mcp_call_sync", side_effect=[
                            {"accepted": False, "detail": "no active goal"},
                            {"known": known, "state": state},
                        ],
                    ) as rpc:
                        ok, detail = self.controller.cancel(handle.task_id)

                    self.assertEqual(ok, known, detail)
                    self.assertEqual(
                        handle.nav_run_id, "" if known else "nav-terminal",
                    )
                    self.assertEqual(rpc.call_args_list, [
                        call("cancel", {"run_id": "nav-terminal"}),
                        call("status", {"run_id": "nav-terminal"}),
                    ])

    def test_cancel_without_navigation_is_a_harmless_noop(self) -> None:
        for state in (None, "exploring", "canceled", "error"):
            with self.subTest(state=state):
                handle = self._handle() if state is not None else None
                if handle is not None:
                    handle.state = state
                self.controller._task = handle
                with patch.object(self.controller, "_mcp_call_sync") as rpc:
                    ok, detail = self.controller.cancel(
                        handle.task_id if handle is not None else None,
                    )

                self.assertTrue(ok, detail)
                rpc.assert_not_called()

    def test_cancel_pending_navigation_never_sends_an_empty_run_id(self) -> None:
        handle = self._handle()
        handle.nav_starting = True
        self.controller._task = handle
        with patch.object(self.controller, "_mcp_call_sync") as rpc:
            ok, detail = self.controller.cancel(handle.task_id)

        self.assertFalse(ok)
        self.assertRegex(detail.lower(), "pending|unconfirmed|start")
        self.assertTrue(handle.cancel_requested)
        self.assertTrue(handle.nav_starting)
        self.assertEqual(handle.nav_run_id, "")
        self.assertTrue(handle.nav_cancel_error)
        rpc.assert_not_called()
        handle.state = "canceled"
        with patch("explore_skill.controller.threading.Thread") as worker:
            with self.assertRaises(RuntimeError):
                self.controller.start(
                    area_hint="", timeout_s=60.0, max_speed_m_s=0.1,
                )
            worker.assert_not_called()
        self.assertIs(self.controller._task, handle)

    def test_late_navigation_reply_is_canceled_by_its_exact_run_id(self) -> None:
        handle = self._handle()
        self.controller._task = handle
        navigate_entered = threading.Event()
        release_reply = threading.Event()
        outcomes = []

        def rpc_response(tool, args):
            if tool == "navigate":
                navigate_entered.set()
                if not release_reply.wait(timeout=2.0):
                    raise TimeoutError("test did not release navigation reply")
                return {"accepted": True, "run_id": "nav-late"}
            if tool == "cancel":
                return {"accepted": True, "detail": "cancel requested"}
            return {"known": True, "state": "CANCELED"}

        def navigate():
            outcomes.append(self.controller._nav_navigate_blocking(
                1.0, 2.0, yaw=None, timeout_s=0.5, cancel_evt=handle,
            ))

        with patch.object(
            self.controller, "_mcp_call_sync", side_effect=rpc_response,
        ) as rpc:
            worker = threading.Thread(target=navigate, daemon=True)
            worker.start()
            try:
                self.assertTrue(navigate_entered.wait(timeout=2.0))
                self.assertTrue(handle.nav_starting)
                ok, detail = self.controller.cancel(handle.task_id)
                self.assertFalse(ok, detail)
                self.assertEqual(rpc.call_count, 1)
            finally:
                release_reply.set()
                worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(outcomes), 1)
        self.assertFalse(outcomes[0][0], outcomes[0][1])
        self.assertIn(call("cancel", {"run_id": "nav-late"}), rpc.call_args_list)
        self.assertIn(call("status", {"run_id": "nav-late"}), rpc.call_args_list)
        for rpc_call in rpc.call_args_list:
            if rpc_call.args[0] in ("cancel", "status"):
                self.assertEqual(rpc_call.args[1], {"run_id": "nav-late"})
        self.assertEqual(handle.nav_run_id, "")
        self.assertFalse(handle.nav_starting)

    def test_async_handler_can_cancel_through_the_real_sync_bridge(self) -> None:
        handle = self._handle()
        handle.nav_run_id = "nav-async"
        self.controller._task = handle
        handler_thread = threading.get_ident()
        rpc_threads = []

        async def rpc_response(tool, args):
            rpc_threads.append(threading.get_ident())
            await asyncio.sleep(0)
            if tool == "status":
                return {"known": True, "state": "CANCELED"}
            return {"accepted": True, "detail": "cancel requested"}

        async def cancel_in_handler():
            return await asyncio.wait_for(
                asyncio.to_thread(self.controller.cancel, handle.task_id),
                timeout=2.0,
            )

        with patch.object(
            self.controller, "_mcp_call",
            new=AsyncMock(side_effect=rpc_response),
        ) as rpc:
            ok, detail = asyncio.run(cancel_in_handler())

        self.assertTrue(ok, detail)
        self.assertEqual(handle.nav_run_id, "")
        self.assertEqual(rpc.await_args_list, [
            call("cancel", {"run_id": "nav-async"}),
            call("status", {"run_id": "nav-async"}),
        ])
        self.assertEqual(len(rpc_threads), 2)
        for rpc_thread in rpc_threads:
            self.assertNotEqual(rpc_thread, handler_thread)

    def test_cancel_and_status_use_their_endpoints_and_independent_clients(self) -> None:
        self.controller._nav_endpoints["cancel"] = "http://cancel.invalid/mcp/"
        self.controller._nav_endpoints["status"] = "http://status.invalid/mcp/"
        clients = [MagicMock(), MagicMock()]
        for client in clients:
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            client.call_tool = AsyncMock(return_value=SimpleNamespace(
                content=[SimpleNamespace(text='{"accepted": true}')],
            ))
        client_factory = MagicMock(side_effect=clients)
        with patch.dict("sys.modules", {
            "fastmcp": SimpleNamespace(Client=client_factory),
        }):
            for tool, run_id in (("cancel", "nav-first"), ("status", "nav-second")):
                response = self.controller._mcp_call_sync(
                    tool, {"run_id": run_id},
                )
                self.assertEqual(response, {"accepted": True})

        self.assertEqual(client_factory.call_args_list, [
            call("http://cancel.invalid/mcp/"), call("http://status.invalid/mcp/"),
        ])
        for client, tool, run_id in zip(
            clients, ("cancel", "status"), ("nav-first", "nav-second"),
        ):
            client.call_tool.assert_awaited_once_with(tool, {"run_id": run_id})

    def test_real_sync_bridge_deadline_does_not_confirm_cancellation(self) -> None:
        handle = self._handle()
        handle.nav_run_id = "nav-slow"
        self.controller._task = handle
        self.controller.NAV_RPC_TIMEOUT_S = 0.01
        self.controller.NAV_CANCEL_POLL_TIMEOUT_S = 0.03

        async def slow_response(tool, args):
            await asyncio.sleep(0.25)
            return {"accepted": True, "detail": "late stop"}

        with patch.object(
            self.controller, "_mcp_call", new=AsyncMock(side_effect=slow_response),
        ) as rpc:
            started = time.monotonic()
            ok, detail = self.controller.cancel(handle.task_id)

        self.assertLess(time.monotonic() - started, 0.5)
        self.assertFalse(ok, detail)
        self.assertEqual(handle.nav_run_id, "nav-slow")
        self.assertTrue(handle.nav_cancel_error)
        rpc.assert_any_await("cancel", {"run_id": "nav-slow"})
        for rpc_call in rpc.await_args_list:
            self.assertIn(rpc_call.args[0], ("cancel", "status"))
            self.assertEqual(rpc_call.args[1], {"run_id": "nav-slow"})

    def test_navigation_controls_allow_slow_mcp_session_initialization(self) -> None:
        clients = [MagicMock(), MagicMock(), MagicMock()]
        for client in clients:
            async def initialize(client=client):
                await asyncio.sleep(1.05)
                return client

            client.__aenter__ = AsyncMock(side_effect=initialize)
            client.__aexit__ = AsyncMock(return_value=False)
            client.call_tool = AsyncMock(return_value=SimpleNamespace(
                content=[SimpleNamespace(text='{"accepted": true}')],
            ))
        with patch.dict("sys.modules", {
            "fastmcp": SimpleNamespace(Client=MagicMock(side_effect=clients)),
        }):
            for tool in ("navigate", "status", "cancel"):
                with self.subTest(tool=tool):
                    response = self.controller._mcp_call_sync(tool, {"run_id": "nav-slow-startup"})
                    self.assertEqual(response, {"accepted": True})

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
