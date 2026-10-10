import unittest

import numpy as np

from explore_skill.frontier import GridView, pick_target


def _grid() -> GridView:
    """10 m x 10 m free room at 0.1 m/cell with two unknown pockets.

    Each pocket is ringed by frontier cells, so the mean of its
    cluster sits inside the unknown pocket itself."""
    data = np.zeros((100, 100), dtype=np.int8)
    data[20:30, 60:70] = -1      # pocket A, centre (6.5, 2.5)
    data[70:80, 60:70] = -1      # pocket B, centre (6.5, 7.5)
    return GridView(data=data, resolution=0.1, origin_x=0.0, origin_y=0.0,
                    width=100, height=100)


class PickTargetTest(unittest.TestCase):
    def test_target_is_a_known_free_frontier_cell(self) -> None:
        gv = _grid()
        target = pick_target(gv, (5.0, 2.5), max_distance_m=8.0)
        self.assertIsNotNone(target)
        cx, cy = gv.world_to_cell(*target.centroid_xy)
        self.assertEqual(gv.data[cy, cx], 0)

    def test_frontier_near_failed_goal_is_skipped(self) -> None:
        gv = _grid()
        first = pick_target(gv, (5.0, 2.5), max_distance_m=8.0)
        self.assertLess(first.centroid_xy[1], 5.0)      # pocket A

        # The next map update moves the target a few cm; it must still
        # be treated as the same failed frontier.
        fx, fy = first.centroid_xy
        second = pick_target(gv, (5.0, 2.5), max_distance_m=8.0,
                             blocked_xy=[(fx + 0.05, fy - 0.05)])
        self.assertIsNotNone(second)
        self.assertGreater(second.centroid_xy[1], 5.0)  # pocket B

        both = [first.centroid_xy, second.centroid_xy]
        self.assertIsNone(pick_target(gv, (5.0, 2.5), max_distance_m=8.0,
                                      blocked_xy=both))


if __name__ == "__main__":
    unittest.main()
