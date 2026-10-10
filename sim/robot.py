# SPDX-License-Identifier: MulanPSL-2.0
"""Robot, lidar and Nav2 parameters, taken from the Lite3 deployment.

Sources: robot-deep_robotics-lite3/config/nav2_params_lite3.yml and
rtabmap_params.yaml, and the Livox MID-360 data sheet.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobotParams:
    # Footprint 0.61 x 0.37 m + 0.05 padding. The planner keeps the centre
    # INSCRIBED away from lethal cells; collisions are checked against the
    # circumscribed circle, which overstates the corners a little.
    inscribed_m: float = 0.235
    circumscribed_m: float = 0.43
    max_vel_m_s: float = 0.35          # FollowPath max_vel_x
    max_yaw_rate: float = 0.6          # max_vel_theta
    xy_goal_tolerance: float = 0.15
    yaw_goal_tolerance: float = 0.15
    planner_tolerance_m: float = 0.5   # NavFn tolerance
    inflation_radius_m: float = 0.9    # global costmap
    costmap_raytrace_m: float = 3.0    # obstacle layer raytrace_max_range
    replan_period_s: float = 1.0       # default BT replans at 1 Hz
    progress_timeout_s: float = 10.0   # movement_time_allowance
    # The default BT on failure: wait 5 s, back up, spin 90°, ~30 s in all.
    # The Lite3 deploy file names a BT without spin and backup, but Humble
    # ignores the key it uses, so the default one runs.
    recovery_s: float = 30.0
    recovery_spin_rad: float = 1.57
    # Lidar (MID-360) and mapper (RTAB-Map).
    lidar_range_m: float = 6.0         # Grid/RangeMax
    lidar_rays: int = 720
    # The ground right around the robot is never observed: the lowest beam
    # is -7° from the top of the body, and Grid/RangeMin is 0.25.
    blind_radius_m: float = 0.5
    map_period_s: float = 1.0          # Rtabmap/DetectionRate
    # Noise, off by default; `python3 -m sim --noise` turns on the values
    # in NOISY. A scan is integrated at a pose off by scan_pose_sigma_m (SLAM
    # jitter), each ray is lost with lidar_dropout, and each return moves
    # along its ray by lidar_range_sigma_m.
    scan_pose_sigma_m: float = 0.0
    lidar_dropout: float = 0.0
    lidar_range_sigma_m: float = 0.0


# MID-360 range noise is about 2 cm; 3 cm of pose jitter is typical of
# RTAB-Map between loop closures. Dropout stands in for dark or shiny
# surfaces that return nothing.
NOISY = dict(scan_pose_sigma_m=0.03, lidar_dropout=0.05,
             lidar_range_sigma_m=0.02)
