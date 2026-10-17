---
description: Autonomous frontier exploration — drives the robot to map and catalog its surroundings (occupancy grid + semantic object registry).
---

# explore_rbnx — autonomous environment exploration skill

Drives the robot through frontier-based exploration of its surroundings,
building a spatial occupancy map (via the mapping service) and a
semantic object registry (via scene's built-in VLM detector) along the
way. User-invocable via the LLM/pilot.

## Interface (3 MCP tools)

### `robonix/skill/explore/explore`

Start an exploration task. Returns immediately with a `run_id`; poll
`status` to track progress.

| param            | type   | default | meaning                                    |
|------------------|--------|---------|--------------------------------------------|
| `area_hint`      | string | `""`    | free-form ("this room", "ground floor")    |
| `timeout_s`      | float  | 600     | hard ceiling, 0 = no timeout               |
| `max_speed_m_s`  | float  | 0.25    | safety cap on forward speed during explore |

Returns `{accepted, run_id, message}`. `accepted=false` if a task is
already running.

### `robonix/skill/explore/explore/status`

Poll an exploration's progress. Empty `run_id` returns the most
recent task.

Returns `{known, state, area_m2, frontiers_left, elapsed_s, eta_s, detail}`.
`state ∈ {PENDING | RUNNING | SUCCEEDED | FAILED | CANCELED | TIMEOUT}`
(terminal: SUCCEEDED / FAILED / CANCELED / TIMEOUT).

### `robonix/skill/explore/explore/cancel`

Abort the active exploration. Idempotent.

## Usage pattern (IMPORTANT — thread the run_id)

1. Call `explore` ONCE. It returns immediately with a `run_id`. **Save that
   exact `run_id`.**
2. To monitor, call `status` with that SAME `run_id` — repeatedly, until
   `state` is a terminal value (`SUCCEEDED | FAILED | CANCELED | TIMEOUT`). Do
   not call `explore` again to monitor; that starts nothing new (a task is
   already running) and loses your handle.
3. To stop it, call `cancel` with that SAME `run_id`.

Passing an empty `run_id` uses the most recent exploration task.

## Behaviour

1. Wait for the occupancy grid (the mapping service publishes it).
2. Loop until done:
   - Find frontier cells (free cells next to unknown) and cluster them.
     Cells the robot has driven over or cannot see under itself count as
     free; the navigation costmap's lethal cells and Scene's furniture
     count as obstacles.
   - Choose a frontier with the configured strategy: `frontier_greedy`
     (default) scores each by its size over the distance to it; `mrtsp`
     orders them as Liu et al. (2025) do and visits the first. Frontiers
     already reached or failed are skipped. With `min_passage_m` set, only
     frontiers in space reached through openings at least that wide count,
     so the robot keeps to the corridor and does not enter rooms through
     their doors.
   - Send the robot to a goal short of the frontier, where its chassis
     fits, facing it, through `service/navigation/navigate`; poll status.
3. End `done` when no frontier is left (with `min_passage_m`, none in the
   corridor) or every remaining one has been
   tried; `error` when navigation fails 5 legs in a row, the map has not
   grown for 180 s, or (with `frontier_greedy`) no frontier is reachable
   even after turning in place.

## What this skill does NOT do

- No SLAM (mapping service does that — atlas-resolved input).
- No collision avoidance (navigation service does that — atlas-resolved).
- No object detection (scene service does that, in-process VLM, on
  the camera frames published by the camera primitive).

The skill is purely a goal-selection loop with safety pre-filters. All
sensor/actuator access is through atlas-resolved contracts; no
hardcoded inter-package topic names.

## Dependencies it Connect()s on atlas at startup

| key             | contract                                | transport |
|-----------------|------------------------------------------|-----------|
| map_topic       | robonix/service/map/occupancy_grid       | ROS2      |
| nav_navigate    | robonix/service/navigation/navigate      | MCP       |
| nav_status      | robonix/service/navigation/navigate/status | MCP     |
| nav_cancel      | robonix/service/navigation/navigate/cancel | MCP     |

Skill refuses to start if any of these is missing — there is no
hardcoded fallback (packaging-spec invariant #1).

Optional, used when atlas has them and skipped otherwise:

| contract                              | used for                                   |
|---------------------------------------|--------------------------------------------|
| robonix/service/navigation/costmap    | obstacles the occupancy grid lacks (table tops a camera sees) |
| robonix/system/scene/list_objects     | furniture as keep-out circles              |
| robonix/system/soma/footprint         | the robot's radius                          |

## Portability

The skill code is identical between webots and the real robot. What
differs is which packages back the consumed contracts:

- webots: mapping=rtabmap (2D lidar + RGBD), nav=simple_nav, chassis=tiago_chassis
- jetson AGX + Mid360: mapping=DLIO (3D livox + IMU), nav=nav2-wrapper, chassis=ranger_chassis

Both back the same `service/map/occupancy_grid` + `service/navigation/*`
contracts; explore neither knows nor cares which is running.
