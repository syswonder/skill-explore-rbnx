# explore simulator

A 2D simulator for the explore skill. It runs the real `ExploreController`
(`_run_task`, frontier selection, termination) against a simulated robot,
lidar, RTAB-Map and Nav2, on a simulated clock, so a 20-minute exploration
takes seconds and needs neither ROS nor Webots.

```sh
python3 -m sim                      # every world
python3 -m sim --world office -v    # one world, with the controller's log
python3 -m sim --scene              # give the controller Scene's table boxes
python3 -m sim --strategy mrtsp     # the `strategy` config key; default nearest
python3 -m sim --png out/           # one picture per run
```

Each run prints its end state, coverage of the reachable free area,
distance, total turning, legs, failed legs, collisions, Nav2 recoveries and
the closest the robot came to a real obstacle.

In the pictures: grey unknown, white mapped free, black mapped obstacle,
orange tabletops and cyan glass (ground truth, invisible to the lidar),
blue trajectory, green goals that succeeded, red goals that failed,
magenta collisions.

## What is modelled, and from where

Parameters live in `robot.py` and come from the Lite3 deployment
(`nav2_params_lite3.yml`, `rtabmap_params.yaml`) and the MID-360 data sheet.

- **World** (`world.py`): walls the lidar sees; tabletops and glass it does
  not see but the robot hits. Tables are four visible legs under an
  invisible top.
- **Lidar and map** (`sensing.py`): 720 rays, 6 m range, free along a ray up
  to its return. Nothing within 0.5 m of the robot is marked free: the
  lowest beam does not reach the ground there, which is the unknown disc
  seen on the robot. The map updates once a second while moving.
- **Nav2** (`nav.py`): costmap from the map plus an obstacle layer that
  clears what the scan sees within 3 m; lethal cells inflated by the
  inscribed radius; unknown not traversable; NavFn's 0.5 m goal tolerance;
  1 Hz replanning; the default BT's recovery on a failed plan; a final turn
  to the goal heading, where no heading means yaw 0, as in the nav service.
- **Not modelled**: localisation drift, dynamic obstacles, the local
  controller's dynamics, map origin changes, the camera.

## Adding a world

Add a function to `world.py` that draws with `_Builder` and register it in
`WORLDS`. Worlds are small on purpose: each should reproduce one situation
the robot got into.
