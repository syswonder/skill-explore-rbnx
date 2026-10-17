# explore_rbnx

Autonomous frontier exploration skill for robonix. Driven by the
LLM/pilot via `rbnx ask "explore"` (or any equivalent intent that
triggers the `robonix/skill/explore/explore` tool).

See `CAPABILITY.md` for the full interface + behaviour spec.

Exploration duration, first-map wait, navigation legs, sweeps, settling and
cached obstacle data age use the deployment's selected `reference`. The
controller's ROS node uses simulation time for `ros_time`. Status queries,
cancellation requests, failure backoff and shutdown joins retain real limits.
Reference interruption ends tasks, including those without a duration limit,
and cancels their navigation runs. Old map, pose, costmap, Scene cache and
visited history cannot continue in a new generation. Progress deadlines stay
frozen during a pause. A current map with no frontiers can finish immediately.
Activation waits for reference time before subscribing to the map so that a
latched map arriving at startup is retained.

## Quickstart

```bash
# from the package root
bash scripts/build.sh   # rbnx codegen + docker build
# then add to your deploy manifest under skill: and rbnx boot
```

Manifest snippet:

```yaml
skill:
  - name: explore
    path: ../../../explore_rbnx
    config: {}
```

The skill registers 3 MCP tools with atlas. Pilot's tool catalog
will include `robonix/skill/explore/{explore, status, cancel}`.

## Layout

```
explore_rbnx/
├── package_manifest.yaml    # 3 caps + build/start hooks
├── capabilities/            # package-local TOMLs + .srv files
│   ├── explore.v1.toml
│   ├── status.v1.toml
│   ├── cancel.v1.toml
│   └── srv/{Explore,GetExploreStatus,CancelExplore}.srv
├── explore_skill/           # python module
│   ├── atlas_bridge.py      # entrypoint: register + serve MCP
│   ├── controller.py        # task state machine and frontier loop
│   ├── frontier.py          # frontier search, clustering, goals, costmap overlay
│   ├── strategies.py        # frontier_greedy and mrtsp
│   ├── map_opt.py           # map cleaning for mrtsp
│   └── soma.py              # robot radius from Soma's footprint
├── scripts/{build,start}.sh
├── docker/{Dockerfile, entrypoint.sh, no_shm_profile.xml}
├── CAPABILITY.md            # LLM-readable spec (registered with atlas)
└── README.md
```

License: MulanPSL-2.0
