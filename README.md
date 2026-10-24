# explore_rbnx

Autonomous frontier exploration skill for robonix. Driven by the
LLM/pilot via `rbnx ask "explore"` (or any equivalent intent that
triggers the `robonix/skill/explore/explore` tool).

See `CAPABILITY.md` for the full interface + behaviour spec.

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

## Cancellation

The cancel tool runs navigation RPCs outside the MCP server's event loop.
It cancels the exact navigation `run_id` owned by this exploration, never
an empty id that could target another caller's goal. An initial navigation
request still awaiting its id reports cancellation pending; the returned id
is canceled as soon as it arrives.

Navigate, status and cancel each allow five seconds for the whole MCP
round trip, including opening and initializing a fresh session. After sending
cancel, the controller polls the same run's status every 200 ms for up to
two seconds. A status RPC already in flight can finish after that polling
window, but still has its five-second deadline. `CANCELING` is not terminal.
Only `known=true` with `SUCCEEDED`, `FAILED`, `CANCELED` or `TIMEOUT` confirms
completion, even if the cancel acknowledgement was lost or rejected.

If an owned run cannot be confirmed terminal, cancellation returns
`ok=false` with an explicit `stop unconfirmed` message. The controller retains
that id and its worker retries once per second after each bounded attempt.
Status stays `RUNNING` with the original termination reason and
`stop unconfirmed` until navigation reports a known terminal state.
A new exploration cannot overwrite that unresolved navigation run.
Once confirmed, ordinary leg and sweep failures resume their existing retry
handling. For task-ending conditions, the original terminal state and reason
are reported, and elapsed time freezes.

### Missing navigation start result

The start-pending flag is cleared on every completed or failed navigate
request. An explicit `accepted=false` is an ordinary rejected leg.
A transport exception, a startup timeout, an unreadable response, or
`accepted=true` without a nonempty string `run_id` leaves the remote outcome
unknown: navigation might have started before the reply was lost.

In that case, exploration sends no further goals or sweeps, finishes locally
as `FAILED`, and reports `navigation start outcome unknown` and
`stop unconfirmed`. Cancel returns promptly with `ok=false` and the same
warning. It cannot confirm a remote stop without an owned run id; it never
uses an empty id or a "latest run" lookup that could target another caller.
The local worker exits, elapsed time freezes, and the controller can accept a
new exploration instead of remaining locked indefinitely.

Accepting a new exploration does not prove that the previous remote goal
stopped. Before retrying after an unknown start result, the operator must
verify navigation or stop motion through the deployment's safety controls.
This skill cannot reconcile a lost start result without a navigation
protocol that preserves request ownership across retries.

## Tests

Run from the package root with Python, NumPy, SciPy and pytest installed:

```bash
python3 -m pytest -q tests
```

The control-flow regressions exercise failed and timed out starts, local
termination and restart, delayed cancellation, unconfirmed status, and exact
navigation run ownership. Frontier selection and exploration policy tests
remain part of the same suite.

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
