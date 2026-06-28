#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
# explore_rbnx container entrypoint.
#
# Single long-running process: explore_skill.atlas_bridge runs an
# rclpy node, registers 3 MCP capabilities with atlas, and serves
# them via FastMCP HTTP. The skill's task lifecycle (driving the
# robot through frontier-based exploration) is internal to that
# process; map / nav are atlas-resolved.
set -eo pipefail

source /opt/ros/humble/setup.bash

cd /explore

export PYTHONPATH="/explore:/explore/rbnx-build/codegen/proto_gen:/explore/rbnx-build/codegen/robonix_mcp_types:${PYTHONPATH:-}"
if [ -d /robonix-api ]; then
    export PYTHONPATH="/robonix-api:${PYTHONPATH}"
fi

mkdir -p /explore/rbnx-build/data

# Run the skill as PID 1 so its stdout/stderr flow straight to the container
# log and SIGTERM reaches it directly. Do NOT pipe through `sed` for a prefix:
# `sed` block-buffers in a pipe, so the skill's log lines never flushed to the
# container's stdout and `rbnx logs -t com.robonix.skill.explore` showed
# nothing. rbnx already tags each component's output, so no prefix is needed.
exec python3 -u -m explore_skill.atlas_bridge
