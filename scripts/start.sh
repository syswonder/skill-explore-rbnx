#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
# explore_rbnx start phase — docker-run wrapper.
#
# Container shape: --network host + --ipc=host so the skill can
# subscribe to /map (mapping container) and call the nav
# service's gRPC endpoint without DDS isolation getting in the way.
#
# Trap: when boot SIGTERMs our PGID, this trap stops the container so
# the skill doesn't outlive the deploy.
set -euo pipefail

CT="${ROBONIX_EXPLORE_CONTAINER:-robonix_explore}"
IMG="${ROBONIX_EXPLORE_IMAGE:-robonix-explore}"

cleanup() {
    docker stop "$CT" >/dev/null 2>&1 || true
    kill -- "-$$" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

docker rm -f "$CT" >/dev/null 2>&1 || true

mkdir -p rbnx-build/data

declare -a EXTRA_MOUNTS=()
if [[ -n "${RBNX_CONFIG_FILE:-}" ]]; then
    EXTRA_MOUNTS+=(-v "${RBNX_CONFIG_FILE}:${RBNX_CONFIG_FILE}:ro")
fi

declare -a ZENOH_ARGS=()
if [[ -n "${ROBONIX_ZENOH_ROUTER:-}" ]]; then
    ZENOH_ARGS=(-e "ROBONIX_ZENOH_ROUTER=${ROBONIX_ZENOH_ROUTER}")
fi
if [[ -n "${ROBONIX_ZENOH_MODE:-}" ]]; then
    ZENOH_ARGS+=(-e "ROBONIX_ZENOH_MODE=${ROBONIX_ZENOH_MODE}")
fi
if [[ -n "${ROBONIX_ZENOH_LISTEN:-}" ]]; then
    ZENOH_ARGS+=(-e "ROBONIX_ZENOH_LISTEN=${ROBONIX_ZENOH_LISTEN}")
fi

exec docker run --rm \
    --name "$CT" \
    --network host \
    --ipc=host \
    -e ROBONIX_ATLAS="${ROBONIX_ATLAS:-127.0.0.1:50051}" \
    -e ROBONIX_CAPABILITY_ID="${ROBONIX_CAPABILITY_ID:-com.robonix.skill.explore}" \
    -e ROBONIX_PKG_HOST_DIR="$(pwd)" \
    -e RBNX_CONFIG_FILE="${RBNX_CONFIG_FILE:-}" \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}" \
    "${ZENOH_ARGS[@]}" \
    -v "$(pwd)":/explore \
    -v "$(rbnx path robonix-api)":/robonix-api:ro \
    "${EXTRA_MOUNTS[@]}" \
    "$IMG"
