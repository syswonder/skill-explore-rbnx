#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
# explore_rbnx build phase — runs `rbnx codegen` (emits proto_gen for
# both atlas-side stubs and the package's own srv definitions under
# capabilities/srv/), then `docker build`. Same shape as mapping_rbnx
# and system/scene.
#
# RBNX_BUILD_CLEAN=1 wipes rbnx-build/ and rebuilds without docker cache.
set -euo pipefail

PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG"

BUILD="rbnx-build"
CLEAN="${RBNX_BUILD_CLEAN:-}"
IMG="${ROBONIX_EXPLORE_IMAGE:-robonix-explore}"

if [[ "$CLEAN" == "1" ]]; then
    echo "[build] clean: removing $BUILD"
    rm -rf "$BUILD"
fi
mkdir -p "$BUILD/data"

# ── 1. Codegen (atlas proto stubs + this package's capabilities/srv/*.srv) ──
if command -v rbnx >/dev/null 2>&1; then
    FLAGS=()
    [[ "$CLEAN" == "1" ]] && FLAGS+=(--clean)
    echo "[build] rbnx codegen --mcp ${FLAGS[*]}"
    rbnx codegen -p "$PKG" --mcp "${FLAGS[@]}"
else
    echo "[build] WARNING: rbnx not in PATH — skipping proto codegen"
    echo "[build]   install robonix-cli + run \`rbnx setup\` once from the robonix source root"
fi

# ── 2. Docker image ─────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
    echo "[build] error: docker not found on PATH" >&2
    exit 1
fi

DOCKER_BUILD_FLAGS=(--network=host)
[[ "$CLEAN" == "1" ]] && DOCKER_BUILD_FLAGS+=(--no-cache)

if [[ "$CLEAN" != "1" ]] && docker image inspect "$IMG" >/dev/null 2>&1; then
    echo "[build] image $IMG present; rebuilding incrementally"
fi

echo "[build] docker build -f docker/Dockerfile -t $IMG"
docker build "${DOCKER_BUILD_FLAGS[@]}" -f docker/Dockerfile -t "$IMG" docker/

echo "[build] done."
