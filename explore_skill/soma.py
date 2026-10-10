# SPDX-License-Identifier: MulanPSL-2.0
"""The robot's size, from Soma.

`robonix/system/soma/footprint` returns the collision polygon the
deployment declares in soma.yaml, with its inscribed and circumscribed
radii. Navigation configures its costmap from the same reply, so reading
it here keeps explore's idea of where the robot fits in step with nav's
instead of carrying a second copy of the robot's dimensions in config.
"""
from __future__ import annotations

import math

SOMA_FOOTPRINT_CONTRACT = "robonix/system/soma/footprint"

# Added to the inscribed radius. Soma's polygon is the bare body; nav pads
# it before planning (footprint_padding: 0.05 on the Lite3), so a gap that
# just fits the body is one nav will not drive through.
FOOTPRINT_MARGIN_M = 0.05


def radius_from_footprint(response) -> float:
    """Clearance radius for a GetFootprint reply: the inscribed radius plus
    the margin. Raises ValueError on a reply that is not a usable polygon."""
    points = [(float(p.x), float(p.y)) for p in response.points]
    inscribed = float(response.inscribed_radius_m)
    circumscribed = float(response.circumscribed_radius_m)
    if len(points) < 3:
        raise ValueError("Soma footprint has fewer than three points")
    if any(not math.isfinite(v) for p in points for v in p):
        raise ValueError("Soma footprint has a non-finite coordinate")
    if not (math.isfinite(inscribed) and math.isfinite(circumscribed)
            and 0.0 < inscribed <= circumscribed):
        raise ValueError("Soma footprint radii are invalid")
    return inscribed + FOOTPRINT_MARGIN_M


def fetch_robot_radius(endpoint: str, timeout_s: float = 5.0) -> float:
    """Call GetFootprint on `endpoint` and return radius_from_footprint."""
    import grpc
    import robonix_contracts_pb2_grpc as contracts_grpc  # type: ignore
    import soma_pb2  # type: ignore

    with grpc.insecure_channel(endpoint) as channel:
        stub = contracts_grpc.RobonixSystemSomaFootprintStub(channel)
        response = stub.GetFootprint(soma_pb2.GetFootprint_Request(),
                                     timeout=timeout_s)
    return radius_from_footprint(response)
