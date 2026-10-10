# SPDX-License-Identifier: MulanPSL-2.0
"""The robot radius comes from Soma's footprint reply."""
from types import SimpleNamespace as NS

import pytest

from explore_skill.soma import FOOTPRINT_MARGIN_M, radius_from_footprint


def _reply(points, inscribed, circumscribed):
    return NS(points=[NS(x=x, y=y) for x, y in points],
              inscribed_radius_m=inscribed,
              circumscribed_radius_m=circumscribed)


LITE3 = [(0.305, 0.185), (0.305, -0.185), (-0.305, -0.185), (-0.305, 0.185)]


def test_lite3_footprint_gives_half_the_padded_width():
    r = radius_from_footprint(_reply(LITE3, 0.185, 0.357))
    assert r == pytest.approx(0.185 + FOOTPRINT_MARGIN_M)
    assert r == pytest.approx(0.235)


@pytest.mark.parametrize("points,inscribed,circumscribed", [
    (LITE3[:2], 0.185, 0.357),                 # not a polygon
    (LITE3, 0.0, 0.357),                       # no inscribed radius
    (LITE3, 0.4, 0.357),                       # inscribed > circumscribed
    (LITE3, float("nan"), 0.357),
    ([(0.3, float("inf"))] + LITE3[1:], 0.185, 0.357),
])
def test_unusable_footprints_are_rejected(points, inscribed, circumscribed):
    with pytest.raises(ValueError):
        radius_from_footprint(_reply(points, inscribed, circumscribed))
