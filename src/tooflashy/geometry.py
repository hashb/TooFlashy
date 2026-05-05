from __future__ import annotations

import math


def visual_angle(*, size: float, distance: float) -> float:
    """Return the full visual angle in radians for an object of physical size x."""
    if distance <= 0:
        raise ValueError("distance must be positive")
    return 2 * math.atan(size / (2 * distance))


def solid_angle_cone(*, peripheral_angle: float) -> float:
    """Return solid angle in steradians for a right circular cone."""
    return 2 * math.pi * (1 - math.cos(peripheral_angle))


def visual_cortex_percent(peripheral_angle: float) -> float:
    """Binnie et al. model percentage of visual cortex stimulated.

    The paper expresses the peripheral angle in degrees in the fitted exponent.
    """
    degrees = math.degrees(peripheral_angle)
    return 100 * (1 - math.exp(-0.0574 * degrees))


def proportion_affected(visual_cortex_percentage: float) -> float:
    """Appendix model for response proportion before full-screen normalization."""
    q = visual_cortex_percentage
    if q > 56.37:
        return 1.0
    if q > 8.75:
        return 0.21 * q - 0.1827
    return 0.0
