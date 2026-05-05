import math

import numpy as np

from tooflashy.color import (
    cie1976_ucs_distance,
    is_red_transition,
    is_saturated_red,
    relative_luminance,
    wcag20_red_critical_value,
)
from tooflashy.geometry import (
    proportion_affected,
    solid_angle_cone,
    visual_angle,
    visual_cortex_percent,
)
from tooflashy.thresholds import luminance_transition_qualifies, michelson_contrast


def test_saturated_red_uses_linear_rgb_ratio_from_paper_table() -> None:
    assert is_saturated_red((0xFF, 0x00, 0x00))
    assert is_saturated_red((0x47, 0x00, 0x00))
    assert not is_saturated_red((0xFF, 0x80, 0x80))
    assert not is_saturated_red((0x00, 0x00, 0x00))


def test_wcag20_red_critical_value_matches_equation() -> None:
    assert wcag20_red_critical_value((255, 0, 0)) == 320
    assert wcag20_red_critical_value((0, 255, 0)) == 0
    assert wcag20_red_critical_value((71, 0, 0)) > 20


def test_wcag22_red_transition_uses_ucs_distance() -> None:
    assert cie1976_ucs_distance((255, 0, 0), (0, 0, 255)) > 0.2
    assert is_red_transition((255, 0, 0), (0, 0, 255))
    assert not is_red_transition((255, 0, 0), (255, 54, 54))


def test_relative_luminance_uses_srgb_transfer_function() -> None:
    assert math.isclose(relative_luminance((255, 255, 255)), 1.0)
    assert math.isclose(relative_luminance((0, 0, 0)), 0.0)
    assert np.isclose(relative_luminance((255, 0, 0)), 0.2126)


def test_luminance_threshold_switches_from_difference_to_michelson() -> None:
    assert luminance_transition_qualifies(0.0, 0.1, reference_luminance=1.0)
    assert not luminance_transition_qualifies(0.0, 0.09, reference_luminance=1.0)
    assert math.isclose(michelson_contrast(180, 200), 20 / 380)
    assert not luminance_transition_qualifies(0.81, 0.86, reference_luminance=1.0)
    assert luminance_transition_qualifies(0.81, 0.92, reference_luminance=1.0)


def test_appendix_viewing_angle_and_risk_model() -> None:
    angle = visual_angle(size=1.0, distance=1.0)
    assert math.isclose(angle, math.radians(53.13010235415598))

    solid = solid_angle_cone(peripheral_angle=math.radians(5.0))
    assert math.isclose(solid, 2 * math.pi * (1 - math.cos(math.radians(5.0))))

    cortex = visual_cortex_percent(math.radians(5.0))
    assert 0 < cortex < 100
    assert proportion_affected(0) == 0
    assert proportion_affected(60) == 1
