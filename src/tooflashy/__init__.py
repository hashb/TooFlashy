"""Photosensitive epilepsy hazard screening helpers.

This package implements the actionable algorithms and equations described in
Jordan and Vanderheiden (2024), "International Guidelines for Photosensitive
Epilepsy: Gap Analysis and Recommendations".
"""

from .analysis import AnalysisResult, FlashEvent, analyze_video
from .color import (
    cie1976_ucs_distance,
    is_red_transition,
    is_saturated_red,
    relative_luminance,
    srgb_to_linear,
    wcag20_red_critical_value,
)
from .geometry import (
    proportion_affected,
    solid_angle_cone,
    visual_angle,
    visual_cortex_percent,
)
from .thresholds import ThresholdProfile, wcag2_profile

__all__ = [
    "AnalysisResult",
    "FlashEvent",
    "ThresholdProfile",
    "analyze_video",
    "cie1976_ucs_distance",
    "is_red_transition",
    "is_saturated_red",
    "proportion_affected",
    "relative_luminance",
    "solid_angle_cone",
    "srgb_to_linear",
    "visual_angle",
    "visual_cortex_percent",
    "wcag20_red_critical_value",
    "wcag2_profile",
]
