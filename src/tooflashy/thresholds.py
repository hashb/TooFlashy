from __future__ import annotations

from dataclasses import dataclass


def michelson_contrast(low: float, high: float) -> float:
    darker = min(low, high)
    brighter = max(low, high)
    denom = brighter + darker
    return 0.0 if denom == 0 else (brighter - darker) / denom


def luminance_transition_qualifies(
    a: float,
    b: float,
    *,
    reference_luminance: float = 1.0,
    michelson_threshold: float = 1 / 17,
) -> bool:
    darker = min(a, b)
    brighter = max(a, b)
    if darker < 0.8 * reference_luminance:
        return (brighter - darker) >= 0.1 * reference_luminance
    return michelson_contrast(darker, brighter) >= michelson_threshold


@dataclass(frozen=True)
class ThresholdProfile:
    name: str
    reference_luminance: float = 1.0
    max_transition_duration_ms: float = 66.0
    synchronized_window_ms: float = 20.0
    fast_flash_spacing_ms: float = 15.0
    max_transition_counts_per_second: int = 6
    min_red_ucs_distance: float = 0.2
    area_mode: str = "wcag-window"
    area_window: tuple[int, int] = (341, 256)
    area_fraction: float = 0.25
    encoded_area_tolerance: float = 0.995

    def hazardous_area_pixels(self, frame_shape: tuple[int, int] | tuple[int, int, int]) -> int:
        height, width = frame_shape[:2]
        if self.area_mode == "screen":
            pixels = width * height * self.area_fraction
            return max(1, int(round(pixels * self.encoded_area_tolerance)))
        window_w, window_h = self.window_size(frame_shape)
        pixels = window_w * window_h * self.area_fraction
        return max(1, int(round(pixels * self.encoded_area_tolerance)))

    def window_size(self, frame_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int]:
        height, width = frame_shape[:2]
        if self.area_mode == "proposed-window":
            return min(width, 416), min(height, 416)
        if self.area_mode == "wcag-window":
            return min(width, self.area_window[0]), min(height, self.area_window[1])
        if self.area_mode == "screen":
            return width, height
        raise ValueError(f"unknown area mode {self.area_mode!r}")


@dataclass(frozen=True)
class NhkJbaRule:
    category: str
    max_flashes_per_second: float
    max_duration_seconds: float | None = None
    sustained_max_flashes_per_second: float | None = None


def _nhk_rule_for_area(category: str, area_fraction: float) -> NhkJbaRule | None:
    if category == "moderate":
        if 0.25 <= area_fraction <= 1.0:
            return NhkJbaRule(
                "moderate",
                max_flashes_per_second=5,
                max_duration_seconds=2,
                sustained_max_flashes_per_second=3,
            )
        return None
    if category == "intermediate":
        if 0.25 <= area_fraction < 0.80:
            return NhkJbaRule("intermediate", max_flashes_per_second=3)
        return None
    if category == "scene-change":
        if 0.80 <= area_fraction <= 1.0:
            return NhkJbaRule("scene-change", max_flashes_per_second=1.5)
        return None
    raise ValueError(f"unknown NHK/JBA category {category!r}")


def classify_nhk_jba_sdr(
    *, brightness_change_percent: float, area_fraction: float
) -> NhkJbaRule | None:
    """Classify an SDR transition using the NHK/JBA ranges summarized in Table 4."""
    if brightness_change_percent < 10:
        return None
    if brightness_change_percent <= 20:
        return _nhk_rule_for_area("moderate", area_fraction)
    return _nhk_rule_for_area("scene-change", area_fraction) or _nhk_rule_for_area(
        "intermediate", area_fraction
    )


def classify_nhk_jba_hdr(
    *, darker_luminance: float, brighter_luminance: float, area_fraction: float
) -> NhkJbaRule | None:
    """Classify an HDR transition using the NHK/JBA ranges summarized in Table 4."""
    darker = min(darker_luminance, brighter_luminance)
    brighter = max(darker_luminance, brighter_luminance)
    difference = brighter - darker
    if darker < 160:
        if difference < 20:
            return None
        if difference <= 40:
            return _nhk_rule_for_area("moderate", area_fraction)
        return _nhk_rule_for_area("scene-change", area_fraction) or _nhk_rule_for_area(
            "intermediate", area_fraction
        )

    if darker <= 0:
        return None
    brightness_ratio = difference / darker
    if brightness_ratio < 1 / 8:
        return None
    if brightness_ratio <= 1 / 4:
        return _nhk_rule_for_area("moderate", area_fraction)
    return _nhk_rule_for_area("scene-change", area_fraction) or _nhk_rule_for_area(
        "intermediate", area_fraction
    )


def wcag2_profile() -> ThresholdProfile:
    return ThresholdProfile(name="wcag2.2")


def proposed_profile() -> ThresholdProfile:
    return ThresholdProfile(name="jordan-vanderheiden-2024-proposed", area_mode="proposed-window")


def iso_9241_391_profile() -> ThresholdProfile:
    return ThresholdProfile(name="iso-9241-391", area_mode="screen")


def itu_r_bt1702_profile() -> ThresholdProfile:
    return ThresholdProfile(name="itu-r-bt.1702", area_mode="screen")


def ofcom_2017_profile() -> ThresholdProfile:
    return ThresholdProfile(name="ofcom-2017", area_mode="screen")


def expert_consensus_profile() -> ThresholdProfile:
    return ThresholdProfile(name="expert-consensus-2005", area_mode="screen")


def nhk_jba_sdr_profile() -> ThresholdProfile:
    return ThresholdProfile(name="nhk-jba-sdr-2020", area_mode="screen")


def profile_for_standard(name: str) -> ThresholdProfile:
    normalized = name.strip().lower().replace("_", "-")
    profiles = {
        "wcag": wcag2_profile,
        "wcag2": wcag2_profile,
        "wcag2.2": wcag2_profile,
        "wcag2-2": wcag2_profile,
        "wcag2-1": wcag2_profile,
        "wcag2-2-sc-2.3.1": wcag2_profile,
        "proposed": proposed_profile,
        "jordan-vanderheiden-2024": proposed_profile,
        "trace24": proposed_profile,
        "iso": iso_9241_391_profile,
        "iso-9241-391": iso_9241_391_profile,
        "itu": itu_r_bt1702_profile,
        "itu-r1702": itu_r_bt1702_profile,
        "itu-r1702-4": itu_r_bt1702_profile,
        "itu-r-bt.1702": itu_r_bt1702_profile,
        "ofcom": ofcom_2017_profile,
        "ofcom2017": ofcom_2017_profile,
        "expert": expert_consensus_profile,
        "expert-consensus": expert_consensus_profile,
        "nhk-jba": nhk_jba_sdr_profile,
        "nhk-jba-sdr": nhk_jba_sdr_profile,
    }
    try:
        return profiles[normalized]()
    except KeyError as exc:
        raise ValueError(f"unknown standard profile {name!r}") from exc
