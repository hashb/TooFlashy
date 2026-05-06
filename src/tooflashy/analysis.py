from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .thresholds import ThresholdProfile, wcag2_profile
from .video import iter_video_frames


_SRGB_VALUES = np.arange(256, dtype=np.float64) / 255.0
_SRGB_LINEAR_LUT = np.where(
    _SRGB_VALUES <= 0.04045,
    _SRGB_VALUES / 12.92,
    ((_SRGB_VALUES + 0.055) / 1.055) ** 2.4,
)


@dataclass(frozen=True)
class FlashEvent:
    kind: str
    frame_index: int
    time_seconds: float
    direction: int
    area_pixels: int


@dataclass(frozen=True)
class AnalysisResult:
    path: Path
    frame_count: int
    fps: float
    events: tuple[FlashEvent, ...] = field(default_factory=tuple)
    failures: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passes(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class _FrameFeatures:
    frame_index: int
    luminance: np.ndarray
    saturated_red: np.ndarray
    u: np.ndarray
    v: np.ndarray


def _linear_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(rgb)
    if values.dtype == np.uint8:
        return (
            _SRGB_LINEAR_LUT[values[..., 0]],
            _SRGB_LINEAR_LUT[values[..., 1]],
            _SRGB_LINEAR_LUT[values[..., 2]],
        )

    values = values.astype(np.float64, copy=False)
    if np.nanmax(values) > 1.0:
        values = values / 255.0
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    return linear[..., 0], linear[..., 1], linear[..., 2]


def _frame_features(rgb: np.ndarray, frame_index: int) -> _FrameFeatures:
    r, g, b = _linear_channels(rgb)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    total = r + g + b
    saturated_red = np.divide(r, total, out=np.zeros_like(total), where=total > 0) >= 0.8

    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    denom = x + 15 * y + 3 * z
    u = np.divide(4 * x, denom, out=np.zeros_like(denom), where=denom != 0)
    v = np.divide(9 * y, denom, out=np.zeros_like(denom), where=denom != 0)

    return _FrameFeatures(
        frame_index=frame_index,
        luminance=luminance,
        saturated_red=saturated_red,
        u=u,
        v=v,
    )


def _luminance_transition_direction_masks(
    previous_lum: np.ndarray,
    current_lum: np.ndarray,
    *,
    immediate_up: np.ndarray,
    immediate_down: np.ndarray,
    profile: ThresholdProfile,
) -> tuple[np.ndarray, np.ndarray]:
    darker = np.minimum(previous_lum, current_lum)
    brighter = np.maximum(previous_lum, current_lum)
    diff = brighter - darker
    denom = brighter + darker
    low_range = darker < 0.8 * profile.reference_luminance
    michelson = np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0)
    lum_threshold = 0.1 * profile.reference_luminance * profile.encoded_area_tolerance
    mask = (low_range & (diff >= lum_threshold)) | (
        ~low_range & (michelson >= 1 / 17)
    )
    return (
        mask & (current_lum > previous_lum) & immediate_up,
        mask & (current_lum < previous_lum) & immediate_down,
    )


def _red_transition_direction_masks(
    previous: _FrameFeatures,
    current: _FrameFeatures,
    *,
    min_ucs_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    one_saturated = np.logical_xor(previous.saturated_red, current.saturated_red)
    distance_sq = (current.u - previous.u) ** 2 + (current.v - previous.v) ** 2
    mask = one_saturated & (distance_sq >= min_ucs_distance * min_ucs_distance)
    return (
        mask & current.saturated_red & ~previous.saturated_red,
        mask & previous.saturated_red & ~current.saturated_red,
    )


def _dominant_event(
    kind: str,
    mask: np.ndarray,
    direction: np.ndarray,
    *,
    profile: ThresholdProfile,
    threshold_pixels: int,
    frame_index: int,
    fps: float,
) -> FlashEvent | None:
    if not np.any(mask):
        return None
    up = _area_count(mask & (direction > 0), profile)
    down = _area_count(mask & (direction < 0), profile)
    area = max(up, down)
    if area < threshold_pixels:
        return None
    return FlashEvent(
        kind=kind,
        frame_index=frame_index,
        time_seconds=frame_index / fps,
        direction=1 if up >= down else -1,
        area_pixels=area,
    )


def _event_from_direction_mask(
    kind: str,
    mask: np.ndarray,
    direction: int,
    *,
    profile: ThresholdProfile,
    threshold_pixels: int,
    frame_index: int,
    fps: float,
) -> FlashEvent | None:
    area = _area_count(mask, profile)
    if area < threshold_pixels:
        return None
    return FlashEvent(
        kind=kind,
        frame_index=frame_index,
        time_seconds=frame_index / fps,
        direction=direction,
        area_pixels=area,
    )


def _area_count(mask: np.ndarray, profile: ThresholdProfile) -> int:
    if profile.area_mode == "screen":
        return int(np.count_nonzero(mask))
    window_w, window_h = profile.window_size(mask.shape)
    if window_w >= mask.shape[1] and window_h >= mask.shape[0]:
        return int(np.count_nonzero(mask))
    counts = cv2.boxFilter(
        mask.astype(np.uint8),
        ddepth=cv2.CV_32S,
        ksize=(window_w, window_h),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    return int(counts.max())


def merge_fast_flash_events(
    events: list[FlashEvent] | tuple[FlashEvent, ...],
    *,
    fast_flash_spacing_ms: float = 15.0,
) -> list[FlashEvent]:
    """Merge fast-flash transition sequences as proposed in Table 7.

    A fast-flash sequence alternates direction, with every second transition
    (the next transition in the same direction) separated by 15 ms or less.
    Such a sequence counts as two transitions total for screening.
    """
    if len(events) < 3:
        return list(events)

    max_spacing = fast_flash_spacing_ms / 1000
    merged: list[FlashEvent] = []
    sequence: list[FlashEvent] = []

    def flush() -> None:
        nonlocal sequence
        if len(sequence) >= 3:
            first_by_direction: dict[int, FlashEvent] = {}
            for event in sequence:
                first_by_direction.setdefault(event.direction, event)
            merged.extend(sorted(first_by_direction.values(), key=lambda event: event.time_seconds))
        else:
            merged.extend(sequence)
        sequence = []

    for event in sorted(events, key=lambda item: item.time_seconds):
        if not sequence:
            sequence = [event]
            continue
        if len(sequence) == 1:
            if event.direction != sequence[-1].direction:
                sequence.append(event)
            else:
                flush()
                sequence = [event]
            continue

        is_alternating = event.direction != sequence[-1].direction
        same_direction_reference = sequence[-2]
        is_fast = event.time_seconds - same_direction_reference.time_seconds <= max_spacing
        if is_alternating and is_fast:
            sequence.append(event)
        else:
            flush()
            sequence = [event]

    flush()
    return merged


def count_failure_in_events(
    events: list[FlashEvent] | tuple[FlashEvent, ...],
    *,
    max_transition_counts_per_second: int = 6,
    fast_flash_spacing_ms: float = 15.0,
) -> bool:
    one_second_epsilon = 1e-9
    by_kind = {"luminance": [], "red": []}
    for event in events:
        by_kind[event.kind].append(event)

    for kind_events in by_kind.values():
        kind_events = merge_fast_flash_events(
            kind_events, fast_flash_spacing_ms=fast_flash_spacing_ms
        )
        for start_idx, start in enumerate(kind_events):
            window = [
                event
                for event in kind_events[start_idx:]
                if event.time_seconds - start.time_seconds < 1.0 - one_second_epsilon
            ]
            if len(window) <= max_transition_counts_per_second:
                continue
            alternating = [window[0]]
            for event in window[1:]:
                if event.direction != alternating[-1].direction:
                    alternating.append(event)
            if len(alternating) > max_transition_counts_per_second:
                return True
    return False


def analyze_video(
    path: str | Path,
    *,
    profile: ThresholdProfile | None = None,
    max_frames: int | None = None,
    reader: str = "ffmpeg",
) -> AnalysisResult:
    profile = profile or wcag2_profile()
    video_path = Path(path)
    fps, frames = iter_video_frames(video_path, engine=reader, max_frames=max_frames)
    return analyze_frames(frames, fps=fps, path=video_path, profile=profile)


def analyze_frames(
    frames: Iterable[np.ndarray],
    *,
    fps: float,
    path: str | Path,
    profile: ThresholdProfile | None = None,
) -> AnalysisResult:
    profile = profile or wcag2_profile()
    video_path = Path(path)
    threshold_pixels: int | None = None
    previous: _FrameFeatures | None = None
    history: list[_FrameFeatures] = []
    pending_masks: dict[tuple[str, int], list[tuple[float, np.ndarray]]] = {}
    events: list[FlashEvent] = []
    last_direction: dict[str, int] = {"luminance": 0, "red": 0}
    frame_index = 0
    max_frame_span = max(1, int(np.floor(profile.max_transition_duration_ms * fps / 1000)))

    for rgb in frames:
        current = _frame_features(rgb, frame_index)
        if threshold_pixels is None:
            threshold_pixels = profile.hazardous_area_pixels(rgb.shape)

        if previous is not None:
            lum_up = np.zeros(current.luminance.shape, dtype=bool)
            lum_down = np.zeros(current.luminance.shape, dtype=bool)
            red_up = np.zeros(current.luminance.shape, dtype=bool)
            red_down = np.zeros(current.luminance.shape, dtype=bool)
            immediate_lum_up = current.luminance > previous.luminance
            immediate_lum_down = current.luminance < previous.luminance
            if profile.count_red_transitions:
                immediate_red_up, immediate_red_down = _red_transition_direction_masks(
                    previous,
                    current,
                    min_ucs_distance=profile.min_red_ucs_distance,
                )
            else:
                immediate_red_up = np.zeros(current.luminance.shape, dtype=bool)
                immediate_red_down = np.zeros(current.luminance.shape, dtype=bool)

            for prior in history:
                if frame_index - prior.frame_index > max_frame_span:
                    continue

                prior_lum_up, prior_lum_down = _luminance_transition_direction_masks(
                    prior.luminance,
                    current.luminance,
                    immediate_up=immediate_lum_up,
                    immediate_down=immediate_lum_down,
                    profile=profile,
                )
                lum_up |= prior_lum_up
                lum_down |= prior_lum_down

                if profile.count_red_transitions:
                    prior_red_up, prior_red_down = _red_transition_direction_masks(
                        prior,
                        current,
                        min_ucs_distance=profile.min_red_ucs_distance,
                    )
                    red_up |= prior_red_up & immediate_red_up
                    red_down |= prior_red_down & immediate_red_down

            for kind, direction, mask in (
                ("luminance", 1, lum_up),
                ("luminance", -1, lum_down),
                ("red", 1, red_up),
                ("red", -1, red_down),
            ):
                event = _synchronized_event(
                    kind,
                    direction,
                    mask,
                    pending_masks=pending_masks,
                    profile=profile,
                    threshold_pixels=threshold_pixels,
                    frame_index=frame_index,
                    fps=fps,
                )
                if event is not None and last_direction[kind] != direction:
                    events.append(event)
                    last_direction[kind] = direction

        previous = current
        history.append(current)
        history = [item for item in history if frame_index - item.frame_index < max_frame_span]
        frame_index += 1

    failures: list[str] = []
    if count_failure_in_events(
        events,
        max_transition_counts_per_second=profile.max_transition_counts_per_second,
        fast_flash_spacing_ms=profile.fast_flash_spacing_ms,
    ):
        failures.append("more than six alternating qualifying transitions in a one-second span")

    return AnalysisResult(
        path=video_path,
        frame_count=frame_index,
        fps=fps,
        events=tuple(events),
        failures=tuple(failures),
    )


def _synchronized_event(
    kind: str,
    direction: int,
    mask: np.ndarray,
    *,
    pending_masks: dict[tuple[str, int], list[tuple[float, np.ndarray]]],
    profile: ThresholdProfile,
    threshold_pixels: int,
    frame_index: int,
    fps: float,
) -> FlashEvent | None:
    key = (kind, direction)
    current_time = frame_index / fps
    sync_seconds = profile.synchronized_window_ms / 1000
    pending = [
        (time_seconds, pending_mask)
        for time_seconds, pending_mask in pending_masks.get(key, [])
        if current_time - time_seconds <= sync_seconds
    ]
    pending_masks[key] = pending

    if not np.any(mask):
        return None

    combined = mask.copy()
    for _, pending_mask in pending:
        combined |= pending_mask

    event = _event_from_direction_mask(
        kind,
        combined,
        direction,
        profile=profile,
        threshold_pixels=threshold_pixels,
        frame_index=frame_index,
        fps=fps,
    )
    if event is not None:
        pending_masks[key] = []
        return event

    pending_masks.setdefault(key, []).append((current_time, mask.copy()))
    return None
