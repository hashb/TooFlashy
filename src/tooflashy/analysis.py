from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .color import red_transition_mask, relative_luminance
from .thresholds import ThresholdProfile, wcag2_profile
from .video import read_video_frames


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


def _luminance_transition_mask(
    previous_lum: np.ndarray, current_lum: np.ndarray, profile: ThresholdProfile
) -> tuple[np.ndarray, np.ndarray]:
    darker = np.minimum(previous_lum, current_lum)
    brighter = np.maximum(previous_lum, current_lum)
    diff = brighter - darker
    low_range = darker < 0.8 * profile.reference_luminance
    michelson = np.divide(diff, brighter + darker, out=np.zeros_like(diff), where=(brighter + darker) != 0)
    mask = (low_range & (diff >= 0.1 * profile.reference_luminance)) | (
        ~low_range & (michelson >= 1 / 17)
    )
    direction = np.where(current_lum > previous_lum, 1, -1)
    return mask, direction


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
                if event.time_seconds - start.time_seconds < 1.0
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
    fps, frames = read_video_frames(video_path, engine=reader, max_frames=max_frames)
    threshold_pixels: int | None = None
    previous_rgb: np.ndarray | None = None
    previous_lum: np.ndarray | None = None
    events: list[FlashEvent] = []
    frame_index = 0

    for rgb in frames:
        lum = relative_luminance(rgb)
        if threshold_pixels is None:
            threshold_pixels = profile.hazardous_area_pixels(rgb.shape)

        if previous_rgb is not None and previous_lum is not None:
            lum_mask, lum_direction = _luminance_transition_mask(previous_lum, lum, profile)
            lum_event = _dominant_event(
                "luminance",
                lum_mask,
                lum_direction,
                profile=profile,
                threshold_pixels=threshold_pixels,
                frame_index=frame_index,
                fps=fps,
            )
            if lum_event is not None:
                events.append(lum_event)

            red_mask, red_direction = red_transition_mask(
                previous_rgb, rgb, min_ucs_distance=profile.min_red_ucs_distance
            )
            red_event = _dominant_event(
                "red",
                red_mask,
                red_direction,
                profile=profile,
                threshold_pixels=threshold_pixels,
                frame_index=frame_index,
                fps=fps,
            )
            if red_event is not None:
                events.append(red_event)

        previous_rgb = rgb
        previous_lum = lum
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
