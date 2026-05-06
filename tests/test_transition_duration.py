import numpy as np

from tooflashy.analysis import analyze_frames
from tooflashy.thresholds import ThresholdProfile


def _gray(value: int) -> np.ndarray:
    return np.full((80, 80, 3), value, dtype=np.uint8)


def test_multi_step_luminance_ramp_within_66ms_counts_as_transition() -> None:
    frames = [_gray(0), _gray(60), _gray(90), _gray(60), _gray(0)]
    profile = ThresholdProfile(name="test", area_mode="screen")

    result = analyze_frames(frames, fps=60, path="memory", profile=profile)

    assert [event.direction for event in result.events] == [1, -1]


def test_slow_luminance_ramp_does_not_count_as_transition() -> None:
    frames = [_gray(0), _gray(45), _gray(60), _gray(72), _gray(100)]
    profile = ThresholdProfile(name="test", area_mode="screen", max_transition_duration_ms=66)

    result = analyze_frames(frames, fps=30, path="memory", profile=profile)

    assert result.events == ()
