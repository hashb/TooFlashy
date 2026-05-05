import numpy as np

from tooflashy.analysis import analyze_frames
from tooflashy.thresholds import ThresholdProfile


def _frame(columns_on: int) -> np.ndarray:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    if columns_on:
        frame[:, :columns_on] = 255
    return frame


def test_same_direction_areas_within_20ms_are_summed() -> None:
    profile = ThresholdProfile(name="test", area_mode="screen", area_fraction=0.25)

    result = analyze_frames([_frame(0), _frame(15), _frame(30)], fps=120, path="memory", profile=profile)

    assert len(result.events) == 1
    assert result.events[0].area_pixels >= 3000


def test_same_direction_areas_outside_20ms_are_not_summed() -> None:
    profile = ThresholdProfile(name="test", area_mode="screen", area_fraction=0.25)

    result = analyze_frames([_frame(0), _frame(15), _frame(30)], fps=30, path="memory", profile=profile)

    assert result.events == ()
