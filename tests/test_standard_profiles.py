import json
from pathlib import Path

from tooflashy.analysis import analyze_video
from tooflashy.thresholds import (
    iso_9241_391_profile,
    itu_r_bt1702_profile,
    ofcom_2017_profile,
    profile_for_standard,
    wcag2_profile,
)


def test_profiles_encode_area_thresholds_from_paper_table() -> None:
    shape = (1080, 1920, 3)

    assert wcag2_profile().area_mode == "wcag-window"
    assert iso_9241_391_profile().area_mode == "screen"
    assert itu_r_bt1702_profile().area_mode == "screen"
    assert ofcom_2017_profile().area_mode == "screen"

    assert wcag2_profile().hazardous_area_pixels(shape) < iso_9241_391_profile().hazardous_area_pixels(shape)


def test_profile_lookup_rejects_unknown_standard() -> None:
    try:
        profile_for_standard("unknown")
    except ValueError as exc:
        assert "unknown standard" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_wide_area_benchmark_differs_by_standard(pse_media_repo: Path) -> None:
    json_file = pse_media_repo / "video_creation" / "wcagc_30fps_area01" / "f010f005.json"
    expected = json.loads(json_file.read_text())["expected_result"]
    video = json_file.parent / "videos" / f"{json_file.stem}.avi"

    assert video.exists()

    for standard, factory in {
        "wcag2_2": wcag2_profile,
        "iso": iso_9241_391_profile,
        "itu_r1702_4": itu_r_bt1702_profile,
        "ofcom2017": ofcom_2017_profile,
    }.items():
        result = analyze_video(video, profile=factory())
        assert result.passes is (expected[standard] == "pass"), standard
