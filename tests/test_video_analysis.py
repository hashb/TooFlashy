from pathlib import Path

import pytest

from pse_media_cases import PSE_VIDEO_CASES, collect_pse_video_cases
from tooflashy.analysis import analyze_frames
from tooflashy.thresholds import profile_for_standard
from tooflashy.video import read_video_frames


def test_all_pse_media_csv_cases_are_discovered(pse_media_repo: Path) -> None:
    cases = collect_pse_video_cases(pse_media_repo)

    assert cases
    assert len(cases) == sum(1 for _ in (pse_media_repo / "video_creation").glob("*/*.json"))


def test_all_pse_media_csv_cases_have_prebuilt_videos(pse_media_repo: Path) -> None:
    missing = [case.video_path for case in collect_pse_video_cases(pse_media_repo) if not case.video_path.exists()]

    assert not missing


@pytest.mark.parametrize("case", PSE_VIDEO_CASES, ids=lambda case: case.test_id)
def test_video_analysis_matches_pse_test_media_expectations(case) -> None:
    fps, frames = read_video_frames(case.video_path)
    mismatches: list[str] = []

    for expectation in case.expectations:
        result = analyze_frames(
            frames,
            fps=fps,
            path=case.video_path,
            profile=profile_for_standard(expectation.standard),
        )
        if result.passes is expectation.expected_pass:
            continue

        mismatches.append(
            f"{expectation.standard}: expected {expectation.outcome!r} "
            f"(passes={expectation.expected_pass}), got passes={result.passes}, "
            f"events={len(result.events)}, failures={list(result.failures)}"
        )

    assert not mismatches, "\n".join(mismatches)
