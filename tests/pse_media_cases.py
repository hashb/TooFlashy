from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PseExpectation:
    standard: str
    outcome: str

    @property
    def expected_pass(self) -> bool:
        return not self.outcome.startswith("fail")


@dataclass(frozen=True)
class PseVideoCase:
    json_path: Path
    video_path: Path
    expectations: tuple[PseExpectation, ...]

    @property
    def test_id(self) -> str:
        return f"{self.json_path.parent.name}/{self.json_path.stem}"


APPLICABLE_STANDARDS_BY_DIRECTORY = {
    "30fps_alternating_01": {
        "iso",
        "itu_r1702_4",
        "ofcom2017",
        "trace24",
        "wcag2_2",
    },
    "broadcast_30fps_01": {"iso", "itu_r1702_4", "ofcom2017"},
    "broadcast_30fps_combo01": {"iso", "itu_r1702_4", "ofcom2017"},
    "broadcast_30fps_inf01": {"iso", "itu_r1702_4", "ofcom2017"},
    "broadcast_30fps_inf02": {"iso", "itu_r1702_4", "ofcom2017"},
    "broadcast_30fps_red01": {"iso", "itu_r1702_4", "ofcom2017"},
    "broadcast_30fps_red02": {"iso", "itu_r1702_4", "ofcom2017"},
    "trace24_30fps_01": {"trace24"},
    "trace24_30fps_combo01": {"trace24"},
    "trace24_30fps_inf01": {"trace24"},
    "trace24_30fps_red01": {"trace24"},
    "trace24_30fps_red02": {"trace24"},
    "wcagc_30fps_area01": {"wcag2_2"},
    "wcagc_30fps_area02": {"wcag2_2"},
    "wcagc_30fps_area03": {"wcag2_2"},
}


def collect_pse_video_cases(repo: Path) -> list[PseVideoCase]:
    cases: list[PseVideoCase] = []
    for json_path in sorted((repo / "video_creation").glob("*/*.json")):
        data = json.loads(json_path.read_text())
        expected = data.get("expected_result", {})
        applicable_standards = APPLICABLE_STANDARDS_BY_DIRECTORY.get(json_path.parent.name)
        expectations = tuple(
            PseExpectation(standard=standard, outcome=str(outcome).lower())
            for standard, outcome in sorted(expected.items())
            if applicable_standards is None or standard in applicable_standards
        )
        cases.append(
            PseVideoCase(
                json_path=json_path,
                video_path=json_path.parent / "videos" / f"{json_path.stem}.avi",
                expectations=expectations,
            )
        )
    return cases


def _default_cases() -> tuple[PseVideoCase, ...]:
    repo = os.environ.get("PSE_TEST_MEDIA")
    if not repo:
        return ()
    return tuple(collect_pse_video_cases(Path(repo)))


PSE_VIDEO_CASES = _default_cases()
