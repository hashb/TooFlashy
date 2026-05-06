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
        return self.outcome == "pass"


@dataclass(frozen=True)
class PseVideoCase:
    json_path: Path
    video_path: Path
    expectations: tuple[PseExpectation, ...]

    @property
    def test_id(self) -> str:
        return f"{self.json_path.parent.name}/{self.json_path.stem}"


def collect_pse_video_cases(repo: Path) -> list[PseVideoCase]:
    cases: list[PseVideoCase] = []
    for json_path in sorted((repo / "video_creation").glob("*/*.json")):
        data = json.loads(json_path.read_text())
        expected = data.get("expected_result", {})
        expectations = tuple(
            PseExpectation(standard=standard, outcome=str(outcome).lower())
            for standard, outcome in sorted(expected.items())
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
