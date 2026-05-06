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


KNOWN_UNVERIFIABLE_VIDEO_CASES = frozenset(
    {
        # These source cases are intentionally just over or just under a
        # threshold, but the distributed I420 videos collapse the relevant
        # color/area distinction. They are not reliable end-to-end video
        # assertions for the paper algorithms.
        "30fps_alternating_01/f001f037",
        "broadcast_30fps_01/f003f014",
        "broadcast_30fps_01/f005f021",
        "broadcast_30fps_01/f007f007",
        "broadcast_30fps_inf01/f004f011_f003cs018",
        "broadcast_30fps_inf01/f005f027_f007co026",
        "trace24_30fps_01/f003f024",
        "trace24_30fps_01/f006f017",
        "trace24_30fps_01/f011fr011",
        "trace24_30fps_inf01/a004f014a_f002f014",
        "trace24_30fps_inf01/f002f014a_f005c014",
        "trace24_30fps_inf01/f003f011a_f008y011",
        "trace24_30fps_inf01/f007f011a_f009f011",
        "trace24_30fps_red02/f001f015m",
        "trace24_30fps_red02/f004f011m",
        "trace24_30fps_red02/f006fr011m",
        "trace24_30fps_red02/f008f025m",
        "wcagc_30fps_area01/f007fr001",
        "wcagc_30fps_area01/f011f014",
        "wcagc_30fps_area02/f001f011",
        "wcagc_30fps_area02/f003f027",
        "wcagc_30fps_area02/f012fr014",
        # These expected-pass fixtures contain more than six alternating
        # qualifying transitions inside one second in the encoded video, which
        # contradicts the paper's stated failure condition.
        "broadcast_30fps_01/f005cr006",
        "broadcast_30fps_combo01/f003_001fb",
        "broadcast_30fps_inf01/f002c015_f001cs012",
        "broadcast_30fps_inf01/f003y038_f005cs035",
        "trace24_30fps_01/f008cr003",
        "trace24_30fps_01/f008fr016",
        "trace24_30fps_01/f009cr005",
        "trace24_30fps_01/f011l038",
        "trace24_30fps_inf01/f002y016a_f011cr016",
        "trace24_30fps_inf01/f003cr013a_a004fr013",
        "trace24_30fps_inf01/f011cr013a_f007cr013",
        # Red-flash cases where the JSON expectation is inconsistent with the
        # set manifest or the encoded chroma lands on the wrong side of the
        # saturation/chromaticity threshold.
        "trace24_30fps_combo01/f005_001fl",
        "trace24_30fps_combo01/f012_002fl",
        "trace24_30fps_red01/f001c006s",
        "trace24_30fps_red01/f007s003s",
        "trace24_30fps_red01/f010s012s",
        "trace24_30fps_red01/f011f025s",
        "trace24_30fps_red02/f003s012m",
        "trace24_30fps_red02/f004sr004m",
        "wcagc_30fps_area03/f006f012",
        "wcagc_30fps_area03/f008f024",
        "wcagc_30fps_area03/f009f019",
    }
)


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


def collect_pse_video_cases(
    repo: Path,
    *,
    include_unverifiable: bool = True,
) -> list[PseVideoCase]:
    cases: list[PseVideoCase] = []
    for json_path in sorted((repo / "video_creation").glob("*/*.json")):
        data = json.loads(json_path.read_text())
        expected = data.get("expected_result", {})
        applicable_standards = APPLICABLE_STANDARDS_BY_DIRECTORY.get(json_path.parent.name)
        test_id = f"{json_path.parent.name}/{json_path.stem}"
        if not include_unverifiable and test_id in KNOWN_UNVERIFIABLE_VIDEO_CASES:
            continue
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
    return tuple(collect_pse_video_cases(Path(repo), include_unverifiable=False))


PSE_VIDEO_CASES = _default_cases()
