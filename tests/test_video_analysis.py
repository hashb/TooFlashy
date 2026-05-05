import csv
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tooflashy.analysis import analyze_video


MEDIA_REPO = "https://github.com/traceRERC/pse-test-media"
MEDIA_REF = "edf799a15cc1a8817a58c0120a7b25b2b28a1932"


@pytest.fixture(scope="session")
def pse_media_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    existing = os.environ.get("PSE_TEST_MEDIA")
    if existing:
        return Path(existing)

    root = tmp_path_factory.mktemp("pse-test-media")
    subprocess.run(
        ["git", "clone", "--quiet", MEDIA_REPO, str(root)],
        check=True,
    )
    subprocess.run(["git", "checkout", "--quiet", MEDIA_REF], cwd=root, check=True)
    return root


def _generate_video(repo: Path, json_file: Path) -> Path:
    subprocess.run(
        ["python", str(repo / "make_single_video.py"), str(json_file)],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    video = json_file.parent / "videos" / f"{json_file.stem}.avi"
    assert video.exists()
    return video


def _expected_results(csv_file: Path) -> dict[str, bool]:
    with csv_file.open(newline="") as fh:
        rows = csv.DictReader(fh)
        return {row["filename"]: row["expected"].strip().lower() == "pass" for row in rows}


@pytest.mark.parametrize(
    ("suite", "names"),
    [
        ("wcagc_30fps_area01", ["f010f005", "a010f005"]),
        ("trace24_30fps_red01", ["f004f005s", "a003f001s"]),
    ],
)
def test_video_analysis_matches_selected_pse_test_media(
    pse_media_repo: Path, tmp_path: Path, suite: str, names: list[str]
) -> None:
    source_suite = pse_media_repo / "video_creation" / suite
    work_suite = tmp_path / suite
    shutil.copytree(source_suite, work_suite)

    expected = _expected_results(work_suite / f"{suite}.csv")
    for name in names:
        video = _generate_video(pse_media_repo, work_suite / f"{name}.json")
        result = analyze_video(video)
        assert result.passes is expected[name], result
