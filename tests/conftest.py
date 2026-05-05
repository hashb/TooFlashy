import os
import subprocess
from pathlib import Path

import pytest


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
