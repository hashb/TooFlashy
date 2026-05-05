from pathlib import Path

import cv2
import numpy as np

from tooflashy.video import read_video_frames


def test_ffmpeg_reader_returns_rgb_frames_and_fps(tmp_path: Path) -> None:
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        2.0,
        (8, 8),
    )
    assert writer.isOpened()
    writer.write(np.full((8, 8, 3), (0, 0, 255), dtype=np.uint8))
    writer.write(np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8))
    writer.release()

    fps, frames = read_video_frames(path, engine="ffmpeg")

    assert fps == 2.0
    assert len(frames) == 2
    assert frames[0].shape == (8, 8, 3)
    assert frames[0][0, 0, 0] > frames[0][0, 0, 2]
    assert frames[1][0, 0, 2] > frames[1][0, 0, 0]
