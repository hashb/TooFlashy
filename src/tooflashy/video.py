from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


def _parse_rate(rate: str | None) -> float:
    if not rate or rate == "0/0":
        return 0.0
    return float(Fraction(rate))


def _ffmpeg_command(
    path: str | Path,
    *,
    max_frames: int | None = None,
) -> list[str]:
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.extend(
        [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    return command


def ffprobe_stream(path: str | Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"no video stream found in {path}")
    return streams[0]


def _iter_ffmpeg_frames(
    path: str | Path,
    *,
    width: int,
    height: int,
    max_frames: int | None = None,
) -> Iterator[np.ndarray]:
    if max_frames is not None and max_frames <= 0:
        return

    command = _ffmpeg_command(path, max_frames=max_frames)
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe was not created")

    frame_bytes = width * height * 3
    completed = False
    try:
        frames_read = 0
        while max_frames is None or frames_read < max_frames:
            raw = proc.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise ValueError(f"truncated frame in {path}")
            frames_read += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            yield frame
        completed = True
    finally:
        proc.stdout.close()
        if not completed and proc.poll() is None:
            proc.terminate()
        return_code = proc.wait()
    if completed and return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _iter_opencv_frames(
    path: str | Path,
    *,
    max_frames: int | None = None,
) -> Iterator[np.ndarray]:
    if max_frames is not None and max_frames <= 0:
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video {path}")
    try:
        frames_read = 0
        while max_frames is None or frames_read < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames_read += 1
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def iter_video_frames(
    path: str | Path,
    *,
    engine: str = "ffmpeg",
    max_frames: int | None = None,
) -> tuple[float, Iterator[np.ndarray]]:
    if engine == "ffmpeg":
        stream = ffprobe_stream(path)
        width = int(stream["width"])
        height = int(stream["height"])
        fps = (
            _parse_rate(stream.get("avg_frame_rate"))
            or _parse_rate(stream.get("r_frame_rate"))
            or 30.0
        )
        frames = _iter_ffmpeg_frames(
            path,
            width=width,
            height=height,
            max_frames=max_frames,
        )
        return fps, frames

    if engine == "opencv":
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"could not open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()
        return fps, _iter_opencv_frames(path, max_frames=max_frames)

    raise ValueError(f"unknown video reader engine {engine!r}")


def read_video_frames(
    path: str | Path,
    *,
    engine: str = "ffmpeg",
    max_frames: int | None = None,
) -> tuple[float, list[np.ndarray]]:
    fps, frames = iter_video_frames(path, engine=engine, max_frames=max_frames)
    return fps, list(frames)
