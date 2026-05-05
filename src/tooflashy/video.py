from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


def _parse_rate(rate: str | None) -> float:
    if not rate or rate == "0/0":
        return 0.0
    return float(Fraction(rate))


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


def _read_ffmpeg(path: str | Path, *, max_frames: int | None = None) -> tuple[float, list[np.ndarray]]:
    stream = ffprobe_stream(path)
    width = int(stream["width"])
    height = int(stream["height"])
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")) or 30.0

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe was not created")

    frame_bytes = width * height * 3
    frames: list[np.ndarray] = []
    try:
        while max_frames is None or len(frames) < max_frames:
            raw = proc.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise ValueError(f"truncated frame in {path}")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            frames.append(frame)
    finally:
        if proc.stdout:
            proc.stdout.close()
        return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return fps, frames


def _read_opencv(path: str | Path, *, max_frames: int | None = None) -> tuple[float, list[np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: list[np.ndarray] = []
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return fps, frames


def read_video_frames(
    path: str | Path,
    *,
    engine: str = "ffmpeg",
    max_frames: int | None = None,
) -> tuple[float, list[np.ndarray]]:
    if engine == "ffmpeg":
        return _read_ffmpeg(path, max_frames=max_frames)
    if engine == "opencv":
        return _read_opencv(path, max_frames=max_frames)
    raise ValueError(f"unknown video reader engine {engine!r}")
