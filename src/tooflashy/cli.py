from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze video for PSE flash hazards.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    result = analyze_video(args.video)
    if args.json:
        print(
            json.dumps(
                {
                    "path": str(result.path),
                    "passes": result.passes,
                    "fps": result.fps,
                    "frame_count": result.frame_count,
                    "event_count": len(result.events),
                    "failures": list(result.failures),
                },
                indent=2,
            )
        )
    else:
        status = "PASS" if result.passes else "FAIL"
        print(f"{status} {result.path}")
        for failure in result.failures:
            print(f"- {failure}")
    return 0 if result.passes else 1
