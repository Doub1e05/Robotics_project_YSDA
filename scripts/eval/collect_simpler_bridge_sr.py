#!/usr/bin/env python3
"""Collect the SR from the shared result directory of the fixed 96-rollout Bridge protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    videos = sorted(args.out_root.glob("result/**/*.mp4"))
    successes = [path for path in videos if path.name.startswith("success_")]
    summary = {
        "protocol": "ftcosmos; stop_video_denoising_step=0; 4 tasks x 24 fixed episode variants",
        "expected_rollouts": 96,
        "completed_rollouts": len(videos),
        "successes": len(successes),
        "success_rate": len(successes) / len(videos) if videos else None,
    }
    if len(videos) != 96:
        raise RuntimeError(f"Expected 96 rollout videos, found {len(videos)}")
    path = args.out_root / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
