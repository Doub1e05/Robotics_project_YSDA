#!/usr/bin/env python3
"""Copy up to 2 rollout videos per eval run into videos/_git_samples/ for git."""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "eval_outputs"
SAMPLES_DIRNAME = "_git_samples"
MAX_SAMPLES = 2


def pick_videos(video_dir: Path) -> list[Path]:
    mp4s = sorted(video_dir.glob("*.mp4"))
    if not mp4s:
        return []
    success = [p for p in mp4s if "_success_" in p.name]
    failure = [p for p in mp4s if "_failure_" in p.name]
    chosen: list[Path] = []
    if success:
        chosen.append(success[0])
    if failure:
        chosen.append(failure[0])
    for p in mp4s:
        if len(chosen) >= MAX_SAMPLES:
            break
        if p not in chosen:
            chosen.append(p)
    return chosen[:MAX_SAMPLES]


def main() -> None:
    if not EVAL_ROOT.is_dir():
        print(f"[skip] no eval_outputs at {EVAL_ROOT}")
        return
    total = 0
    for video_dir in sorted(EVAL_ROOT.rglob("videos")):
        if video_dir.name != "videos" or SAMPLES_DIRNAME in video_dir.parts:
            continue
        if "rank" in video_dir.parts:
            continue
        mp4s = list(video_dir.glob("*.mp4"))
        if not mp4s:
            continue
        out_dir = video_dir / SAMPLES_DIRNAME
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        for src in pick_videos(video_dir):
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            total += 1
            print(f"[sample] {dst.relative_to(REPO_ROOT)}")
    print(f"[done] copied {total} sample videos")


if __name__ == "__main__":
    main()
