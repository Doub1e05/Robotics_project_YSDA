#!/usr/bin/env python3
"""Run repeated LIBERO-PRO smoke evals and collect flat outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SUCCESS_RE = re.compile(r"Success:\s*(True|False)")
TASK_RE = re.compile(r"Task:\s*(.+)")


@dataclass
class RunRecord:
    run_id: int
    success: bool
    task: str
    source_video: str
    saved_video: str
    source_log: str
    saved_log: str
    exit_code: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--num-runs", type=int, default=30)
    parser.add_argument(
        "--libero-pro-root",
        type=Path,
        default=Path("/home/motovilovil/Robotics/robotics_project/LIBERO-PRO"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/motovilovil/Robotics/robotics_project/mimic-video/eval_outputs/libero_spatial_pro"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial",
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    return parser.parse_args()


def list_logs(log_dir: Path) -> set[Path]:
    return set(log_dir.glob("EVAL-*.txt"))


def list_videos(rollout_dir: Path) -> set[Path]:
    return set(rollout_dir.glob("*/*.mp4"))


def require_single_new(after: set[Path], before: set[Path], kind: str) -> Path:
    created = sorted(after - before)
    if len(created) != 1:
        raise RuntimeError(f"Expected exactly one new {kind}, found {len(created)}")
    return created[0]


def parse_run_log(log_path: Path) -> tuple[bool, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    success_match = SUCCESS_RE.search(text)
    task_match = TASK_RE.search(text)
    if success_match is None:
        raise RuntimeError(f"Could not parse success from {log_path}")
    success = success_match.group(1) == "True"
    task = task_match.group(1).strip() if task_match else ""
    return success, task


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[RunRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")


def write_csv(path: Path, records: list[RunRecord]) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else list(RunRecord.__annotations__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def main() -> int:
    args = parse_args()
    libero_root = args.libero_pro_root.resolve()
    output_root = args.output_root.resolve()
    log_dir = libero_root / "experiments" / "logs"
    rollout_dir = libero_root / "rollouts"
    videos_dir = output_root / "videos"
    saved_logs_dir = output_root / "logs"
    ensure_dirs([videos_dir, saved_logs_dir])

    run_script = libero_root / "scripts" / "run_eval.sh"
    if not run_script.exists():
        raise FileNotFoundError(f"Missing LIBERO-PRO eval helper: {run_script}")

    records: list[RunRecord] = []
    env = os.environ.copy()
    env["LIBERO_PRO_NUM_TRIALS"] = str(args.num_trials_per_task)

    for run_id in range(1, args.num_runs + 1):
        before_logs = list_logs(log_dir)
        before_videos = list_videos(rollout_dir)
        cmd = [
            "bash",
            str(run_script),
            str(args.gpu),
            args.checkpoint,
            args.suite,
            str(args.max_tasks),
        ]
        print(f"[run {run_id:03d}/{args.num_runs:03d}] starting", flush=True)
        completed = subprocess.run(cmd, cwd=libero_root, env=env, check=False)
        after_logs = list_logs(log_dir)
        after_videos = list_videos(rollout_dir)
        new_log = require_single_new(after_logs, before_logs, "log")
        new_video = require_single_new(after_videos, before_videos, "video")
        success, task = parse_run_log(new_log)

        status = "success" if success else "failure"
        saved_video = videos_dir / f"run_{run_id:03d}_{status}.mp4"
        saved_log = saved_logs_dir / f"run_{run_id:03d}.log"
        shutil.copy2(new_video, saved_video)
        shutil.copy2(new_log, saved_log)

        record = RunRecord(
            run_id=run_id,
            success=success,
            task=task,
            source_video=str(new_video),
            saved_video=str(saved_video),
            source_log=str(new_log),
            saved_log=str(saved_log),
            exit_code=completed.returncode,
        )
        records.append(record)
        print(
            f"[run {run_id:03d}/{args.num_runs:03d}] done success={success} exit_code={completed.returncode}",
            flush=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"LIBERO-PRO eval exited with code {completed.returncode} on run {run_id}")

    successes = sum(1 for record in records if record.success)
    summary = {
        "suite": args.suite,
        "checkpoint": args.checkpoint,
        "gpu": args.gpu,
        "num_runs": args.num_runs,
        "num_trials_per_task": args.num_trials_per_task,
        "max_tasks": args.max_tasks,
        "successes": successes,
        "failures": args.num_runs - successes,
        "success_rate": successes / args.num_runs if args.num_runs else 0.0,
        "videos_dir": str(videos_dir),
        "logs_dir": str(saved_logs_dir),
    }

    write_jsonl(output_root / "runs.jsonl", records)
    write_csv(output_root / "runs.csv", records)
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
