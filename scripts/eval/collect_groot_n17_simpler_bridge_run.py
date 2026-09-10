#!/usr/bin/env python3
"""Write a self-describing aggregate summary for one GR00T Bridge run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--planner", required=True)
    parser.add_argument("--candidate-seeds", default=None)
    args = parser.parse_args()

    tasks = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((args.out_root / "results").glob("*.json"))]
    completed_rollouts = sum(task["episodes"] for task in tasks)
    if len(tasks) != 4 or completed_rollouts != 96:
        raise RuntimeError(f"Expected four completed task files / 96 rollouts, got {len(tasks)} / {completed_rollouts}.")
    successes = sum(task["successes"] for task in tasks)
    summary = {
        "model": "nvidia/GR00T-N1.7-SimplerEnv-Bridge",
        "planner": args.planner,
        "environment_seed_start": 0,
        "execution_horizon": 4,
        "tasks": tasks,
        "completed_rollouts": completed_rollouts,
        "successes": successes,
        "success_rate": successes / completed_rollouts,
    }
    if args.candidate_seeds:
        summary["candidate_seeds"] = [int(value) for value in args.candidate_seeds.split(",")]
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
