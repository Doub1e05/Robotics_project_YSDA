#!/usr/bin/env python3
"""Aggregate a complete standard LIBERO Spatial decoder-medoid run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--candidate-seeds", required=True)
    args = parser.parse_args()
    tasks = [json.loads(path.read_text()) for path in sorted((args.out_root / "results").glob("*.json"))]
    episodes = sum(task["episodes"] for task in tasks)
    if len(tasks) != 10 or episodes != 100:
        raise RuntimeError(f"Expected 10 tasks / 100 episodes, got {len(tasks)} / {episodes}.")
    successes = sum(task["successes"] for task in tasks)
    seeds = [int(value) for value in args.candidate_seeds.split(",")]
    summary = {
        "model": "GR00T-N1.7-LIBERO/libero_spatial",
        "benchmark": "LIBERO Spatial",
        "planner": "decoder_action_token_medoid",
        "candidate_seeds": seeds,
        "distance": "time-aligned cosine distance over final DiT action tokens; first four weighted x4",
        "tasks": tasks,
        "completed_rollouts": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
