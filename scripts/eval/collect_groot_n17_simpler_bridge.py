#!/usr/bin/env python3
"""Combine the four fixed GR00T SIMPLER-Bridge task result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--candidate-seeds", default="1,999,998")
    args = parser.parse_args()
    results = []
    for path in sorted((args.out_root / "results").glob("*.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    if len(results) != 4 or sum(result["episodes"] for result in results) != 96:
        raise RuntimeError(f"Expected four 24-episode task files, got {len(results)} files: {results}")
    successes = sum(result["successes"] for result in results)
    episodes = sum(result["episodes"] for result in results)
    summary = {
        "model": "nvidia/GR00T-N1.7-SimplerEnv-Bridge",
        "planner": "consensus-only action-space medoid",
        "candidate_seeds": [int(value) for value in args.candidate_seeds.split(",")],
        "environment_seed_start": 0,
        "execution_horizon": 4,
        "tasks": results,
        "completed_rollouts": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
