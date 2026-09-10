#!/usr/bin/env python3
"""Write the aggregate SIMPLER-Bridge summary for GR00T decoder token medoid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    tasks = [json.loads(path.read_text()) for path in sorted((args.out_root / "results").glob("*.json"))]
    episodes = sum(task["episodes"] for task in tasks)
    if len(tasks) != 4 or episodes != 96:
        raise RuntimeError(f"Expected 4 completed task files / 96 rollouts, got {len(tasks)} / {episodes}.")
    successes = sum(task["successes"] for task in tasks)
    summary = {
        "model": "nvidia/GR00T-N1.7-SimplerEnv-Bridge",
        "planner": "decoder_action_token_medoid",
        "candidate_seeds": [1, 999, 998],
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
