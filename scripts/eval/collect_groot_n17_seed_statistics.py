#!/usr/bin/env python3
"""Aggregate fixed-action-seed GR00T Bridge summaries into mean and std."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, stdev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for path in sorted(args.out_root.glob("seed_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        candidate_seeds = summary.get("candidate_seeds", [])
        if len(candidate_seeds) != 1:
            raise RuntimeError(f"Expected exactly one action seed in {path}.")
        runs.append(
            {
                "action_seed": candidate_seeds[0],
                "successes": summary["successes"],
                "completed_rollouts": summary["completed_rollouts"],
                "success_rate": summary["success_rate"],
                "summary_path": str(path.relative_to(args.out_root)),
            }
        )
    if len(runs) < 2 or any(run["completed_rollouts"] != 96 for run in runs):
        raise RuntimeError("Expected at least two complete 96-rollout seed summaries.")
    values = [run["success_rate"] for run in runs]
    aggregate = {
        "model": "nvidia/GR00T-N1.7-SimplerEnv-Bridge",
        "planner": "baseline_fixed_action_seed",
        "runs": runs,
        "mean_success_rate": fmean(values),
        "sample_std_success_rate": stdev(values),
        "num_seeds": len(values),
    }
    (args.out_root / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
