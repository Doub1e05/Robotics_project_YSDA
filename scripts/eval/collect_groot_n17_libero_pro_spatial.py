#!/usr/bin/env python3
"""Aggregate completed LIBERO-PRO Spatial task results."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import tyro


@dataclass
class Config:
    out_root: Path
    action_seed: int = 1


def main(config: Config) -> None:
    rows = []
    for path in sorted((config.out_root / "results").glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    episodes = sum(int(row["episodes"]) for row in rows)
    successes = sum(int(row["successes"]) for row in rows)
    by_split = {}
    for split in ("lan", "object", "swap", "task"):
        selected = [row for row in rows if row["split"] == split]
        split_episodes = sum(int(row["episodes"]) for row in selected)
        split_successes = sum(int(row["successes"]) for row in selected)
        by_split[split] = {
            "tasks_complete": len(selected),
            "episodes": split_episodes,
            "successes": split_successes,
            "success_rate": split_successes / split_episodes if split_episodes else None,
        }
    summary = {
        "benchmark": "LIBERO-PRO Spatial",
        "model": "GR00T-N1.7-LIBERO/libero_spatial",
        "method": "baseline",
        "action_seed": config.action_seed,
        "expected_tasks": 40,
        "expected_episodes": 400,
        "tasks_complete": len(rows),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else None,
        "by_split": by_split,
    }
    (config.out_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
