#!/usr/bin/env python3
"""Aggregate a GR00T LIBERO-Plus Spatial fixed-slice arm."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import tyro


@dataclass
class Config:
    arm_root: Path
    method: str
    action_seeds: str
    expected_episodes: int = 100


def main(config: Config) -> None:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((config.arm_root / "results").glob("*.json"))
    ]
    episodes = sum(int(row["episodes"]) for row in rows)
    successes = sum(int(row["successes"]) for row in rows)
    categories = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        cat_episodes = sum(int(row["episodes"]) for row in selected)
        cat_successes = sum(int(row["successes"]) for row in selected)
        categories[category] = {
            "episodes": cat_episodes,
            "successes": cat_successes,
            "success_rate": cat_successes / cat_episodes if cat_episodes else None,
        }
    summary = {
        "benchmark": "LIBERO-Plus Spatial fixed first-100 slice",
        "model": "GR00T-N1.7-LIBERO/libero_spatial",
        "method": config.method,
        "action_seeds": [int(value) for value in config.action_seeds.split(",")],
        "task_indices": [0, config.expected_episodes - 1],
        "expected_episodes": config.expected_episodes,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else None,
        "by_category": categories,
    }
    (config.arm_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
