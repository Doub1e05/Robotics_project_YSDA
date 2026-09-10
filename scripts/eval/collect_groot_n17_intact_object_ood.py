#!/usr/bin/env python3
"""Collect JSON results for the official eight-task INT-ACT Object OOD subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = {
    "OOD Relation": [
        "widowx_cube_on_plate_clean",
        "widowx_small_plate_on_green_cube_clean",
        "widowx_carrot_on_sponge_clean",
        "widowx_eggplant_on_sponge_clean",
    ],
    "OOD Source": [
        "widowx_coke_can_on_plate_clean",
        "widowx_pepsi_on_plate_clean",
    ],
    "OOD Target": ["widowx_carrot_on_keyboard_clean"],
    "OOD Source+Target": ["widowx_coke_can_on_keyboard_clean"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--action-seed", type=int, required=True)
    args = parser.parse_args()

    task_stats = {}
    categories = {}
    for category, task_keys in TASKS.items():
        category_successes = 0
        category_episodes = 0
        for task_key in task_keys:
            path = args.out_root / "results" / f"{task_key}.json"
            if path.exists():
                item = json.loads(path.read_text(encoding="utf-8"))
                successes = int(item["successes"])
                episodes = int(item["episodes"])
            else:
                successes = episodes = 0
            task_stats[task_key] = {
                "category": category,
                "successes": successes,
                "episodes": episodes,
                "success_rate": successes / episodes if episodes else None,
            }
            category_successes += successes
            category_episodes += episodes
        categories[category] = {
            "successes": category_successes,
            "episodes": category_episodes,
            "success_rate": category_successes / category_episodes if category_episodes else None,
        }

    successes = sum(item["successes"] for item in task_stats.values())
    episodes = sum(item["episodes"] for item in task_stats.values())
    summary = {
        "model": "nvidia/GR00T-N1.7-SimplerEnv-Bridge",
        "protocol": "INT-ACT Object OOD official 8-task subset; 24 episodes/task",
        "action_seed": args.action_seed,
        "environment_seed_start": 0,
        "execution_horizon": 4,
        "num_tasks": 8,
        "episodes_per_task": 24,
        "categories": categories,
        "tasks": task_stats,
        "overall": {
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes if episodes else None,
        },
    }
    (args.out_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_root / "summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write("category\tsuccesses\tepisodes\tsuccess_rate\n")
        for category, item in categories.items():
            handle.write(
                f"{category}\t{item['successes']}\t{item['episodes']}\t{item['success_rate']}\n"
            )
        handle.write(f"OVERALL\t{successes}\t{episodes}\t{summary['overall']['success_rate']}\n")
    print(json.dumps(summary["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
