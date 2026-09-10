#!/usr/bin/env python3
"""Collect SR for one 8-task INT-ACT Object OOD decoder-token-medoid run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = {
    "OOD Relation": [
        ("widowx_cube_on_plate_clean", "PutGreenCubeOnPlateInScene-v2"),
        ("widowx_small_plate_on_green_cube_clean", "PutSmallPlateOnGreenCubeInScene-v2"),
        ("widowx_carrot_on_sponge_clean", "PutCarrotOnSpongeLargerInScene-v2"),
        ("widowx_eggplant_on_sponge_clean", "PutEggplantOnSpongeLargerInScene-v2"),
    ],
    "OOD Source": [
        ("widowx_coke_can_on_plate_clean", "PutCokeCanOnPlateInScene-v2"),
        ("widowx_pepsi_on_plate_clean", "PutPepsiCanOnPlateInScene-v2"),
    ],
    "OOD Target": [("widowx_carrot_on_keyboard_clean", "PutCarrotOnKeyboardInScene-v2")],
    "OOD Source+Target": [("widowx_coke_can_on_keyboard_clean", "PutCokeCanOnKeyboardInScene-v2")],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--candidate-seeds", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    result_root = args.out_root / "result"
    task_stats = {}
    for category, tasks in TASKS.items():
        for task, env in tasks:
            files = list(result_root.glob(f"{env}_{args.tag}/**/*.mp4"))
            task_stats[task] = {
                "category": category,
                "environment": env,
                "successes": sum(path.name.startswith("success_") for path in files),
                "episodes": len(files),
            }
            task_stats[task]["sr"] = (
                task_stats[task]["successes"] / task_stats[task]["episodes"]
                if task_stats[task]["episodes"]
                else None
            )
    categories = {}
    for category, tasks in TASKS.items():
        selected = [task_stats[task] for task, _ in tasks]
        successes = sum(item["successes"] for item in selected)
        episodes = sum(item["episodes"] for item in selected)
        categories[category] = {"successes": successes, "episodes": episodes, "sr": successes / episodes if episodes else None}
    successes = sum(item["successes"] for item in task_stats.values())
    episodes = sum(item["episodes"] for item in task_stats.values())
    if episodes != 192:
        raise RuntimeError(f"Expected 192 completed Object OOD episodes, found {episodes}.")
    summary = {
        "protocol": "MIMIC-Video Bridge; decoder action-token medoid; 8-task INT-ACT Object OOD; 24 episodes/task",
        "candidate_seeds": [int(value) for value in args.candidate_seeds.split(",")],
        "categories": categories,
        "tasks": task_stats,
        "overall": {"successes": successes, "episodes": episodes, "sr": successes / episodes},
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
