#!/usr/bin/env python3
"""Collect SR for the eight-task INT-ACT Object OOD subset."""
from __future__ import annotations
import argparse, json
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
    "OOD Target": [
        ("widowx_carrot_on_keyboard_clean", "PutCarrotOnKeyboardInScene-v2"),
    ],
    "OOD Source+Target": [
        ("widowx_coke_can_on_keyboard_clean", "PutCokeCanOnKeyboardInScene-v2"),
    ],
}


def stats(seed_root: Path, env: str) -> dict[str, float | int | None]:
    files = list((seed_root / "result").glob(f"{env}_intact_object_ood_consensus_medoid_seed0/**/*.mp4"))
    successes = sum(path.name.startswith("success_") for path in files)
    episodes = len(files)
    return {"successes": successes, "episodes": episodes, "sr": successes / episodes if episodes else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--candidate-seeds", default="2,996,997")
    args = parser.parse_args()
    root = args.out_root
    seed_root = root / "seed0"
    task_stats = {}
    for category, entries in TASKS.items():
        for task, env in entries:
            task_stats[task] = {**stats(seed_root, env), "category": category, "environment": env}
    categories = {}
    for category, entries in TASKS.items():
        selected = [task_stats[task] for task, _ in entries]
        successes = sum(int(item["successes"]) for item in selected)
        episodes = sum(int(item["episodes"]) for item in selected)
        categories[category] = {"successes": successes, "episodes": episodes, "sr": successes / episodes if episodes else None}
    successes = sum(int(item["successes"]) for item in task_stats.values())
    episodes = sum(int(item["episodes"]) for item in task_stats.values())
    summary = {
        "protocol": "MIMIC-Video bridge checkpoint; consensus-medoid-only; 8-task INT-ACT Object OOD subset; 24 episodes/task; candidate seeds are fixed per chunk",
        "candidate_seeds": [int(x) for x in args.candidate_seeds.split(",")],
        "num_tasks": 8,
        "episodes_per_task": 24,
        "seeds": {"0": {"candidate_seeds": [int(x) for x in args.candidate_seeds.split(",")], "categories": categories, "tasks": task_stats, "overall": {"successes": successes, "episodes": episodes, "sr": successes / episodes if episodes else None}}},
        "overall": {"successes": successes, "episodes": episodes, "sr": successes / episodes if episodes else None},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (root / "summary.tsv").open("w") as handle:
        handle.write("seed\tcategory\tsuccesses\tepisodes\tsr\n")
        for category, item in categories.items():
            handle.write(f"0\t{category}\t{item['successes']}\t{item['episodes']}\t{item['sr']}\n")
        handle.write(f"0\tOVERALL\t{successes}\t{episodes}\t{summary['overall']['sr']}\n")
    for category, item in categories.items():
        print(f"{category}: {item['successes']}/{item['episodes']}={item['sr']:.4f}" if item["sr"] is not None else f"{category}: 0/0=NA")
    print(f"ALL: {successes}/{episodes}={summary['overall']['sr']:.4f}" if episodes else "ALL: 0/0=NA")


if __name__ == "__main__":
    main()
