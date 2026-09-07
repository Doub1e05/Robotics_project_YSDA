#!/usr/bin/env python3
"""Collect INT-ACT Object Diversity SR by seed and official subcategory."""
from __future__ import annotations
import argparse, json
from pathlib import Path

CATEGORIES = {
    "OOD Source": ["widowx_coke_can_on_plate_clean", "widowx_pepsi_on_plate_clean", "widowx_orange_juice_on_plate_clean", "widowx_nut_on_plate_clean"],
    "OOD Target": ["widowx_carrot_on_keyboard_clean", "widowx_eggplant_on_keyboard_clean", "widowx_carrot_on_ramekin_clean", "widowx_carrot_on_wheel_clean"],
    "OOD Source+Target": ["widowx_coke_can_on_keyboard_clean", "widowx_coke_can_on_ramekin_clean", "widowx_coke_can_on_wheel_clean", "widowx_nut_on_wheel_clean"],
    "OOD Relation": ["widowx_cube_on_plate_clean", "widowx_small_plate_on_green_cube_clean", "widowx_carrot_on_sponge_clean", "widowx_eggplant_on_sponge_clean"],
}
ALL = [task for tasks in CATEGORIES.values() for task in tasks]
ENV = {
    "widowx_cube_on_plate_clean":"PutGreenCubeOnPlateInScene-v2", "widowx_small_plate_on_green_cube_clean":"PutSmallPlateOnGreenCubeInScene-v2", "widowx_carrot_on_sponge_clean":"PutCarrotOnSpongeLargerInScene-v2", "widowx_eggplant_on_sponge_clean":"PutEggplantOnSpongeLargerInScene-v2",
    "widowx_coke_can_on_plate_clean":"PutCokeCanOnPlateInScene-v2", "widowx_pepsi_on_plate_clean":"PutPepsiCanOnPlateInScene-v2", "widowx_orange_juice_on_plate_clean":"PutOrangeJuiceOnPlateInScene-v2", "widowx_nut_on_plate_clean":"PutNutOnPlateInScene-v2",
    "widowx_carrot_on_keyboard_clean":"PutCarrotOnKeyboardInScene-v2", "widowx_eggplant_on_keyboard_clean":"PutEggplantOnKeyboardInScene-v2", "widowx_carrot_on_ramekin_clean":"PutCarrotOnRamekinInScene-v2", "widowx_carrot_on_wheel_clean":"PutCarrotOnWheelInScene-v2",
    "widowx_coke_can_on_keyboard_clean":"PutCokeCanOnKeyboardInScene-v2", "widowx_coke_can_on_ramekin_clean":"PutCokeCanOnRamekinInScene-v2", "widowx_coke_can_on_wheel_clean":"PutCokeCanOnWheelInScene-v2", "widowx_nut_on_wheel_clean":"PutNutOnWheelInScene-v2",
}
def task_stats(root: Path, task: str) -> dict:
    files = list((root / "result").glob(f"{ENV[task]}_intact_object_ood*/**/*.mp4"))
    return {"successes": sum(p.name.startswith("success_") for p in files), "episodes": len(files), "sr": (sum(p.name.startswith("success_") for p in files) / len(files) if files else None)}
def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out-root", type=Path, required=True); args = ap.parse_args()
    summary = {"protocol": "MIMIC-Video Bridge checkpoint; stop_video_denoising_step=0; INT-ACT Object Diversity; 16 tasks x 24 episodes x 3 seeds", "seeds": {}}
    for seed_dir in sorted(args.out_root.glob("seed*")):
        if not seed_dir.is_dir(): continue
        seed = seed_dir.name.removeprefix("seed")
        tasks = {task: task_stats(seed_dir, task) for task in ALL}
        cats = {}
        for category, names in CATEGORIES.items():
            ok = sum(tasks[t]["successes"] for t in names); n = sum(tasks[t]["episodes"] for t in names)
            cats[category] = {"successes": ok, "episodes": n, "sr": ok / n if n else None}
        ok = sum(x["successes"] for x in tasks.values()); n = sum(x["episodes"] for x in tasks.values())
        summary["seeds"][seed] = {"categories": cats, "overall": {"successes": ok, "episodes": n, "sr": ok / n if n else None}, "tasks": tasks}
    all_ok = sum(v["overall"]["successes"] for v in summary["seeds"].values()); all_n = sum(v["overall"]["episodes"] for v in summary["seeds"].values())
    summary["overall"] = {"successes": all_ok, "episodes": all_n, "sr": all_ok / all_n if all_n else None}
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out_root / "summary.tsv").open("w") as f:
        f.write("seed\tcategory\tsuccesses\tepisodes\tsr\n")
        for seed, data in summary["seeds"].items():
            for category, x in data["categories"].items(): f.write(f"{seed}\t{category}\t{x['successes']}\t{x['episodes']}\t{x['sr']}\n")
            x = data["overall"]; f.write(f"{seed}\tOVERALL\t{x['successes']}\t{x['episodes']}\t{x['sr']}\n")
        f.write(f"ALL\tOVERALL\t{all_ok}\t{all_n}\t{summary['overall']['sr']}\n")
    for seed, data in summary["seeds"].items():
        print(f"seed={seed}", *(f"{k}: {v['successes' ]}/{v['episodes' ]}=" + (f"{v['sr' ]:.4f}" if v['sr' ] is not None else "NA") for k,v in data["categories"].items()), f"overall: {data['overall']['successes']}/{data['overall']['episodes']}=" + (f"{data['overall']['sr']:.4f}" if data['overall']['sr'] is not None else "NA"))
    print(f"ALL: {all_ok}/{all_n}={summary['overall']['sr']:.4f}")
if __name__ == "__main__": main()
