#!/usr/bin/env python3
"""Aggregate per-query SIMPLER Bridge diagnostics sidecars into JSONL and CSV."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def flatten(prefix: str, value, row: dict) -> None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        row[prefix] = value
    elif isinstance(value, dict):
        for key, nested in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), nested, row)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    expected = {}
    with args.manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            expected[(row["task"], int(row["episode"]))] = row["expected_status"]
    rows = []
    for path in sorted(args.out_root.glob("result/**/*.diagnostics.json")):
        parts = path.parts
        task = next((part for part in parts if part.endswith("-v0")), "unknown")
        stem = path.name
        marker = "_obj_episode_"
        if marker not in stem:
            continue
        episode = int(stem.split(marker, 1)[1].split("_", 1)[0])
        actual = "success" if stem.startswith("success_") else "failure"
        for record in json.loads(path.read_text(encoding="utf-8")):
            row = {"task": task, "episode": episode, "expected_status": expected.get((task, episode)), "actual_status": actual}
            flatten("", record, row)
            rows.append(row)
    output = args.out_root / "metrics"
    output.mkdir(parents=True, exist_ok=True)
    (output / "query_metrics.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    columns = sorted({key for row in rows for key in row})
    with (output / "query_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    summary = {"expected_episodes": len(expected), "completed_episodes": len({(r["task"], r["episode"]) for r in rows}), "query_records": len(rows)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
