#!/usr/bin/env python3
"""Build fixed 70/30 episode split and train chunk-level CatBoost on train only."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_STATE = 42
REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO_ROOT / "eval_outputs/libero_spatial/baseline/trace_100ep/metrics"
OUTPUT_DIR = METRICS_DIR / "chunk_regeneration_classifiers_v2_70_30"

FEATURES = [
    "chunk_action_delta_norm_mean",
    "flow_prediction_error",
    "action_chunk_temporal_consistency",
    "plan_drift",
    "action_entropy",
    "video_action_mutual_information",
    "mlp_activation_mean",
    "task_semantic_alignment",
    "video_latent_variance_mean",
    "representation_drift_rate",
    "decoder_hidden_norm_mean",
    "object_interaction_confidence",
    "video_latent_norm_std",
    "sampling_stability",
    "chunk_action_variance",
    "video_latent_entropy",
    "decoder_hidden_norm_std",
    "mlp_sparsity",
    "video_action_alignment_score",
    "video_action_cosine_similarity",
    "goal_latent_distance",
    "latent_success_alignment",
    "residual_stream_norm",
    "chunk_action_delta_norm_max",
    "attention_sparsity",
    "video_latent_delta_l2_std",
    "action_chunk_smoothness",
    "action_uncertainty",
    "instruction_attention_score",
    "chunk_gripper_switches",
    "latent_oscillation_score",
    "video_latent_cosine_initial_final",
    "video_latent_norm_mean",
    "score_norm_mean",
    "latent_path_efficiency",
    "query_latency_sec",
]

# Val failures: spread across tasks that have failures (10 total).
VAL_FAILURE_COUNTS = {0: 2, 1: 1, 3: 1, 4: 2, 5: 1, 7: 1, 9: 2}
VAL_SUCCESS_PER_TASK = 2  # 10 tasks x 2 = 20 success val episodes


def load_episodes(traces_path: Path) -> list[dict]:
    episodes = []
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)["meta"]
        episodes.append(
            {
                "task_id": int(meta["task_id"]),
                "episode_idx": int(meta["episode_idx"]),
                "total_episode_idx": int(meta["total_episode_idx"]),
                "success": bool(meta["success"]),
                "task_description": meta.get("task_description", ""),
            }
        )
    return episodes


def pick_random(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n > len(pool):
        raise ValueError(f"Requested {n} episodes but only {len(pool)} available.")
    return rng.sample(pool, n)


def build_split(episodes: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    by_task: dict[int, dict[str, list[dict]]] = defaultdict(lambda: {"success": [], "failure": []})
    for ep in episodes:
        key = "success" if ep["success"] else "failure"
        by_task[ep["task_id"]][key].append(ep)

    val: list[dict] = []
    for task_id, count in VAL_FAILURE_COUNTS.items():
        picked = pick_random(by_task[task_id]["failure"], count, rng)
        for ep in picked:
            ep = dict(ep)
            ep["val_reason"] = "failure_quota"
            val.append(ep)

    for task_id in range(10):
        picked = pick_random(by_task[task_id]["success"], VAL_SUCCESS_PER_TASK, rng)
        for ep in picked:
            ep = dict(ep)
            ep["val_reason"] = "success_quota"
            val.append(ep)

    val_keys = {(e["task_id"], e["episode_idx"]) for e in val}
    train = [ep for ep in episodes if (ep["task_id"], ep["episode_idx"]) not in val_keys]

    val_fail = sum(1 for e in val if not e["success"])
    val_succ = sum(1 for e in val if e["success"])
    train_fail = sum(1 for e in train if not e["success"])
    train_succ = sum(1 for e in train if e["success"])

    if len(val) != 30 or val_fail != 10 or val_succ != 20:
        raise RuntimeError(f"Bad val split: n={len(val)} fail={val_fail} succ={val_succ}")
    if len(train) != 70:
        raise RuntimeError(f"Bad train split: n={len(train)}")
    if train_fail != 21 or train_succ != 49:
        raise RuntimeError(f"Bad train composition: fail={train_fail} succ={train_succ}")

    return {"train": train, "val": val, "meta": {
        "random_state": RANDOM_STATE,
        "val_failure_counts_by_task": VAL_FAILURE_COUNTS,
        "val_success_per_task": VAL_SUCCESS_PER_TASK,
        "train_episodes": len(train),
        "val_episodes": len(val),
        "train_failures": train_fail,
        "train_successes": train_succ,
        "val_failures": val_fail,
        "val_successes": val_succ,
    }}


def load_chunk_df(metrics_dir: Path) -> pd.DataFrame:
    csv_path = metrics_dir / "chunk_metrics.csv"
    df = pd.read_csv(csv_path)
    df["success"] = df["success"].map(lambda v: v is True or str(v).strip().lower() in {"true", "1", "yes"})
    df["needs_regeneration"] = (~df["success"]).astype(int)
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    # chunk_id is intentionally excluded from FEATURES.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def episode_key_series(df: pd.DataFrame) -> pd.Series:
    return df["task_id"].astype(str) + ":" + df["episode_idx"].astype(str)


def evaluate(y_true: pd.Series, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_failure": float(precision_score(y_true, pred, zero_division=0)),
        "recall_failure": float(recall_score(y_true, pred, zero_division=0)),
        "f1_failure": float(f1_score(y_true, pred, zero_division=0)),
        "regen_rate": float(pred.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    rng = random.Random(RANDOM_STATE)
    traces_path = METRICS_DIR / "episode_traces.jsonl"
    episodes = load_episodes(traces_path)
    split = build_split(episodes, rng)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_path = OUTPUT_DIR / "episode_split_70_30.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)

    val_failures = [e for e in split["val"] if not e["success"]]
    eval_episodes = ",".join(f"{e['task_id']}:{e['episode_idx']}" for e in sorted(val_failures, key=lambda x: (x["task_id"], x["episode_idx"])))
    (OUTPUT_DIR / "val_failure_eval_episodes.txt").write_text(eval_episodes + "\n", encoding="utf-8")

    df = load_chunk_df(METRICS_DIR)
    val_keys = {(e["task_id"], e["episode_idx"]) for e in split["val"]}
    df["_episode_key"] = list(zip(df["task_id"].astype(int), df["episode_idx"].astype(int)))
    train_df = df[~df["_episode_key"].isin(val_keys)].copy()
    val_df = df[df["_episode_key"].isin(val_keys)].copy()

    X_train, y_train = train_df[FEATURES], train_df["needs_regeneration"].astype(int)
    X_val, y_val = val_df[FEATURES], val_df["needs_regeneration"].astype(int)

    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=6,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=80,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    val_prob = model.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, val_prob, 0.5)
    metrics["train_chunks"] = int(len(train_df))
    metrics["val_chunks"] = int(len(val_df))
    metrics["train_episodes"] = int(train_df.groupby(["task_id", "episode_idx"]).ngroups)
    metrics["val_episodes"] = int(val_df.groupby(["task_id", "episode_idx"]).ngroups)

    model.save_model(str(OUTPUT_DIR / "catboost_chunk_regen.cbm"))
    (OUTPUT_DIR / "catboost_chunk_regen_features.json").write_text(
        json.dumps({"features": FEATURES, "threshold": 0.38, "split": "70_train_30_val"}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    importance = pd.DataFrame({"feature": FEATURES, "importance": model.get_feature_importance()})
    importance.sort_values("importance", ascending=False).to_csv(
        OUTPUT_DIR / "catboost_feature_importance.csv", index=False
    )

    print(f"Wrote split: {split_path}")
    print(f"Val failure eval episodes ({len(val_failures)}): {eval_episodes}")
    print(f"Train episodes: {split['meta']['train_episodes']}, val episodes: {split['meta']['val_episodes']}")
    print(f"Val chunk metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
