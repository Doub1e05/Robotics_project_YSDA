#!/usr/bin/env python3
"""Train action-level CatBoost for dynamic execute horizon."""

from __future__ import annotations

import json
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
SPLIT_PATH = METRICS_DIR / "chunk_regeneration_classifiers_v2_70_30" / "episode_split_70_30.json"
OUTPUT_DIR = METRICS_DIR / "action_regeneration_classifiers_v2_70_30"

MAX_TIMESTEPS_FAILURE = 120
MIN_EXEC = 3
MAX_EXEC = 8

FEATURES = [
    "action_offset_in_chunk",
    "model_action_delta_norm",
    "model_action_norm",
    "model_gripper_command",
    "model_gripper_sign",
    "model_rotation6d_norm",
    "model_translation_norm",
]


def action_fail_metrics(y_true: np.ndarray, fail_prob: np.ndarray, fail_threshold: float) -> dict[str, float]:
    pred_fail = (fail_prob >= fail_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_fail, labels=[0, 1]).ravel()
    return {
        "threshold_fail": float(fail_threshold),
        "roc_auc_fail": float(roc_auc_score(y_true, fail_prob)),
        "average_precision_fail": float(average_precision_score(y_true, fail_prob)),
        "accuracy": float(accuracy_score(y_true, pred_fail)),
        "precision_fail": float(precision_score(y_true, pred_fail, zero_division=0)),
        "recall_fail": float(recall_score(y_true, pred_fail, zero_division=0)),
        "f1_fail": float(f1_score(y_true, pred_fail, zero_division=0)),
        "predicted_fail_rate": float(pred_fail.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def simulate_horizon(group: pd.DataFrame, threshold_success: float) -> int:
    probs = group.sort_values("action_offset_in_chunk")["success_prob"].to_numpy(dtype=float)
    if probs.size == 0:
        return MIN_EXEC
    max_actions = min(MAX_EXEC, probs.size)
    horizon = min(MIN_EXEC, max_actions)
    for idx in range(horizon, max_actions):
        if probs[idx] >= threshold_success:
            horizon = idx + 1
        else:
            break
    return max(1, int(horizon))


def threshold_sweep(val_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    chunks = list(val_df.groupby(["total_episode_idx", "chunk_id", "inference_step_idx"], sort=False))
    for threshold in np.round(np.linspace(0.45, 0.85, 41), 3):
        confidence = (val_df["success_prob"] >= threshold).astype(int)
        confidence_rate = float(confidence.mean())
        horizons = [simulate_horizon(group, float(threshold)) for _, group in chunks]
        rows.append(
            {
                "action_success_threshold": float(threshold),
                "confidence_rate": confidence_rate,
                "avg_execute_horizon": float(np.mean(horizons)),
                "p_execute_min3": float(np.mean(np.array(horizons) == MIN_EXEC)),
                "p_execute_max8": float(np.mean(np.array(horizons) == MAX_EXEC)),
            }
        )
    return pd.DataFrame(rows)


def pick_threshold(sweep_df: pd.DataFrame) -> float:
    feasible = sweep_df[
        (sweep_df["confidence_rate"] >= 0.35)
        & (sweep_df["confidence_rate"] <= 0.75)
        & (sweep_df["avg_execute_horizon"] >= 4.0)
        & (sweep_df["avg_execute_horizon"] <= 7.0)
    ].copy()
    target_horizon = 5.5
    if feasible.empty:
        feasible = sweep_df.copy()
    feasible["objective"] = (
        -(feasible["avg_execute_horizon"] - target_horizon).abs()
        - (feasible["confidence_rate"] - 0.5).abs()
        + 0.2 * feasible["p_execute_max8"]
    )
    return float(feasible.sort_values("objective", ascending=False).iloc[0]["action_success_threshold"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))

    action_df = pd.read_csv(METRICS_DIR / "action_metrics.csv")
    action_df["success"] = action_df["success"].map(
        lambda v: v is True or str(v).strip().lower() in {"true", "1", "yes"}
    )
    action_df["needs_regeneration"] = (~action_df["success"]).astype(int)
    # Match chunk trimming policy: failures only considered for first 6 seconds.
    action_df = action_df[action_df["success"] | (action_df["timestep"] < MAX_TIMESTEPS_FAILURE)].copy()

    for col in FEATURES:
        action_df[col] = pd.to_numeric(action_df[col], errors="coerce")

    train_keys = {(int(e["task_id"]), int(e["episode_idx"])) for e in split["train"]}
    val_keys = {(int(e["task_id"]), int(e["episode_idx"])) for e in split["val"]}
    action_df["_episode_key"] = list(zip(action_df["task_id"].astype(int), action_df["episode_idx"].astype(int)))
    train_df = action_df[action_df["_episode_key"].isin(train_keys)].copy()
    val_df = action_df[action_df["_episode_key"].isin(val_keys)].copy()

    X_train = train_df[FEATURES]
    y_train = train_df["needs_regeneration"].astype(int)
    X_val = val_df[FEATURES]
    y_val = val_df["needs_regeneration"].astype(int)

    model = CatBoostClassifier(
        iterations=1400,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=6,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=100,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    val_fail_prob = model.predict_proba(X_val)[:, 1]
    val_success_prob = 1.0 - val_fail_prob
    val_df["fail_prob"] = val_fail_prob
    val_df["success_prob"] = val_success_prob

    fail_metrics = action_fail_metrics(y_val.to_numpy(), val_fail_prob, 0.5)
    sweep_df = threshold_sweep(val_df)
    threshold = pick_threshold(sweep_df)

    model.save_model(str(OUTPUT_DIR / "catboost_action_regen.cbm"))
    (OUTPUT_DIR / "catboost_action_regen_features.json").write_text(
        json.dumps(
            {
                "features": FEATURES,
                "action_success_threshold": threshold,
                "max_timesteps_failure": MAX_TIMESTEPS_FAILURE,
                "min_execute_actions": MIN_EXEC,
                "max_execute_actions": MAX_EXEC,
                "train_episodes": len(split["train"]),
                "val_episodes": len(split["val"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "validation_fail_metrics_at_0p5.json").write_text(
        json.dumps(fail_metrics, indent=2), encoding="utf-8"
    )
    sweep_df.to_csv(OUTPUT_DIR / "action_success_threshold_sweep.csv", index=False)
    (OUTPUT_DIR / "recommended_action_success_threshold.txt").write_text(f"{threshold:.3f}\n", encoding="utf-8")

    importance = pd.DataFrame({"feature": FEATURES, "importance": model.get_feature_importance()})
    importance.sort_values("importance", ascending=False).to_csv(
        OUTPUT_DIR / "catboost_action_feature_importance.csv", index=False
    )

    print(f"Train actions: {len(train_df)}, val actions: {len(val_df)}")
    print(f"Validation fail metrics @0.5: {json.dumps(fail_metrics, indent=2)}")
    print(f"Recommended success threshold: {threshold:.3f}")


if __name__ == "__main__":
    main()
