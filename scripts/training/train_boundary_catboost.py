#!/usr/bin/env python3
"""Train chunk-level CatBoost on boundary run (task0/init9, 40 rollouts)."""

from __future__ import annotations

import json
import random
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
BASELINE_METRICS = (
    REPO_ROOT / "eval_outputs/libero_spatial/boundary_runs/task0_init9_baseline_40rollouts/metrics"
)
ARTIFACTS_DIR = REPO_ROOT / "artifacts/libero_spatial/boundary_task0_init9"
OUTPUT_DIR = BASELINE_METRICS / "chunk_regeneration_classifiers_boundary"

FEATURES = [
    "chunk_action_delta_norm_mean",
    "flow_prediction_error",
    "action_chunk_temporal_consistency",
    "plan_drift",
    "action_entropy",
    "video_action_mutual_information",
    "mlp_activation_mean",
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_rollout_split(episodes: list[dict], val_fraction: float = 0.3) -> dict:
    rng = random.Random(RANDOM_STATE)
    success = [e for e in episodes if e["success"]]
    failure = [e for e in episodes if not e["success"]]
    n_val_success = max(1, round(len(success) * val_fraction))
    n_val_failure = max(1, round(len(failure) * val_fraction))
    val = rng.sample(success, n_val_success) + rng.sample(failure, n_val_failure)
    val_ids = {e["rollout_id"] for e in val}
    train = [e for e in episodes if e["rollout_id"] not in val_ids]
    return {"train": train, "val": val, "val_rollout_ids": sorted(val_ids)}


def main() -> None:
    traces_path = BASELINE_METRICS / "episode_traces.jsonl"
    if not traces_path.is_file():
        raise FileNotFoundError(f"Missing {traces_path}. Run baseline boundary eval first.")

    episodes = []
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)["meta"]
        episodes.append(
            {
                "rollout_id": int(meta["rollout_id"]),
                "rollout_seed": int(meta["rollout_seed"]),
                "success": bool(meta["success"]),
                "task_id": int(meta["task_id"]),
                "episode_idx": int(meta["episode_idx"]),
            }
        )

    split = build_rollout_split(episodes)
    val_ids = set(split["val_rollout_ids"])

    df = pd.read_csv(BASELINE_METRICS / "chunk_metrics.csv")
    df["success"] = df["success"].map(lambda v: v is True or str(v).strip().lower() in {"true", "1", "yes"})
    df["needs_regeneration"] = (~df["success"]).astype(int)
    df["rollout_id"] = df["rollout_id"].astype(int)

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    train_df = df[~df["rollout_id"].isin(val_ids)].copy()
    val_df = df[df["rollout_id"].isin(val_ids)].copy()

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
    metrics.update(
        {
            "train_chunks": int(len(train_df)),
            "val_chunks": int(len(val_df)),
            "train_rollouts": int(train_df["rollout_id"].nunique()),
            "val_rollouts": int(val_df["rollout_id"].nunique()),
            "train_episode_success_rate": float(train_df.groupby("rollout_id")["success"].first().mean()),
            "val_episode_success_rate": float(val_df.groupby("rollout_id")["success"].first().mean()),
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(ARTIFACTS_DIR / "catboost_chunk_regen.cbm"))
    model.save_model(str(OUTPUT_DIR / "catboost_chunk_regen.cbm"))

    (ARTIFACTS_DIR / "catboost_chunk_regen_features.json").write_text(
        json.dumps({"features": FEATURES, "source": "boundary_task0_init9_baseline_40rollouts"}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "rollout_split_70_30.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fi = pd.DataFrame(
        {"feature": FEATURES, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    fi.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Model: {ARTIFACTS_DIR / 'catboost_chunk_regen.cbm'}")


if __name__ == "__main__":
    main()
