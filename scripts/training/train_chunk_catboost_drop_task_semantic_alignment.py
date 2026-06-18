#!/usr/bin/env python3
"""
Train chunk-level CatBoost on the fixed 70/30 split, dropping `task_semantic_alignment`.

Outputs a new model and reports under:
  <metrics_dir>/chunk_regeneration_classifiers_v3_drop_task_semantic_alignment/
"""

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

SPLIT_DIR = METRICS_DIR / "chunk_regeneration_classifiers_v2_70_30"
SPLIT_PATH = SPLIT_DIR / "episode_split_70_30.json"
FEATURES_PATH = SPLIT_DIR / "catboost_chunk_regen_features.json"

OUTPUT_DIR = METRICS_DIR / "chunk_regeneration_classifiers_v3_drop_task_semantic_alignment"


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
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Missing split file: {SPLIT_PATH}")
    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    val_keys = {(int(e["task_id"]), int(e["episode_idx"])) for e in split["val"]}

    original_features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))["features"]
    features = [f for f in original_features if f != "task_semantic_alignment"]
    if len(features) != len(original_features) - 1:
        raise RuntimeError("Expected to drop exactly one feature: task_semantic_alignment")

    df = pd.read_csv(METRICS_DIR / "chunk_metrics.csv")
    df["success"] = df["success"].map(lambda v: v is True or str(v).strip().lower() in {"true", "1", "yes"})
    df["needs_regeneration"] = (~df["success"]).astype(int)

    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["_episode_key"] = list(zip(df["task_id"].astype(int), df["episode_idx"].astype(int)))
    train_df = df[~df["_episode_key"].isin(val_keys)].copy()
    val_df = df[df["_episode_key"].isin(val_keys)].copy()

    X_train, y_train = train_df[features], train_df["needs_regeneration"].astype(int)
    X_val, y_val = val_df[features], val_df["needs_regeneration"].astype(int)

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
    metrics["dropped_feature"] = "task_semantic_alignment"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUTPUT_DIR / "catboost_chunk_regen.cbm"))

    (OUTPUT_DIR / "catboost_chunk_regen_features.json").write_text(
        json.dumps(
            {
                "features": features,
                "dropped_feature": "task_semantic_alignment",
                "base_features_file": str(FEATURES_PATH.relative_to(REPO_ROOT)),
                "split_file": str(SPLIT_PATH.relative_to(REPO_ROOT)),
                "split": "70_train_30_val",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    importance = pd.DataFrame({"feature": features, "importance": model.get_feature_importance()})
    importance.sort_values("importance", ascending=False).to_csv(
        OUTPUT_DIR / "catboost_feature_importance.csv", index=False
    )

    print(f"[saved] {OUTPUT_DIR / 'catboost_chunk_regen.cbm'}")
    print(f"[saved] {OUTPUT_DIR / 'validation_metrics.json'}")


if __name__ == "__main__":
    main()

