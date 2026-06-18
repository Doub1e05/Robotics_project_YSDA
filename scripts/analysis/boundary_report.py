"""Analysis and plots for fixed-init multi-seed boundary experiments (mimic-video metrics)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ONLINE_CHUNK_METRICS = [
    "action_entropy",
    "action_uncertainty",
    "sampling_stability",
    "chunk_action_variance",
    "chunk_action_delta_norm_mean",
    "chunk_action_delta_norm_max",
    "flow_prediction_error",
    "plan_drift",
    "representation_drift_rate",
    "video_latent_variance_mean",
    "video_latent_entropy",
    "video_latent_norm_std",
    "video_latent_delta_l2_std",
    "goal_latent_distance",
    "latent_oscillation_score",
    "query_latency_sec",
]

POST_CHUNK_METRICS = [
    "flow_prediction_error",
    "representation_drift_rate",
    "plan_drift",
    "video_latent_delta_l2_std",
    "video_action_mutual_information",
]

LEAKAGE_PREFIXES = ("prediction_error_", "actual_", "video_path", "final_t")


def merge_rank_metrics(metrics_root: Path, world_size: int | None = None) -> Path:
    """Merge per-rank episode_traces into metrics_root/episode_traces.jsonl."""
    metrics_root = Path(metrics_root)
    rank_dirs = sorted(metrics_root.glob("rank[0-9]*"))
    if not rank_dirs:
        return metrics_root

    if world_size is not None:
        rank_dirs = [metrics_root / f"rank{i}" for i in range(world_size)]

    traces: list[dict] = []
    for rank_dir in rank_dirs:
        path = rank_dir / "episode_traces.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                traces.append(json.loads(line))

    traces.sort(key=lambda ep: int(ep["meta"]["rollout_id"]))
    for idx, ep in enumerate(traces, start=1):
        ep["meta"]["total_episode_idx"] = idx

    with (metrics_root / "episode_traces.jsonl").open("w", encoding="utf-8") as f:
        for ep in traces:
            f.write(json.dumps(ep, allow_nan=True) + "\n")

    chunk_df = traces_to_chunk_df(traces)
    chunk_df.to_csv(metrics_root / "chunk_metrics.csv", index=False)
    chunk_df.to_json(metrics_root / "chunk_metrics.jsonl", orient="records", lines=True)

    summary = {
        "num_episodes": len(traces),
        "num_successes": int(sum(bool(ep["meta"]["success"]) for ep in traces)),
        "success_rate": sum(bool(ep["meta"]["success"]) for ep in traces) / max(len(traces), 1),
        "total_chunks": int(len(chunk_df)),
    }
    (metrics_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_root


def load_episode_traces(metrics_dir: Path) -> list[dict]:
    metrics_dir = Path(metrics_dir)
    if (metrics_dir / "episode_traces.jsonl").is_file():
        path = metrics_dir / "episode_traces.jsonl"
    elif list(metrics_dir.glob("rank[0-9]*")):
        merge_rank_metrics(metrics_dir)
        path = metrics_dir / "episode_traces.jsonl"
    else:
        raise FileNotFoundError(f"No episode_traces.jsonl under {metrics_dir}")

    traces: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            traces.append(json.loads(line))
    return traces


def traces_to_chunk_df(traces: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for ep in traces:
        meta = ep["meta"]
        for chunk in ep["chunks"]:
            row = {
                "rollout_seed": meta["rollout_seed"],
                "rollout_id": meta.get("rollout_id"),
                "success": bool(meta["success"]),
                "task_id": meta["task_id"],
                "episode_idx": meta["episode_idx"],
                "chunk_id": chunk["chunk_id"],
                "query_timestep": chunk["query_timestep"],
                "inference_step_idx": chunk["inference_step_idx"],
            }
            row.update(chunk.get("metrics") or {})
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col in {"success", "task_description"}:
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            pass
    return df


def earliest_success_max_chunk(traces: list[dict]) -> int:
    success_chunks = [
        int(ep["meta"]["chunk_count"]) - 1
        for ep in traces
        if bool(ep["meta"]["success"]) and int(ep["meta"]["chunk_count"]) > 0
    ]
    if not success_chunks:
        return int(max((int(ep["meta"]["chunk_count"]) - 1 for ep in traces), default=0))
    return int(min(success_chunks))


def filter_to_common_horizon(df: pd.DataFrame, max_chunk_id: int) -> pd.DataFrame:
    out = df[df["chunk_id"] <= max_chunk_id].copy()
    return out


def episode_level_features(chunk_df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for (rollout_seed, success), group in chunk_df.groupby(["rollout_seed", "success"], sort=False):
        row = {"rollout_seed": rollout_seed, "success": bool(success), "fail": int(not success)}
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = group[metric].astype(float)
            row[f"{metric}__mean"] = float(values.mean())
            row[f"{metric}__std"] = float(values.std(ddof=0))
            row[f"{metric}__max"] = float(values.max())
            row[f"{metric}__last"] = float(values.iloc[-1])
            if len(values) >= 3:
                row[f"{metric}__last3_mean"] = float(values.tail(3).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def oriented_auc(y_fail: np.ndarray, scores: np.ndarray) -> tuple[float, str]:
    auc_hi = roc_auc_score(y_fail, scores)
    auc_lo = roc_auc_score(y_fail, -scores)
    if auc_hi >= auc_lo:
        return float(auc_hi), "higher->fail"
    return float(auc_lo), "lower->fail"


def best_balanced_accuracy(y_fail: np.ndarray, scores: np.ndarray, higher_is_fail: bool) -> tuple[float, float]:
    oriented = scores if higher_is_fail else -scores
    thresholds = np.unique(oriented)
    if thresholds.size == 0:
        return float("nan"), float("nan")
    best_acc = -1.0
    best_thr = float(thresholds[0])
    for thr in thresholds:
        pred = oriented >= thr
        tpr = pred[y_fail == 1].mean() if (y_fail == 1).any() else 0.0
        tnr = (~pred)[y_fail == 0].mean() if (y_fail == 0).any() else 0.0
        acc = 0.5 * (tpr + tnr)
        if acc > best_acc:
            best_acc = float(acc)
            best_thr = float(thr)
    return best_acc, best_thr


def rank_episode_features(episode_df: pd.DataFrame) -> pd.DataFrame:
    y = episode_df["fail"].to_numpy(dtype=int)
    rows = []
    for col in episode_df.columns:
        if col in {"rollout_seed", "success", "fail"}:
            continue
        scores = episode_df[col].astype(float).to_numpy()
        if np.unique(scores[~np.isnan(scores)]).size < 2:
            continue
        mask = ~np.isnan(scores)
        if mask.sum() < 4 or len(np.unique(y[mask])) < 2:
            continue
        auc, direction = oriented_auc(y[mask], scores[mask])
        higher_is_fail = direction == "higher->fail"
        bal_acc, thr = best_balanced_accuracy(y[mask], scores[mask], higher_is_fail)
        rows.append(
            {
                "feature": col,
                "oriented_auc": auc,
                "direction": direction,
                "best_balanced_accuracy": bal_acc,
                "fail_mean": float(np.nanmean(scores[(y == 1) & mask])),
                "success_mean": float(np.nanmean(scores[(y == 0) & mask])),
            }
        )
    return pd.DataFrame(rows).sort_values("oriented_auc", ascending=False)


def compute_risk_scores(episode_df: pd.DataFrame) -> pd.DataFrame:
    out = episode_df.copy()

    def col(name: str) -> pd.Series:
        key = f"{name}__mean"
        return out[key] if key in out.columns else pd.Series(np.nan, index=out.index)

    out["risk_high_uncertainty"] = (
        col("action_entropy") + col("action_uncertainty") + col("chunk_action_variance")
    )
    out["risk_overconfidence_latent"] = -(
        col("video_latent_variance_mean") + col("sampling_stability") + col("video_latent_norm_std")
    )
    out["risk_overconfidence_action"] = -(
        col("action_uncertainty") + col("chunk_action_variance") + col("chunk_action_delta_norm_mean")
    )
    return out


def evaluate_risk_scores(episode_df: pd.DataFrame) -> pd.DataFrame:
    scored = compute_risk_scores(episode_df)
    y = scored["fail"].to_numpy(dtype=int)
    rows = []
    for score in ["risk_high_uncertainty", "risk_overconfidence_latent", "risk_overconfidence_action"]:
        values = scored[score].astype(float).to_numpy()
        mask = ~np.isnan(values)
        if mask.sum() < 4 or len(np.unique(y[mask])) < 2:
            continue
        auc, direction = oriented_auc(y[mask], values[mask])
        higher_is_fail = direction == "higher->fail"
        bal_acc, thr = best_balanced_accuracy(y[mask], values[mask], higher_is_fail)
        rows.append(
            {
                "score": score,
                "auc": auc,
                "direction": direction,
                "balanced_accuracy": bal_acc,
                "threshold": thr,
            }
        )
    return pd.DataFrame(rows)


def query_prefix_auc(chunk_df: pd.DataFrame, metric: str, max_chunk_id: int) -> pd.DataFrame:
    rows = []
    for chunk_id in range(max_chunk_id + 1):
        sub = chunk_df[chunk_df["chunk_id"] <= chunk_id]
        ep = episode_level_features(sub, [metric])
        if metric + "__mean" not in ep.columns:
            continue
        y = ep["fail"].to_numpy(dtype=int)
        scores = ep[f"{metric}__mean"].astype(float).to_numpy()
        mask = ~np.isnan(scores)
        if mask.sum() < 4 or len(np.unique(y[mask])) < 2:
            auc = float("nan")
        else:
            auc, _ = oriented_auc(y[mask], scores[mask])
        rows.append({"chunk_id": chunk_id, "metric": metric, "auc": auc})
    return pd.DataFrame(rows)


def plot_query_timeseries(
    chunk_df: pd.DataFrame,
    metrics: list[str],
    out_path: Path,
    title: str,
) -> None:
    grouped = (
        chunk_df.groupby(["success", "chunk_id"])[metrics]
        .mean()
        .reset_index()
    )
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        for success, label, color in [(True, "success", "#2ca02c"), (False, "fail", "#d62728")]:
            part = grouped[grouped["success"] == success]
            if metric not in part.columns:
                continue
            ax.plot(part["chunk_id"], part[metric], marker="o", label=label, color=color, linewidth=1.8)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("chunk_id (query)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_episode_boxplots(episode_df: pd.DataFrame, features: list[str], out_path: Path, title: str) -> None:
    n = len(features)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, feature in zip(axes, features):
        data = [
            episode_df.loc[episode_df["success"], feature].astype(float),
            episode_df.loc[~episode_df["success"], feature].astype(float),
        ]
        ax.boxplot(data, labels=["success", "fail"])
        ax.set_title(feature.replace("__mean", ""))
        ax.grid(True, axis="y", alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_auc_by_query(auc_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for metric, part in auc_df.groupby("metric"):
        ax.plot(part["chunk_id"], part["auc"], marker="o", label=metric)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("max chunk_id in prefix")
    ax.set_ylabel("oriented AUC (fail vs success)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_risk_histograms(episode_df: pd.DataFrame, out_dir: Path) -> None:
    scored = compute_risk_scores(episode_df)
    for score in ["risk_high_uncertainty", "risk_overconfidence_latent", "risk_overconfidence_action"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for success, label, color in [(True, "success", "#2ca02c"), (False, "fail", "#d62728")]:
            values = scored.loc[scored["success"] == success, score].astype(float)
            ax.hist(values, bins=12, alpha=0.55, label=label, color=color)
        ax.set_title(score)
        ax.set_xlabel("score")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{score}_hist.png", dpi=150)
        plt.close(fig)


def run_full_analysis(metrics_dir: Path, analysis_dir: Path | None = None) -> dict[str, object]:
    metrics_dir = Path(metrics_dir)
    analysis_dir = Path(analysis_dir or metrics_dir / "analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    traces = load_episode_traces(metrics_dir)
    chunk_df = traces_to_chunk_df(traces)
    max_chunk_id = earliest_success_max_chunk(traces)
    chunk_df = filter_to_common_horizon(chunk_df, max_chunk_id)

    episode_df = episode_level_features(chunk_df, ONLINE_CHUNK_METRICS)
    ranked = rank_episode_features(episode_df)
    ranked.to_csv(analysis_dir / "ranked_episode_features.csv", index=False)
    episode_df.to_csv(analysis_dir / "episode_features.csv", index=False)
    chunk_df.to_csv(analysis_dir / "chunk_metrics_common_horizon.csv", index=False)

    uncertainty_metrics = [m for m in ONLINE_CHUNK_METRICS[:8] if m in chunk_df.columns]
    post_metrics = [m for m in POST_CHUNK_METRICS if m in chunk_df.columns]
    plot_query_timeseries(
        chunk_df,
        uncertainty_metrics,
        analysis_dir / "uncertainty_query_timeseries.png",
        f"Query-level metrics (chunk<= {max_chunk_id})",
    )
    if post_metrics:
        plot_query_timeseries(
            chunk_df,
            post_metrics,
            analysis_dir / "prediction_error_query_timeseries.png",
            "Post-chunk / drift metrics",
        )

    top_features = ranked.head(8)["feature"].tolist()
    if top_features:
        plot_episode_boxplots(
            episode_df,
            top_features,
            analysis_dir / "episode_metric_boxplots.png",
            "Top episode-level separators",
        )

    risk_eval = evaluate_risk_scores(episode_df)
    risk_eval.to_csv(analysis_dir / "risk_score_eval.csv", index=False)
    plot_risk_histograms(episode_df, analysis_dir)

    auc_parts = []
    for metric in uncertainty_metrics[:5]:
        auc_parts.append(query_prefix_auc(chunk_df, metric, max_chunk_id))
    if auc_parts:
        auc_df = pd.concat(auc_parts, ignore_index=True)
        auc_df.to_csv(analysis_dir / "online_metric_auc_by_query.csv", index=False)
        plot_auc_by_query(
            auc_df,
            analysis_dir / "online_metric_auc_by_query.png",
            "Online metric AUC by query prefix",
        )

    summary = {
        "num_rollouts": len(traces),
        "num_success": int(sum(bool(ep["meta"]["success"]) for ep in traces)),
        "num_fail": int(sum(not bool(ep["meta"]["success"]) for ep in traces)),
        "max_chunk_id_common": max_chunk_id,
        "success_rate": float(np.mean([bool(ep["meta"]["success"]) for ep in traces])),
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary": summary,
        "episode_df": episode_df,
        "chunk_df": chunk_df,
        "ranked": ranked,
        "risk_eval": risk_eval,
        "analysis_dir": analysis_dir,
    }


def episode_outcomes_table(metrics_dir: Path) -> pd.DataFrame:
    traces = load_episode_traces(metrics_dir)
    rows = []
    for ep in traces:
        meta = ep["meta"]
        rows.append(
            {
                "rollout_id": int(meta["rollout_id"]),
                "rollout_seed": int(meta["rollout_seed"]),
                "success": bool(meta["success"]),
                "step_count": int(meta["step_count"]),
                "chunk_count": int(meta["chunk_count"]),
                "regen_strategy": meta.get("regen_strategy", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("rollout_id")


def compare_boundary_runs(
    baseline_metrics_dir: Path,
    strategy_metrics_dir: Path,
    analysis_dir: Path,
    baseline_label: str = "baseline",
    strategy_label: str = "catboost_select3",
) -> dict[str, object]:
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    base = episode_outcomes_table(baseline_metrics_dir)
    strat = episode_outcomes_table(strategy_metrics_dir)
    paired = base.merge(
        strat,
        on=["rollout_id", "rollout_seed"],
        suffixes=(f"_{baseline_label}", f"_{strategy_label}"),
    )
    paired["improved"] = (~paired[f"success_{baseline_label}"]) & paired[f"success_{strategy_label}"]
    paired["regressed"] = paired[f"success_{baseline_label}"] & (~paired[f"success_{strategy_label}"])

    summary = {
        f"{baseline_label}_sr": float(base["success"].mean()),
        f"{strategy_label}_sr": float(strat["success"].mean()),
        f"{baseline_label}_n": int(len(base)),
        f"{strategy_label}_n": int(len(strat)),
        "paired_improved": int(paired["improved"].sum()),
        "paired_regressed": int(paired["regressed"].sum()),
        "paired_both_success": int((paired[f"success_{baseline_label}"] & paired[f"success_{strategy_label}"]).sum()),
        "paired_both_fail": int((~paired[f"success_{baseline_label}"] & ~paired[f"success_{strategy_label}"]).sum()),
    }
    paired.to_csv(analysis_dir / "paired_outcomes.csv", index=False)
    (analysis_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        [baseline_label, strategy_label],
        [summary[f"{baseline_label}_sr"], summary[f"{strategy_label}_sr"]],
        color=["#7f7f7f", "#1f77b4"],
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("success rate")
    axes[0].set_title("Episode success rate")
    axes[0].grid(True, axis="y", alpha=0.3)

    cats = ["both ok", "improved", "regressed", "both fail"]
    counts = [
        summary["paired_both_success"],
        summary["paired_improved"],
        summary["paired_regressed"],
        summary["paired_both_fail"],
    ]
    axes[1].bar(cats, counts, color=["#2ca02c", "#9467bd", "#d62728", "#ff7f0e"])
    axes[1].set_title("Paired outcomes (same seed)")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(analysis_dir / "baseline_vs_strategy_sr.png", dpi=150)
    plt.close(fig)

    return {"summary": summary, "paired": paired, "analysis_dir": analysis_dir}
