from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_ROOT = (
    REPO_ROOT
    / "eval_outputs"
    / "libero_spatial"
    / "encoder_hidden"
    / "task0_init9_10rollouts_6sec"
    / "metrics"
)

SCALAR_METRICS = [
    "encoder_initial_final_pooled_cosine",
    "encoder_adjacent_pooled_cosine_mean",
    "encoder_adjacent_token_cosine_mean",
    "encoder_pooled_norm_mean",
    "encoder_pooled_norm_std",
    "encoder_delta_norm_mean",
    "encoder_delta_norm_std",
    "query_latency_sec",
]

CHUNK_TIMESERIES_METRICS = [
    "encoder_initial_final_pooled_cosine",
    "encoder_adjacent_pooled_cosine_mean",
    "encoder_adjacent_token_cosine_mean",
    "encoder_delta_norm_mean",
]

SUCCESS_COLOR = "#2a9d8f"
FAILURE_COLOR = "#d1495b"


def _set_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("ggplot")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_representation_metrics(metrics_root: Path = DEFAULT_METRICS_ROOT) -> pd.DataFrame:
    metrics_root = Path(metrics_root)
    rank_paths = sorted(metrics_root.glob("rank*/representation_metrics.jsonl"))
    if not rank_paths:
        raise FileNotFoundError(f"No rank*/representation_metrics.jsonl under {metrics_root}")

    frames = [pd.read_json(path, lines=True) for path in rank_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["total_episode_idx", "chunk_id"]).reset_index(drop=True)
    df["success"] = df["success"].astype(bool)
    df["success_label"] = np.where(df["success"], "success", "failure")
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["z_t"] = out["encoder_pooled_hw_l2_normalized_t_d"].map(lambda x: np.asarray(x, dtype=np.float32))
    out["dz_t"] = out["encoder_pooled_hw_l2_normalized_delta_t_d"].map(lambda x: np.asarray(x, dtype=np.float32))
    out["adj_pooled_curve"] = out["encoder_adjacent_pooled_cosine_t_minus_1"].map(
        lambda x: np.asarray(x, dtype=np.float32)
    )
    out["adj_token_curve"] = out["encoder_adjacent_token_cosine_mean_t_minus_1"].map(
        lambda x: np.asarray(x, dtype=np.float32)
    )
    out["z_abs_mean_per_t"] = out["z_t"].map(lambda arr: np.mean(np.abs(arr), axis=1))
    out["z_norm_per_t"] = out["z_t"].map(lambda arr: np.linalg.norm(arr, axis=1))
    out["dz_norm_per_t"] = out["dz_t"].map(lambda arr: np.linalg.norm(arr, axis=1))
    out["mean_z"] = out["z_t"].map(lambda arr: arr.mean(axis=0))
    out["final_z"] = out["z_t"].map(lambda arr: arr[-1])
    return out


def build_episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    episode = (
        df.groupby("total_episode_idx", as_index=False)
        .agg(
            success=("success", "first"),
            termination_reason=("termination_reason", "first"),
            chunk_count=("chunk_id", "nunique"),
            task_id=("task_id", "first"),
            episode_idx=("episode_idx", "first"),
            query_latency_mean=("query_latency_sec", "mean"),
            initial_final_cos_mean=("encoder_initial_final_pooled_cosine", "mean"),
            pooled_cos_mean=("encoder_adjacent_pooled_cosine_mean", "mean"),
            token_cos_mean=("encoder_adjacent_token_cosine_mean", "mean"),
            delta_norm_mean=("encoder_delta_norm_mean", "mean"),
        )
    )
    episode["success_label"] = np.where(episode["success"], "success", "failure")
    return episode


def print_summary(df: pd.DataFrame) -> dict[str, object]:
    episode = build_episode_summary(df)
    summary = {
        "num_chunks": int(len(df)),
        "num_episodes": int(len(episode)),
        "num_successes": int(episode["success"].sum()),
        "num_failures": int((~episode["success"]).sum()),
        "success_rate": float(episode["success"].mean()),
        "xattn_layer_idx": int(df["encoder_xattn_layer_idx"].iloc[0]),
        "hidden_shape": list(df["encoder_hidden_state_shape"].iloc[0]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _label_color(success: bool) -> tuple[str, str]:
    return ("success", SUCCESS_COLOR) if success else ("failure", FAILURE_COLOR)


def _stack_curves(series: pd.Series) -> np.ndarray:
    return np.stack(series.to_list(), axis=0)


def plot_scalar_boxplots(df: pd.DataFrame) -> plt.Figure:
    _set_style()
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    axes = axes.ravel()
    for ax, metric in zip(axes, SCALAR_METRICS, strict=False):
        if sns is not None:
            sns.boxplot(data=df, x="success_label", y=metric, ax=ax, palette=[FAILURE_COLOR, SUCCESS_COLOR])
            sns.stripplot(data=df, x="success_label", y=metric, ax=ax, color="black", alpha=0.35, size=3)
        else:  # pragma: no cover
            values = [df.loc[df["success_label"] == label, metric].to_numpy() for label in ["failure", "success"]]
            ax.boxplot(values, labels=["failure", "success"])
        ax.set_title(metric.replace("encoder_", "").replace("_", " "))
        ax.set_xlabel("")
    fig.suptitle("Chunk-level scalar metrics: success vs failure", fontsize=18)
    return fig


def plot_chunk_timeseries(df: pd.DataFrame) -> plt.Figure:
    _set_style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()
    for ax, metric in zip(axes, CHUNK_TIMESERIES_METRICS, strict=False):
        for success in [False, True]:
            label, color = _label_color(success)
            grouped = (
                df[df["success"] == success]
                .groupby("chunk_id")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )
            if grouped.empty:
                continue
            x = grouped["chunk_id"].to_numpy()
            y = grouped["mean"].to_numpy()
            s = grouped["std"].fillna(0.0).to_numpy()
            ax.plot(x, y, marker="o", label=label, color=color)
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.2)
        ax.set_title(metric.replace("encoder_", "").replace("_", " "))
        ax.set_xlabel("chunk id")
    axes[0].legend()
    fig.suptitle("Chunk progression inside an episode", fontsize=18)
    return fig


def _plot_curve_grouped(ax: plt.Axes, df: pd.DataFrame, column: str, x: np.ndarray, title: str, ylabel: str) -> None:
    for success in [False, True]:
        label, color = _label_color(success)
        subset = df[df["success"] == success]
        if subset.empty:
            continue
        arr = _stack_curves(subset[column])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)


def plot_temporal_curves(df: pd.DataFrame) -> plt.Figure:
    _set_style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()
    _plot_curve_grouped(
        axes[0],
        df,
        "adj_pooled_curve",
        np.arange(1, 16),
        "Adjacent pooled cosine over latent time",
        "cosine",
    )
    _plot_curve_grouped(
        axes[1],
        df,
        "adj_token_curve",
        np.arange(1, 16),
        "Adjacent token cosine mean over latent time",
        "cosine",
    )
    _plot_curve_grouped(
        axes[2],
        df,
        "z_norm_per_t",
        np.arange(16),
        "Pooled latent norm over time",
        "L2 norm",
    )
    _plot_curve_grouped(
        axes[3],
        df,
        "dz_norm_per_t",
        np.arange(1, 16),
        "Delta norm over time",
        "L2 norm",
    )
    axes[0].legend()
    fig.suptitle("Temporal dynamics of pooled hidden states", fontsize=18)
    return fig


def plot_temporal_heatmaps(df: pd.DataFrame) -> plt.Figure:
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    failure = _stack_curves(df[df["success"] == False]["z_abs_mean_per_t"])  # noqa: E712
    success = _stack_curves(df[df["success"] == True]["z_abs_mean_per_t"])  # noqa: E712
    failure_mean = failure.mean(axis=0, keepdims=True)
    success_mean = success.mean(axis=0, keepdims=True)
    diff = success_mean - failure_mean
    cmap = "coolwarm"
    for ax, arr, title in zip(
        axes,
        [failure_mean, success_mean, diff],
        ["failure mean |z_t|", "success mean |z_t|", "success - failure"],
        strict=False,
    ):
        im = ax.imshow(arr, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("latent timestep")
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Average pooled activation magnitude over time", fontsize=18)
    return fig


def plot_example_episodes(df: pd.DataFrame) -> plt.Figure:
    _set_style()
    episode = build_episode_summary(df)
    success_ids = episode.loc[episode["success"], "total_episode_idx"].tolist()
    failure_ids = episode.loc[~episode["success"], "total_episode_idx"].tolist()
    chosen = []
    if failure_ids:
        chosen.append((failure_ids[0], "failure", FAILURE_COLOR))
    if success_ids:
        chosen.append((success_ids[0], "success", SUCCESS_COLOR))

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()
    metrics = CHUNK_TIMESERIES_METRICS
    for ax, metric in zip(axes, metrics, strict=False):
        for episode_id, label, color in chosen:
            subset = df[df["total_episode_idx"] == episode_id].sort_values("chunk_id")
            ax.plot(subset["chunk_id"], subset[metric], marker="o", label=f"{label} ep {episode_id}", color=color)
        ax.set_title(metric.replace("encoder_", "").replace("_", " "))
        ax.set_xlabel("chunk id")
    axes[0].legend()
    fig.suptitle("Example rollout trajectories", fontsize=18)
    return fig


def plot_chunk_pca(df: pd.DataFrame) -> plt.Figure:
    _set_style()
    features = np.stack(df["mean_z"].to_list(), axis=0)
    coords = PCA(n_components=2, random_state=0).fit_transform(features)
    plot_df = df.copy()
    plot_df["pca1"] = coords[:, 0]
    plot_df["pca2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for success in [False, True]:
        subset = plot_df[plot_df["success"] == success]
        label, color = _label_color(success)
        ax.scatter(subset["pca1"], subset["pca2"], c=color, label=label, alpha=0.7, s=50)
    ax.set_title("PCA of chunk-level mean pooled hidden states")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    return fig


def plot_top_dimension_differences(df: pd.DataFrame, top_k: int = 20) -> plt.Figure:
    _set_style()
    success = np.stack(df[df["success"] == True]["final_z"].to_list(), axis=0)  # noqa: E712
    failure = np.stack(df[df["success"] == False]["final_z"].to_list(), axis=0)  # noqa: E712
    diff = success.mean(axis=0) - failure.mean(axis=0)
    top_idx = np.argsort(np.abs(diff))[-top_k:]
    top_vals = diff[top_idx]

    order = np.argsort(top_vals)
    top_idx = top_idx[order]
    top_vals = top_vals[order]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    colors = [SUCCESS_COLOR if value > 0 else FAILURE_COLOR for value in top_vals]
    ax.barh([str(i) for i in top_idx], top_vals, color=colors)
    ax.set_title("Top final-state dimensions by success-failure mean difference")
    ax.set_xlabel("success mean - failure mean")
    ax.set_ylabel("dimension id")
    return fig


def plot_episode_level_summary(df: pd.DataFrame) -> plt.Figure:
    _set_style()
    episode = build_episode_summary(df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    counts = episode["success_label"].value_counts().reindex(["failure", "success"]).fillna(0)
    axes[0].bar(counts.index, counts.values, color=[FAILURE_COLOR, SUCCESS_COLOR])
    axes[0].set_title("Episode counts")
    axes[0].set_ylabel("count")

    if sns is not None:
        sns.scatterplot(
            data=episode,
            x="chunk_count",
            y="query_latency_mean",
            hue="success_label",
            palette={"failure": FAILURE_COLOR, "success": SUCCESS_COLOR},
            s=90,
            ax=axes[1],
        )
    else:  # pragma: no cover
        for label, color in [("failure", FAILURE_COLOR), ("success", SUCCESS_COLOR)]:
            subset = episode[episode["success_label"] == label]
            axes[1].scatter(subset["chunk_count"], subset["query_latency_mean"], c=color, label=label)
        axes[1].legend()
    axes[1].set_title("Episode-level latency vs number of chunks")
    axes[1].set_xlabel("chunk count")
    axes[1].set_ylabel("mean query latency (sec)")
    return fig


def load_representation_metrics_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_json(path, lines=True)
    df = df.sort_values(["total_episode_idx", "chunk_id"]).reset_index(drop=True)
    df["success"] = df["success"].astype(bool)
    df["success_label"] = np.where(df["success"], "success", "failure")
    return df


def plot_rollout_overlay(df: pd.DataFrame, metrics: list[str] | None = None) -> plt.Figure:
    _set_style()
    metrics = metrics or [
        "encoder_initial_final_pooled_cosine",
        "encoder_delta_norm_mean",
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics, strict=False):
        for episode_id, group in df.groupby("total_episode_idx"):
            group = group.sort_values("chunk_id")
            success = bool(group["success"].iloc[0])
            color = SUCCESS_COLOR if success else FAILURE_COLOR
            ax.plot(
                group["chunk_id"],
                group[metric],
                marker="o",
                alpha=0.85,
                color=color,
                label=f"ep {episode_id} ({'ok' if success else 'fail'})",
            )
        ax.set_title(metric.replace("encoder_", "").replace("_", " "))
        ax.set_xlabel("chunk id")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Same init, different seeds: per-rollout trajectories", fontsize=16)
    return fig


def plot_cross_rollout_variability(df: pd.DataFrame, metric: str, chunk_ids: list[int] | None = None) -> plt.Figure:
    _set_style()
    if chunk_ids is None:
        chunk_ids = sorted(df["chunk_id"].unique())[:: max(1, len(df["chunk_id"].unique()) // 6)][:6]
    plot_rows = []
    for chunk_id in chunk_ids:
        subset = df[df["chunk_id"] == chunk_id]
        for _, row in subset.iterrows():
            plot_rows.append(
                {
                    "chunk_id": int(chunk_id),
                    "value": float(row[metric]),
                    "success_label": row["success_label"],
                }
            )
    plot_df = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    if sns is not None:
        sns.boxplot(data=plot_df, x="chunk_id", y="value", hue="success_label", ax=ax, palette=[FAILURE_COLOR, SUCCESS_COLOR])
        sns.stripplot(
            data=plot_df,
            x="chunk_id",
            y="value",
            hue="success_label",
            dodge=True,
            ax=ax,
            color="black",
            alpha=0.5,
            size=4,
            legend=False,
        )
    else:  # pragma: no cover
        for chunk_id in chunk_ids:
            values = plot_df.loc[plot_df["chunk_id"] == chunk_id, "value"].to_numpy()
            ax.boxplot(values, positions=[chunk_id])
    ax.set_title(f"Cross-rollout spread: {metric.replace('encoder_', '')}")
    ax.set_xlabel("chunk id (same query index across rollouts)")
    fig.suptitle("Stochastic spread at fixed chunk indices", fontsize=16)
    return fig


def plot_scalar_correlation(df: pd.DataFrame) -> plt.Figure:
    _set_style()
    cols = [c for c in SCALAR_METRICS if c in df.columns]
    corr = df[cols].corr(numeric_only=True)
    short = [c.replace("encoder_", "").replace("_", "\n") for c in corr.columns]
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if sns is not None:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, xticklabels=short, yticklabels=short)
    else:  # pragma: no cover
        im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(short)), short, rotation=45, ha="right")
        ax.set_yticks(range(len(short)), short)
        plt.colorbar(im, ax=ax)
    ax.set_title("Correlation between chunk-level scalar encoder metrics")
    return fig


def plot_rollout_fingerprint_heatmap(df: pd.DataFrame) -> plt.Figure:
    _set_style()
    episode = build_episode_summary(df)
    cols = [
        "initial_final_cos_mean",
        "pooled_cos_mean",
        "token_cos_mean",
        "delta_norm_mean",
        "query_latency_mean",
    ]
    mat = episode[cols].to_numpy(dtype=float)
    mat = (mat - mat.mean(axis=0, keepdims=True)) / (mat.std(axis=0, keepdims=True) + 1e-8)
    labels = [
        f"ep{int(r.total_episode_idx)} ({'ok' if r.success else 'fail'})"
        for r in episode.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(labels))), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(len(cols)), [c.replace("_", "\n") for c in cols], rotation=30, ha="right")
    ax.set_title("Z-scored episode fingerprints (one row per rollout)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="z-score")
    return fig
