#!/usr/bin/env python3
"""Two-sample Kolmogorov–Smirnov test: success vs failure chunk metric distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PLOT_METRIC_COLUMNS = [
    "video_latent_norm_mean",
    "video_latent_norm_std",
    "video_latent_delta_l2_std",
    "video_latent_cosine_initial_final",
    "video_latent_variance_mean",
    "video_latent_entropy",
    "flow_prediction_error",
    "latent_path_efficiency",
    "attention_sparsity",
    "video_action_alignment_score",
    "decoder_hidden_norm_mean",
    "decoder_hidden_norm_std",
    "mlp_activation_mean",
    "mlp_sparsity",
    "residual_stream_norm",
    "action_entropy",
    "action_uncertainty",
    "action_chunk_temporal_consistency",
    "action_chunk_smoothness",
    "goal_latent_distance",
    "representation_drift_rate",
    "plan_drift",
    "object_interaction_confidence",
    "instruction_attention_score",
    "task_semantic_alignment",
    "latent_oscillation_score",
    "video_action_cosine_similarity",
    "video_action_mutual_information",
    "score_norm_mean",
    "sampling_stability",
    "latent_success_alignment",
    "chunk_action_variance",
    "chunk_action_delta_norm_mean",
    "chunk_action_delta_norm_max",
    "chunk_gripper_switches",
    "query_latency_sec",
]


def _load_chunk_metrics(metrics_dir: Path) -> pd.DataFrame:
    csv_path = metrics_dir / "chunk_metrics.csv"
    jsonl_path = metrics_dir / "chunk_metrics.jsonl"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif jsonl_path.exists():
        df = pd.read_json(jsonl_path, lines=True)
    else:
        raise FileNotFoundError(f"No chunk metrics found under {metrics_dir}")

    df["success"] = df["success"].map(
        lambda value: value is True or str(value).strip().lower() in {"true", "1", "yes"}
    )
    df["chunk_id"] = df["chunk_id"].astype(int)
    return df


def _ks_test(success_values: np.ndarray, failure_values: np.ndarray) -> tuple[float, float]:
    """Return (ks_statistic, p_value) for two-sample KS test (two-sided)."""
    result = stats.ks_2samp(success_values, failure_values, method="auto")
    return float(result.statistic), float(result.pvalue)


def compute_ks_table(
    df: pd.DataFrame,
    *,
    metrics: list[str],
    min_samples_per_group: int,
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        if metric not in df.columns:
            continue

        metric_df = df[["chunk_id", "success", metric]].copy()
        metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
        metric_df = metric_df.dropna(subset=[metric])
        if metric_df.empty:
            continue

        for chunk_id in sorted(metric_df["chunk_id"].unique()):
            chunk_slice = metric_df[metric_df["chunk_id"] == chunk_id]
            success_values = chunk_slice.loc[chunk_slice["success"], metric].to_numpy(dtype=float)
            failure_values = chunk_slice.loc[~chunk_slice["success"], metric].to_numpy(dtype=float)

            n_success = int(success_values.size)
            n_failure = int(failure_values.size)

            if n_success < min_samples_per_group or n_failure < min_samples_per_group:
                ks_stat = float("nan")
                p_value = float("nan")
            else:
                ks_stat, p_value = _ks_test(success_values, failure_values)

            significant = bool(np.isfinite(p_value) and p_value < alpha)
            rows.append(
                {
                    "metric": metric,
                    "chunk_id": int(chunk_id),
                    "ks_statistic": ks_stat,
                    "p_value": p_value,
                    "significant_at_alpha": significant,
                    "alpha": alpha,
                    "n_success": n_success,
                    "n_failure": n_failure,
                    "success_mean": float(success_values.mean()) if n_success else float("nan"),
                    "failure_mean": float(failure_values.mean()) if n_failure else float("nan"),
                    "success_median": float(np.median(success_values)) if n_success else float("nan"),
                    "failure_median": float(np.median(failure_values)) if n_failure else float("nan"),
                    "mean_diff_failure_minus_success": (
                        float(failure_values.mean() - success_values.mean())
                        if n_success and n_failure
                        else float("nan")
                    ),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    valid = result["p_value"].notna()
    result.loc[valid, "rank_within_chunk"] = (
        result.loc[valid]
        .groupby("chunk_id")["p_value"]
        .rank(method="dense", ascending=True, na_option="bottom")
    )
    result.loc[valid, "rank_global"] = result.loc[valid, "p_value"].rank(
        method="dense", ascending=True, na_option="bottom"
    )
    result = result.sort_values(["chunk_id", "p_value"], ascending=[True, True], na_position="last")
    return result.reset_index(drop=True)


def save_outputs(table: pd.DataFrame, output_dir: Path, *, alpha: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    main_path = output_dir / "chunk_metric_ks.csv"
    table.to_csv(main_path, index=False, float_format="%.8g")

    significant = table[table["significant_at_alpha"] == True].copy()  # noqa: E712
    significant = significant.sort_values(["chunk_id", "p_value"], ascending=[True, True])

    sig_path = output_dir / "chunk_metric_ks_significant.csv"
    significant.to_csv(sig_path, index=False, float_format="%.8g")

    pivot_p = table.pivot(index="metric", columns="chunk_id", values="p_value")
    pivot_p = pivot_p.sort_index()
    pivot_p.to_csv(output_dir / "chunk_metric_ks_pvalue_pivot.csv", float_format="%.8g")

    pivot_sig = table.pivot(index="metric", columns="chunk_id", values="significant_at_alpha")
    pivot_sig = pivot_sig.sort_index()
    pivot_sig.to_csv(output_dir / "chunk_metric_ks_significant_pivot.csv")

    # Human-readable summary grouped by chunk_id.
    lines = [
        "# Значимые отличия success vs failure (Kolmogorov–Smirnov)",
        "",
        f"Критерий: `p_value < {alpha}` (двусторонний two-sample KS test).",
        f"Всего пар (metric, chunk_id): {len(table)}",
        f"Значимых пар: {len(significant)}",
        "",
    ]
    if significant.empty:
        lines.append("_Значимых отличий не найдено._")
    else:
        for chunk_id in sorted(significant["chunk_id"].unique()):
            chunk_rows = significant[significant["chunk_id"] == chunk_id].sort_values("p_value")
            lines.append(f"## chunk_id = {int(chunk_id)} ({len(chunk_rows)} метрик)")
            lines.append("")
            lines.append("| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |")
            lines.append("|--------|---------|--------------|-----------|-----------|---------------------------|")
            for row in chunk_rows.itertuples(index=False):
                diff = row.mean_diff_failure_minus_success
                diff_str = f"{diff:.6g}" if np.isfinite(diff) else "nan"
                lines.append(
                    f"| `{row.metric}` | {row.p_value:.6g} | {row.ks_statistic:.6g} | "
                    f"{row.n_success} | {row.n_failure} | {diff_str} |"
                )
            lines.append("")

        lines.append("## Сводка по метрикам (на каких chunk значимо)")
        lines.append("")
        metric_chunks = (
            significant.groupby("metric")
            .agg(
                n_significant_chunks=("chunk_id", "count"),
                significant_chunk_ids=("chunk_id", lambda s: ", ".join(str(int(x)) for x in sorted(s))),
            )
            .reset_index()
            .sort_values("n_significant_chunks", ascending=False)
        )
        lines.append("| metric | число значимых chunk | chunk_id |")
        lines.append("|--------|---------------------|----------|")
        for row in metric_chunks.itertuples(index=False):
            lines.append(
                f"| `{row.metric}` | {row.n_significant_chunks} | {row.significant_chunk_ids} |"
            )
        lines.append("")

    summary_md = output_dir / "chunk_metric_ks_significant_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    sig_records = significant[["metric", "chunk_id", "p_value", "ks_statistic"]].to_dict(orient="records")
    summary_json = output_dir / "chunk_metric_ks_significant.json"
    summary_json.write_text(
        json.dumps(
            {
                "alpha": alpha,
                "n_total_pairs": len(table),
                "n_significant_pairs": len(significant),
                "significant_pairs": sig_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    readme = output_dir / "README.txt"
    readme.write_text(
        f"""chunk_metric_ks.csv — полная таблица KS-тестов (metric × chunk_id).
  ks_statistic: статистика Kolmogorov–Smirnov (чем больше, тем сильнее расхождение CDF).
  p_value: p-value двустороннего two-sample KS test (scipy.stats.ks_2samp).
  significant_at_alpha: True если p_value < {alpha}.
  rank_within_chunk / rank_global: ранг по возрастанию p_value (1 = наиболее значимо).

chunk_metric_ks_significant.csv — только значимые пары (p < {alpha}).
chunk_metric_ks_significant_summary.md — удобный markdown-отчёт по chunk и по метрикам.
chunk_metric_ks_significant.json — тот же список значимых пар в JSON.
chunk_metric_ks_pvalue_pivot.csv — pivot: p_value[metric, chunk_id].
chunk_metric_ks_significant_pivot.csv — pivot: True/False значимость.
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "eval_outputs/libero_spatial/baseline/trace_100ep/metrics",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <metrics-dir>/chunk_ks_analysis",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum success and failure samples per (metric, chunk_id)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for p_value",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_dir = args.metrics_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else metrics_dir / "chunk_ks_analysis"

    df = _load_chunk_metrics(metrics_dir)
    table = compute_ks_table(
        df,
        metrics=PLOT_METRIC_COLUMNS,
        min_samples_per_group=args.min_samples,
        alpha=args.alpha,
    )
    save_outputs(table, output_dir, alpha=args.alpha)

    n_sig = int((table["significant_at_alpha"] == True).sum()) if not table.empty else 0  # noqa: E712
    print(f"[done] rows={len(table)} significant(p<{args.alpha})={n_sig}")
    print(f"[saved] {output_dir / 'chunk_metric_ks.csv'}")
    print(f"[saved] {output_dir / 'chunk_metric_ks_significant_summary.md'}")
    if n_sig:
        best = table[table["significant_at_alpha"]].sort_values("p_value").iloc[0]
        print(
            f"[most significant] chunk={int(best['chunk_id'])} metric={best['metric']} "
            f"p={best['p_value']:.6g} ks={best['ks_statistic']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
