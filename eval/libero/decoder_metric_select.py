"""Decoder hidden-state metrics for chunk candidate selection."""

from __future__ import annotations

import math

import numpy as np

DECODER_METRIC_SELECT_LAYER = 19
DECODER_METRIC_SELECT_NAME = "decoder_delta_norm_mean"
DECODER_METRIC_SELECT_SUBSET = "full_chunk"
DECODER_METRIC_SELECT_REDUCE = "max"
DECODER_LINEAR_COMBO_V1_NAME = "decoder_linear_combo_v1"
DECODER_RANK_COMBO_V1_NAME = "decoder_rank_combo_v1"


def decoder_metric_selection_score(
    diagnostics: dict[str, object] | None,
    *,
    layer_idx: int = DECODER_METRIC_SELECT_LAYER,
    metric_name: str = DECODER_METRIC_SELECT_NAME,
    action_subset: str = DECODER_METRIC_SELECT_SUBSET,
) -> float:
    if not diagnostics:
        raise ValueError("Missing diagnostics for decoder metric selection.")
    layer_metrics = diagnostics.get("layer_metrics")
    if not isinstance(layer_metrics, dict):
        raise ValueError("Decoder diagnostics are missing 'layer_metrics'.")
    metrics = layer_metrics.get(layer_idx, layer_metrics.get(str(layer_idx)))
    if not isinstance(metrics, dict):
        raise ValueError(f"Decoder diagnostics do not contain layer {layer_idx}.")
    subset_metrics = metrics.get(action_subset)
    if not isinstance(subset_metrics, dict):
        raise ValueError(
            f"Decoder diagnostics for layer {layer_idx} do not contain action subset {action_subset!r}."
        )
    metric_value = subset_metrics.get(metric_name)
    if metric_value is None:
        raise ValueError(
            f"Decoder diagnostics for layer {layer_idx} / {action_subset} do not contain {metric_name!r}."
        )
    score = float(metric_value)
    if not math.isfinite(score):
        raise ValueError(
            f"Decoder diagnostics for layer {layer_idx} / {action_subset} returned non-finite "
            f"{metric_name}={metric_value!r}."
        )
    return score


def zscore_list(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    if not math.isfinite(std) or std < 1e-8:
        return [0.0 for _ in values]
    return [float((value - mean) / std) for value in arr]


def decoder_linear_combo_v1_scores(
    candidate_diagnostics: list[dict[str, object] | None],
    *,
    layer_idx: int,
    action_subset: str,
) -> tuple[list[float], list[dict[str, float]]]:
    metric_names = {
        "decoder_norm_std": "ns",
        "decoder_adjacent_cosine_mean": "ac",
        "decoder_delta_norm_mean": "dn",
    }
    raw_values: dict[str, list[float]] = {alias: [] for alias in metric_names.values()}
    for diagnostics in candidate_diagnostics:
        for metric_name, alias in metric_names.items():
            raw_values[alias].append(
                decoder_metric_selection_score(
                    diagnostics,
                    layer_idx=layer_idx,
                    metric_name=metric_name,
                    action_subset=action_subset,
                )
            )

    z_ns = zscore_list(raw_values["ns"])
    z_ac = zscore_list(raw_values["ac"])
    z_dn = zscore_list(raw_values["dn"])

    scores: list[float] = []
    details: list[dict[str, float]] = []
    for idx in range(len(candidate_diagnostics)):
        score = float(z_ns[idx] - 0.5 * z_ac[idx] + 0.5 * z_dn[idx])
        scores.append(score)
        details.append(
            {
                "decoder_norm_std_raw": float(raw_values["ns"][idx]),
                "decoder_adjacent_cosine_mean_raw": float(raw_values["ac"][idx]),
                "decoder_delta_norm_mean_raw": float(raw_values["dn"][idx]),
                "decoder_norm_std_z": float(z_ns[idx]),
                "decoder_adjacent_cosine_mean_z": float(z_ac[idx]),
                "decoder_delta_norm_mean_z": float(z_dn[idx]),
                "decoder_linear_combo_v1_score": score,
            }
        )
    return scores, details
