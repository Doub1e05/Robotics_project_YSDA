"""Consensus-medoid scoring for stochastic action-chunk candidates."""

from __future__ import annotations

import math

import numpy as np


def _rotation_matrix_from_6d(orientation_6d: np.ndarray) -> np.ndarray:
    """Convert the model's 6D rotation representation to a rotation matrix."""
    orientation_6d = np.asarray(orientation_6d, dtype=np.float64)
    if orientation_6d.shape != (6,):
        raise ValueError(f"Expected a 6D rotation vector, got shape {orientation_6d.shape}.")

    first = orientation_6d[:3]
    second = orientation_6d[3:]
    first = first / max(float(np.linalg.norm(first)), 1e-9)
    second = second - float(np.dot(second, first)) * first
    second = second / max(float(np.linalg.norm(second)), 1e-9)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=0)


def _rotation_geodesic_distance(first_6d: np.ndarray, second_6d: np.ndarray) -> float:
    """Return the geodesic distance between two 6D rotations, normalized to [0, 1]."""
    first_matrix = _rotation_matrix_from_6d(first_6d)
    second_matrix = _rotation_matrix_from_6d(second_6d)
    relative = first_matrix @ second_matrix.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine) / math.pi)


def action_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    translation_weight: float,
    rotation_weight: float,
    gripper_weight: float,
) -> float:
    """Distance between two raw 10D model actions in control-relevant units."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (10,) or second.shape != (10,):
        raise ValueError(f"Expected two 10D actions, got {first.shape} and {second.shape}.")

    translation = float(np.linalg.norm(first[:3] - second[:3]) / math.sqrt(3.0))
    rotation = _rotation_geodesic_distance(first[3:9], second[3:9])
    gripper = float(np.sign(first[9]) != np.sign(second[9]))
    return (
        translation_weight * translation
        + rotation_weight * rotation
        + gripper_weight * gripper
    )


def trajectory_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    horizon: int,
    temporal_discount: float,
    translation_weight: float,
    rotation_weight: float,
    gripper_weight: float,
) -> float:
    """Discounted mean action distance over the prefix that will be executed."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1:] != (10,) or second.shape[1:] != (10,):
        raise ValueError(f"Expected action chunks shaped (T, 10), got {first.shape} and {second.shape}.")

    prefix_length = min(int(horizon), int(first.shape[0]), int(second.shape[0]))
    if prefix_length < 1:
        raise ValueError("Consensus-medoid requires at least one action per candidate.")
    if not 0.0 < temporal_discount <= 1.0:
        raise ValueError("temporal_discount must be in (0, 1].")

    weights = np.power(float(temporal_discount), np.arange(prefix_length, dtype=np.float64))
    distances = np.asarray(
        [
            action_distance(
                first[timestep],
                second[timestep],
                translation_weight=translation_weight,
                rotation_weight=rotation_weight,
                gripper_weight=gripper_weight,
            )
            for timestep in range(prefix_length)
        ],
        dtype=np.float64,
    )
    return float(np.average(distances, weights=weights))


def _smoothness_cost(
    actions: np.ndarray,
    *,
    horizon: int,
    temporal_discount: float,
    translation_weight: float,
    rotation_weight: float,
    gripper_weight: float,
) -> float:
    prefix_length = min(int(horizon), int(actions.shape[0]))
    if prefix_length < 2:
        return 0.0
    return trajectory_distance(
        actions[: prefix_length - 1],
        actions[1:prefix_length],
        horizon=prefix_length - 1,
        temporal_discount=temporal_discount,
        translation_weight=translation_weight,
        rotation_weight=rotation_weight,
        gripper_weight=gripper_weight,
    )


def consensus_medoid_costs(
    candidates: list[np.ndarray],
    *,
    previous_action: np.ndarray | None,
    horizon: int,
    temporal_discount: float = 0.9,
    translation_weight: float = 1.0,
    rotation_weight: float = 0.5,
    gripper_weight: float = 0.25,
    continuity_weight: float = 0.25,
    smoothness_weight: float = 0.10,
    gripper_switch_weight: float = 0.10,
) -> tuple[list[float], list[dict[str, object]], list[list[float]]]:
    """Score candidates by consensus plus conservative trajectory regularizers.

    Lower cost is better. The consensus term is the mean distance from one
    candidate to all other candidates. Remaining terms break close medoid ties
    in favor of continuity with the previous command and a smooth executable
    prefix.
    """
    if len(candidates) < 2:
        raise ValueError("Consensus-medoid requires at least two candidates.")
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    for name, value in {
        "translation_weight": translation_weight,
        "rotation_weight": rotation_weight,
        "gripper_weight": gripper_weight,
        "continuity_weight": continuity_weight,
        "smoothness_weight": smoothness_weight,
        "gripper_switch_weight": gripper_switch_weight,
    }.items():
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative.")

    chunks = [np.asarray(candidate, dtype=np.float64) for candidate in candidates]
    pairwise = np.zeros((len(chunks), len(chunks)), dtype=np.float64)
    for first_idx in range(len(chunks)):
        for second_idx in range(first_idx + 1, len(chunks)):
            distance = trajectory_distance(
                chunks[first_idx],
                chunks[second_idx],
                horizon=horizon,
                temporal_discount=temporal_discount,
                translation_weight=translation_weight,
                rotation_weight=rotation_weight,
                gripper_weight=gripper_weight,
            )
            pairwise[first_idx, second_idx] = distance
            pairwise[second_idx, first_idx] = distance

    costs: list[float] = []
    details: list[dict[str, object]] = []
    for candidate_idx, actions in enumerate(chunks):
        other_distances = np.delete(pairwise[candidate_idx], candidate_idx)
        consensus_cost = float(other_distances.mean())
        continuity_cost = (
            action_distance(
                np.asarray(previous_action, dtype=np.float64),
                actions[0],
                translation_weight=translation_weight,
                rotation_weight=rotation_weight,
                gripper_weight=gripper_weight,
            )
            if previous_action is not None
            else 0.0
        )
        smoothness_cost = _smoothness_cost(
            actions,
            horizon=horizon,
            temporal_discount=temporal_discount,
            translation_weight=translation_weight,
            rotation_weight=rotation_weight,
            gripper_weight=gripper_weight,
        )
        prefix_length = min(int(horizon), int(actions.shape[0]))
        gripper_signs = np.sign(actions[:prefix_length, 9])
        gripper_switches = int(np.sum(gripper_signs[1:] != gripper_signs[:-1]))

        total_cost = float(
            consensus_cost
            + continuity_weight * continuity_cost
            + smoothness_weight * smoothness_cost
            + gripper_switch_weight * gripper_switches
        )
        costs.append(total_cost)
        details.append(
            {
                "consensus_medoid_cost": total_cost,
                "consensus_mean_pairwise_distance": consensus_cost,
                "consensus_continuity_distance": float(continuity_cost),
                "consensus_smoothness_distance": float(smoothness_cost),
                "consensus_gripper_switches": gripper_switches,
                "consensus_pairwise_distances": pairwise[candidate_idx].tolist(),
                "consensus_horizon": int(prefix_length),
            }
        )

    return costs, details, pairwise.tolist()
