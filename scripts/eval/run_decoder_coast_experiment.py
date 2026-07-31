"""Run a COAST-style decoder steering experiment on one fixed LIBERO episode."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import tyro

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts" / "eval") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

from run_boundary_multi_seed import run_boundary_multi_seed  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_decoder_samples(
    traces_path: Path,
    *,
    decoder_block_idx: int,
    success: bool,
    max_query_timestep: int,
    action_tokens_only: bool,
) -> torch.Tensor:
    rows = _read_jsonl(traces_path)
    collected: list[torch.Tensor] = []
    for episode in rows:
        meta = episode["meta"]
        if bool(meta["success"]) != bool(success):
            continue
        for chunk in episode["chunks"]:
            if int(chunk["query_timestep"]) >= int(max_query_timestep):
                continue
            artifact_path = Path(chunk["decoder_replay_artifact_path"])
            if not artifact_path.is_file():
                continue
            payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
            hidden_by_block = payload.get("decoder_hidden_states_by_block", {})
            if int(decoder_block_idx) not in hidden_by_block:
                continue
            hidden = hidden_by_block[int(decoder_block_idx)].float()
            if hidden.ndim != 3:
                raise ValueError(f"Expected decoder hidden state [B, T, D], got {tuple(hidden.shape)}")
            hidden = hidden[0]
            if action_tokens_only:
                obs_token_count = int(payload["state_B_HO_O"].shape[1])
                hidden = hidden[obs_token_count:]
            collected.append(hidden)
    if not collected:
        label = "success" if success else "failure"
        raise ValueError(f"No decoder samples collected for {label=}, block={decoder_block_idx}")
    return torch.cat(collected, dim=0)


def _sample_equal_rows(
    success_samples_N_D: torch.Tensor,
    failure_samples_N_D: torch.Tensor,
    *,
    seed: int,
    max_samples_per_class: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    target = min(int(success_samples_N_D.shape[0]), int(failure_samples_N_D.shape[0]))
    if max_samples_per_class is not None:
        target = min(target, int(max_samples_per_class))
    if target < 2:
        raise ValueError(f"Need at least 2 samples per class, got target={target}")

    def _pick(x_N_D: torch.Tensor) -> torch.Tensor:
        idx = list(range(int(x_N_D.shape[0])))
        rng.shuffle(idx)
        keep = torch.tensor(idx[:target], dtype=torch.long)
        return x_N_D.index_select(0, keep)

    return _pick(success_samples_N_D), _pick(failure_samples_N_D)


def _correlation(x_N_D: torch.Tensor) -> torch.Tensor:
    return (x_N_D.T @ x_N_D) / float(max(int(x_N_D.shape[0]), 1))


def _conceptor_from_samples(x_N_D: torch.Tensor, aperture: float) -> torch.Tensor:
    d = int(x_N_D.shape[1])
    r_D_D = _correlation(x_N_D)
    eye_D_D = torch.eye(d, dtype=r_D_D.dtype)
    reg = (1.0 / (float(aperture) ** 2)) * eye_D_D
    return r_D_D @ torch.linalg.inv(r_D_D + reg)


def _relative_delta_scale(
    steer_matrix_D_D: torch.Tensor,
    samples_N_D: torch.Tensor,
    *,
    target_relative_shift: float,
) -> float:
    delta_N_D = samples_N_D @ steer_matrix_D_D
    delta_norm = delta_N_D.norm(dim=1)
    base_norm = samples_N_D.norm(dim=1).clamp_min(1e-6)
    raw_ratio = float(torch.median(delta_norm / base_norm).item())
    if raw_ratio <= 1e-8:
        return 1.0
    return float(target_relative_shift / raw_ratio)


def build_contrastive_conceptor(
    baseline_metrics_dir: Path,
    out_path: Path,
    *,
    decoder_block_idx: int,
    aperture: float,
    max_query_timestep: int,
    action_tokens_only: bool,
    sampling_seed: int,
    max_samples_per_class: int | None,
    target_relative_shift: float,
) -> dict[str, object]:
    traces_path = baseline_metrics_dir / "episode_traces.jsonl"
    success_samples = _load_decoder_samples(
        traces_path,
        decoder_block_idx=decoder_block_idx,
        success=True,
        max_query_timestep=max_query_timestep,
        action_tokens_only=action_tokens_only,
    )
    failure_samples = _load_decoder_samples(
        traces_path,
        decoder_block_idx=decoder_block_idx,
        success=False,
        max_query_timestep=max_query_timestep,
        action_tokens_only=action_tokens_only,
    )
    success_samples, failure_samples = _sample_equal_rows(
        success_samples,
        failure_samples,
        seed=sampling_seed,
        max_samples_per_class=max_samples_per_class,
    )
    c_success = _conceptor_from_samples(success_samples, aperture=aperture)
    c_failure = _conceptor_from_samples(failure_samples, aperture=aperture)
    steer_matrix = c_success - c_failure
    strength = _relative_delta_scale(
        steer_matrix,
        torch.cat((success_samples, failure_samples), dim=0),
        target_relative_shift=target_relative_shift,
    )
    payload = {
        "conceptor_success_D_D": c_success.cpu(),
        "conceptor_failure_D_D": c_failure.cpu(),
        "steer_matrix_D_D": steer_matrix.cpu(),
        "target_block_indices": [int(decoder_block_idx)],
        "metadata": {
            "decoder_block_idx": int(decoder_block_idx),
            "aperture": float(aperture),
            "action_tokens_only": bool(action_tokens_only),
            "sampling_seed": int(sampling_seed),
            "num_samples_per_class": int(success_samples.shape[0]),
            "target_relative_shift": float(target_relative_shift),
            "derived_strength": float(strength),
            "success_quota": float(torch.trace(c_success).item() / c_success.shape[0]),
            "failure_quota": float(torch.trace(c_failure).item() / c_failure.shape[0]),
            "steer_fro_norm": float(torch.linalg.norm(steer_matrix).item()),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return payload["metadata"]


def _load_summary(metrics_dir: Path) -> dict[str, object]:
    return json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))


def run_decoder_coast_experiment(
    task_id: int = 0,
    episode_idx: int = 9,
    task_suite_name: str = "libero_spatial",
    num_rollouts: int = 30,
    timeout_seconds: float = 6.0,
    rollout_root: Path = REPO_ROOT / "eval_outputs/libero_spatial/decoder_coast/task0_init9_30rollouts_6sec",
    seed_base: int = 61000,
    seed_stride: int = 97,
    decoder_block_idx: int = 20,
    aperture: float = 8.0,
    max_samples_per_class: int | None = 12000,
    target_relative_shift: float = 0.15,
    diagnostics_mode: str = "encoder_hidden",
    representation_layer_indices: list[int] = [4, 8, 12, 16, 20, 24, 28],
    t5_embeddings_path: Path = Path(
        "/home/motovilovil/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8/libero_t5_embeddings.pkl"
    ),
) -> None:
    max_control_steps = int(round(timeout_seconds * 20.0))
    baseline_dir = rollout_root / "baseline"
    steered_dir = rollout_root / "steered"
    conceptor_path = rollout_root / "artifacts" / f"decoder_block{decoder_block_idx}_conceptor.pt"

    run_boundary_multi_seed(
        task_id=task_id,
        episode_idx=episode_idx,
        task_suite_name=task_suite_name,
        num_rollouts=num_rollouts,
        seed_base=seed_base,
        seed_stride=seed_stride,
        max_control_steps=max_control_steps,
        save_videos=False,
        rollout_dir=baseline_dir,
        metrics_dir=baseline_dir / "metrics",
        diagnostics_mode=diagnostics_mode,
        representation_only=True,
        representation_layer_indices=representation_layer_indices,
        decoder_capture_block_indices=[decoder_block_idx],
        use_cuda_graphs=False,
        t5_embeddings_path=t5_embeddings_path,
        replay_payload_mode="decoder_only",
        save_env_timeline=False,
        save_representation_metrics=False,
        progress_label="Baseline rollouts",
    )

    conceptor_meta = build_contrastive_conceptor(
        baseline_dir / "metrics",
        conceptor_path,
        decoder_block_idx=decoder_block_idx,
        aperture=aperture,
        max_query_timestep=max_control_steps,
        action_tokens_only=True,
        sampling_seed=seed_base,
        max_samples_per_class=max_samples_per_class,
        target_relative_shift=target_relative_shift,
    )

    run_boundary_multi_seed(
        task_id=task_id,
        episode_idx=episode_idx,
        task_suite_name=task_suite_name,
        num_rollouts=num_rollouts,
        seed_base=seed_base,
        seed_stride=seed_stride,
        max_control_steps=max_control_steps,
        save_videos=False,
        rollout_dir=steered_dir,
        metrics_dir=steered_dir / "metrics",
        diagnostics_mode=diagnostics_mode,
        representation_only=True,
        representation_layer_indices=representation_layer_indices,
        decoder_capture_block_indices=[],
        decoder_steering_path=conceptor_path,
        decoder_steering_strength=float(conceptor_meta["derived_strength"]),
        decoder_steering_block_indices=[decoder_block_idx],
        use_cuda_graphs=False,
        t5_embeddings_path=t5_embeddings_path,
        replay_payload_mode="none",
        save_env_timeline=False,
        save_representation_metrics=False,
        progress_label="COAST steered rollouts",
    )

    comparison = {
        "task_suite_name": task_suite_name,
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "num_rollouts": int(num_rollouts),
        "timeout_seconds": float(timeout_seconds),
        "max_control_steps": int(max_control_steps),
        "decoder_block_idx": int(decoder_block_idx),
        "conceptor_path": str(conceptor_path),
        "conceptor": conceptor_meta,
        "baseline_summary": _load_summary(baseline_dir / "metrics"),
        "steered_summary": _load_summary(steered_dir / "metrics"),
    }
    comparison["success_rate_delta"] = (
        float(comparison["steered_summary"]["success_rate"]) - float(comparison["baseline_summary"]["success_rate"])
    )
    out_path = rollout_root / "comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    tyro.cli(run_decoder_coast_experiment)
