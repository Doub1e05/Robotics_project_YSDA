from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import tqdm
import tyro

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "eval" / "libero") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "eval" / "libero"))

from run import (  # noqa: E402
    _termination_reason,
    _write_trace_outputs,
    benchmark,
    get_libero_env,
    get_libero_image,
    save_rollout_video,
    set_seed_everywhere,
    VAMInference,
)


def _load_episode_trace(metrics_path: Path, total_episode_idx: int) -> dict[str, object]:
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if int(row["meta"]["total_episode_idx"]) == int(total_episode_idx):
            return row
    raise ValueError(f"Episode with total_episode_idx={total_episode_idx} not found in {metrics_path}")


def _load_chunk_artifact(episode_trace: dict[str, object], chunk_id: int) -> tuple[dict[str, object], dict[str, object]]:
    for chunk in episode_trace["chunks"]:
        if int(chunk["chunk_id"]) == int(chunk_id):
            payload = torch.load(chunk["decoder_replay_artifact_path"], map_location="cpu", weights_only=False)
            return chunk, payload
    raise ValueError(f"chunk_id={chunk_id} not found")


def _prime_policy_history(policy: VAMInference, env, env_timeline_path: Path, query_timestep: int) -> dict:
    timeline = np.load(env_timeline_path)
    sim_states = timeline["sim_states"]
    start = max(0, int(query_timestep) - (policy._image_history.maxlen - 1))
    for idx in range(start, int(query_timestep) + 1):
        obs = env.regenerate_obs_from_state(sim_states[idx])
        image = get_libero_image(obs)
        policy._add_image_to_history(policy._process_image(image))
        policy._add_lowdim_to_history(policy._state_from_observation(obs))
    obs = env.regenerate_obs_from_state(sim_states[int(query_timestep)])
    return obs


def _decoder_only_chunk(policy: VAMInference, payload: dict[str, object], seed: int) -> np.ndarray:
    hidden_grid = payload["encoder_hidden_grid_B_T_H_W_D"].cuda().to(dtype=torch.bfloat16)
    hidden_shape = hidden_grid.shape
    crossattn_emb = hidden_grid.reshape(hidden_shape[0], -1, hidden_shape[-1])
    state_tensor = payload["state_B_HO_O"].cuda().to(dtype=torch.float32)
    context = payload["context_timesteps_B_1"].cuda().to(dtype=torch.float32)
    pred_actions = policy.model.world2action_pipeline(
        state_B_HO_O=state_tensor,
        crossattn_emb=crossattn_emb,
        context_timesteps_B_1=context,
        seed=seed,
        use_cuda_graphs=False,
    )
    return pred_actions[0].float().cpu().numpy()


def run_replay_episode(
    *,
    env,
    policy: VAMInference,
    task_description: str,
    env_timeline_path: Path,
    chunk_trace: dict[str, object],
    payload: dict[str, object],
    replay_seed: int,
    max_steps: int,
) -> tuple[bool, list[np.ndarray], dict[str, object]]:
    policy.reset(task_description)
    obs = _prime_policy_history(policy, env, env_timeline_path, int(chunk_trace["query_timestep"]))

    first_actions = _decoder_only_chunk(policy, payload, replay_seed)
    policy.action_buffer = first_actions.copy()
    policy.action_buffer_idx = 0
    policy._execute_horizon = policy.num_execute_actions
    policy.last_query_actions = first_actions.copy()
    policy.last_query_execute_horizon = int(policy.num_execute_actions)
    policy.last_query_selected_candidate_idx = 0
    policy.last_query_candidate_probs = []
    policy.last_query_candidate_chunk_metrics = []
    policy.last_query_regen_attempts = 0
    policy.last_query_was_regenerated = False
    policy.last_query_regen_probability = None
    policy.last_query_latency_sec = 0.0
    policy.last_query_representation_metrics = dict(chunk_trace["representation_metrics"])
    policy.last_query_diagnostics = dict(chunk_trace["representation_metrics"])
    policy._chunk_counter = int(chunk_trace["chunk_id"]) - 1
    policy._query_counter = int(chunk_trace["chunk_id"])

    replay_images: list[np.ndarray] = []
    trace: dict[str, object] = {"chunks": [], "action_count": 0}
    success = False
    previous_raw_action: np.ndarray | None = None

    trace["chunks"].append(
        {
            "chunk_id": int(chunk_trace["chunk_id"]),
            "inference_step_idx": 0,
            "query_timestep": int(chunk_trace["query_timestep"]),
            "query_latency_sec": 0.0,
            "regen_probability": None,
            "regen_attempts": 0,
            "was_regenerated": False,
            "regen_strategy": "none",
            "num_candidates": 1,
            "selected_candidate_idx": 0,
            "candidate_regen_probabilities": [],
            "candidate_chunk_metrics": [],
            "execute_horizon": int(policy.num_execute_actions),
            "action_conf_threshold": float(policy.action_conf_threshold),
            "action_success_probabilities": [],
            "representation_metrics": dict(chunk_trace["representation_metrics"]),
            "decoder_replay_artifact_path": "",
            "metrics": {},
            "actions": [],
        }
    )

    for local_step in range(max_steps):
        image = get_libero_image(obs)
        replay_images.append(image)
        action = policy.step(image, task_description, obs)

        if policy.last_step_used_query:
            representation_metrics = (
                dict(chunk_trace["representation_metrics"])
                if int(policy.last_chunk_id or -1) == int(chunk_trace["chunk_id"])
                else dict(policy.last_query_representation_metrics or {})
            )
            trace["chunks"].append(
                {
                    "chunk_id": int(policy.last_chunk_id if policy.last_chunk_id is not None else len(trace["chunks"])),
                    "inference_step_idx": int(len(trace["chunks"])),
                    "query_timestep": int(chunk_trace["query_timestep"]) + local_step,
                    "query_latency_sec": float(policy.last_query_latency_sec or 0.0),
                    "regen_probability": policy.last_query_regen_probability,
                    "regen_attempts": int(policy.last_query_regen_attempts),
                    "was_regenerated": bool(policy.last_query_was_regenerated),
                    "regen_strategy": policy.regen_strategy,
                    "num_candidates": 1,
                    "selected_candidate_idx": 0,
                    "candidate_regen_probabilities": [],
                    "candidate_chunk_metrics": [],
                    "execute_horizon": int(policy.last_query_execute_horizon),
                    "action_conf_threshold": float(policy.action_conf_threshold),
                    "action_success_probabilities": list(policy.last_query_action_success_probs),
                    "representation_metrics": representation_metrics,
                    "decoder_replay_artifact_path": "",
                    "metrics": {},
                    "actions": [],
                }
            )

        action_record = {
            "action_index": int(trace["action_count"]),
            "timestep": int(chunk_trace["query_timestep"]) + local_step,
            "chunk_id": int(policy.last_chunk_id),
            "action_offset_in_chunk": int(policy.last_action_offset_in_chunk),
            "metrics": {},
        }
        trace["chunks"][-1]["actions"].append(action_record)
        trace["action_count"] = int(trace["action_count"]) + 1
        previous_raw_action = policy.last_raw_action.copy() if policy.last_raw_action is not None else previous_raw_action

        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            break

    trace["exception"] = None
    return success, replay_images, trace


def replay_from_saved_chunk(
    source_metrics_path: pathlib.Path,
    source_rank_dir: pathlib.Path,
    out_dir: pathlib.Path,
    total_episode_idx: int,
    chunk_id: int = 8,
    num_replays: int = 10,
    replay_horizon_steps: int = 40,
    replay_seed_base: int = 80000,
    task_suite_name: str = "libero_spatial",
    task_id: int = 0,
    episode_idx: int = 9,
    gpu_label: str = "",
    vam_experiment_name: str = "w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128",
    vam_video_model_path: pathlib.Path = REPO_ROOT / "model/checkpoints/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt",
    vam_action_model_path: pathlib.Path = REPO_ROOT / "model/checkpoints/action_decoder/w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000019998.pt",
    vam_dataset_statistics_path: pathlib.Path = REPO_ROOT / "model/checkpoints/dataset_statistics/libero_spatial_one.json",
    t5_embeddings_path: pathlib.Path | None = None,
) -> None:
    out_dir = Path(out_dir)
    videos_dir = out_dir / "videos"
    metrics_dir = out_dir / "metrics"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    episode_trace = _load_episode_trace(Path(source_metrics_path), total_episode_idx)
    chunk_trace, payload = _load_chunk_artifact(episode_trace, chunk_id)
    env_timeline_path = Path(episode_trace["meta"]["env_timeline_path"])
    task_description = str(episode_trace["meta"]["task_description"])

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    env, _ = get_libero_env(task)

    policy = VAMInference(
        vam_experiment_name,
        str(vam_video_model_path),
        str(vam_action_model_path),
        vam_dataset_statistics_path,
        img_horizon=5,
        lowdim_horizon=1,
        stop_video_denoising_step=0,
        num_execute_actions=5,
        rollout_dir=out_dir,
        t5_embeddings_path=t5_embeddings_path,
        use_cuda_graphs=False,
        regen_strategy="none",
        seed=replay_seed_base,
        diagnostics_mode="encoder_hidden",
        representation_layer_indices=[4, 8, 12, 16, 20, 24, 28],
    )

    episode_traces: list[dict[str, object]] = []
    try:
        for replay_idx in tqdm.tqdm(range(num_replays), desc=f"Replay ep{total_episode_idx}"):
            replay_seed = replay_seed_base + replay_idx
            set_seed_everywhere(replay_seed)
            env.reset()
            env.set_init_state(task_suite.get_task_init_states(task_id)[episode_idx])
            obs = env.regenerate_obs_from_state(payload["sim_state_before_query"].numpy())

            success, replay_images, trace = run_replay_episode(
                env=env,
                policy=policy,
                task_description=task_description,
                env_timeline_path=env_timeline_path,
                chunk_trace=chunk_trace,
                payload=payload,
                replay_seed=replay_seed,
                max_steps=replay_horizon_steps,
            )

            video_idx = replay_idx + 1
            save_rollout_video(
                replay_images,
                video_idx,
                success,
                f"{task_description}_from_ep{total_episode_idx}_chunk{chunk_id}",
                videos_dir,
            )
            episode_traces.append(
                {
                    "meta": {
                        "total_episode_idx": int(video_idx),
                        "source_total_episode_idx": int(total_episode_idx),
                        "source_chunk_id": int(chunk_id),
                        "replay_seed": int(replay_seed),
                        "gpu_label": gpu_label,
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "task_description": task_description,
                        "success": bool(success),
                        "termination_reason": _termination_reason(success, trace["exception"], int(trace["action_count"]), replay_horizon_steps),
                        "exception": "",
                        "step_count": int(trace["action_count"]),
                        "max_steps": int(replay_horizon_steps),
                        "num_steps_wait": 0,
                        "inference_step_count": int(len(trace["chunks"])),
                        "chunk_count": int(len(trace["chunks"])),
                        "action_count": int(trace["action_count"]),
                    },
                    "chunks": trace["chunks"],
                }
            )
            _write_trace_outputs(
                metrics_dir,
                episode_traces,
                write_chunk_metrics=False,
                write_candidate_metrics=False,
                write_representation_metrics=True,
                write_action_metrics=False,
            )
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(replay_from_saved_chunk)
