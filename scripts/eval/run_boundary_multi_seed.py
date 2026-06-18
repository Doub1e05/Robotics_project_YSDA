"""Run multiple stochastic rollouts on one fixed LIBERO task/init_state."""

from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path

import tqdm
import tyro

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "eval" / "libero") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "eval" / "libero"))

from run import (  # noqa: E402
    LIBERO_SUITE_MAX_STEPS,
    _termination_reason,
    _trim_failure_episode_metrics,
    _write_trace_outputs,
    benchmark,
    get_libero_env,
    run_episode,
    save_rollout_video,
    set_seed_everywhere,
    VAMInference,
)


def assigned_rollout_ids(num_rollouts: int, eval_rank: int, eval_world_size: int) -> list[int]:
    return [rollout_id for rollout_id in range(num_rollouts) if rollout_id % eval_world_size == eval_rank]


def episode_video_path(videos_dir: Path, rollout_id: int) -> Path | None:
    matches = sorted(videos_dir.glob(f"episode{rollout_id + 1}_*.mp4"))
    return matches[0] if matches else None


def run_boundary_multi_seed(
    task_id: int = 0,
    episode_idx: int = 9,
    task_suite_name: str = "libero_spatial",
    num_rollouts: int = 40,
    seed_base: int = 50_000,
    seed_stride: int = 97,
    max_control_steps: int = 220,
    num_steps_wait: int = 10,
    save_videos: bool = True,
    max_videos: int = 0,
    video_catchup: bool = False,
    rollout_dir: pathlib.Path = REPO_ROOT / "eval_outputs/libero_spatial/boundary_runs/task0_init9_baseline",
    metrics_dir: pathlib.Path | None = None,
    vam_experiment_name: str = "w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128",
    vam_video_model_path: str = str(
        REPO_ROOT / "model/checkpoints/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt"
    ),
    vam_action_model_path: pathlib.Path = REPO_ROOT
    / "model/checkpoints/action_decoder/w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000019998.pt",
    vam_dataset_statistics_path: pathlib.Path = REPO_ROOT / "model/checkpoints/dataset_statistics/libero_spatial_one.json",
    vam_img_horizon: int = 5,
    vam_lowdim_horizon: int = 1,
    vam_stop_video_denoising_step: int = 0,
    vam_num_execute_actions: int = 5,
    t5_embeddings_path: pathlib.Path | None = None,
    use_cuda_graphs: bool = False,
    regen_strategy: str = "none",
    regen_model_path: pathlib.Path | None = None,
    regen_num_candidates: int = 3,
    eval_rank: int = 0,
    eval_world_size: int = 1,
    resume: bool = True,
    diagnostics_mode: str = "default",
) -> None:
    rollout_dir = Path(rollout_dir)
    metrics_root = Path(metrics_dir or rollout_dir / "metrics")
    if eval_world_size > 1:
        metrics_dir = metrics_root / f"rank{eval_rank}"
        rollout_dir = rollout_dir / f"rank{eval_rank}"
    else:
        metrics_dir = metrics_root
    videos_dir = rollout_dir / "videos"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "task_suite_name": task_suite_name,
        "task_id": task_id,
        "episode_idx": episode_idx,
        "num_rollouts": num_rollouts,
        "seed_base": seed_base,
        "seed_stride": seed_stride,
        "max_control_steps": max_control_steps,
        "regen_strategy": regen_strategy,
        "eval_rank": eval_rank,
        "eval_world_size": eval_world_size,
        "video_catchup": video_catchup,
        "diagnostics_mode": diagnostics_mode,
    }
    (metrics_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    episode_traces: list[dict[str, object]] = []
    completed_seeds: set[int] = set()
    traces_path = metrics_dir / "episode_traces.jsonl"
    if resume and traces_path.is_file():
        for line in traces_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ep = json.loads(line)
            episode_traces.append(ep)
            completed_seeds.add(int(ep["meta"]["rollout_seed"]))

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    if episode_idx >= len(initial_states):
        raise ValueError(f"episode_idx={episode_idx} out of range for task {task_id} ({len(initial_states)} inits).")

    env, task_description = get_libero_env(task)
    max_steps = min(max_control_steps, LIBERO_SUITE_MAX_STEPS[task_suite_name])

    policy = VAMInference(
        vam_experiment_name,
        vam_video_model_path,
        str(vam_action_model_path),
        vam_dataset_statistics_path,
        vam_img_horizon,
        vam_lowdim_horizon,
        vam_stop_video_denoising_step,
        vam_num_execute_actions,
        rollout_dir,
        t5_embeddings_path,
        use_cuda_graphs,
        regen_model_path,
        0.38,
        0,
        regen_strategy,
        regen_num_candidates,
        None,
        0.62,
        3,
        8,
        seed_base,
        diagnostics_mode,
    )

    rollout_ids = assigned_rollout_ids(num_rollouts, eval_rank, eval_world_size)
    videos_saved = sum(1 for rollout_id in rollout_ids if episode_video_path(videos_dir, rollout_id))
    metrics_pending = [
        rollout_id
        for rollout_id in rollout_ids
        if seed_base + seed_stride * rollout_id not in completed_seeds
    ]
    videos_pending = [
        rollout_id
        for rollout_id in rollout_ids
        if episode_video_path(videos_dir, rollout_id) is None
    ]
    if video_catchup:
        pending = videos_pending
        print(
            f"[rank {eval_rank}] video catchup: {len(videos_pending)} missing videos "
            f"(metrics already complete for this rank)"
        )
    else:
        pending = metrics_pending
        print(
            f"[rank {eval_rank}] metrics pending: {len(metrics_pending)} | "
            f"videos saved: {videos_saved}/{len(rollout_ids)}"
        )

    try:
        for rollout_id in tqdm.tqdm(pending, desc=f"Rollouts rank{eval_rank}"):
            rollout_seed = seed_base + seed_stride * rollout_id
            metrics_done = rollout_seed in completed_seeds
            if metrics_done and not video_catchup:
                continue

            set_seed_everywhere(rollout_seed)
            policy.seed = rollout_seed
            policy.reset(task_description)

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            success, replay_images, trace = run_episode(
                env,
                policy,
                task_description,
                obs,
                max_steps,
                num_steps_wait,
            )

            video_idx = rollout_id + 1
            should_save_video = save_videos and episode_video_path(videos_dir, rollout_id) is None
            if max_videos > 0:
                should_save_video = should_save_video and videos_saved < max_videos
            if should_save_video:
                save_rollout_video(
                    replay_images,
                    video_idx,
                    success,
                    task_description,
                    videos_dir,
                )
                videos_saved += 1

            if metrics_done:
                continue

            episode_step_count = int(trace["action_count"])
            episode_trace = {
                "meta": {
                    "rollout_id": rollout_id,
                    "rollout_seed": rollout_seed,
                    "task_id": int(task_id),
                    "episode_idx": int(episode_idx),
                    "total_episode_idx": int(video_idx),
                    "task_description": task_description,
                    "success": bool(success),
                    "termination_reason": _termination_reason(
                        success, trace["exception"], episode_step_count, max_steps
                    ),
                    "exception": trace["exception"] or "",
                    "step_count": episode_step_count,
                    "max_steps": int(max_steps),
                    "num_steps_wait": int(num_steps_wait),
                    "inference_step_count": int(len(trace["chunks"])),
                    "chunk_count": int(len(trace["chunks"])),
                    "action_count": int(trace["action_count"]),
                    "regen_strategy": str(regen_strategy),
                    "max_control_steps": int(max_steps),
                },
                "chunks": trace["chunks"],
            }
            episode_traces.append(_trim_failure_episode_metrics(episode_trace))
            completed_seeds.add(rollout_seed)
            _write_trace_outputs(metrics_dir, episode_traces)
    finally:
        env.close()

    successes = sum(bool(ep["meta"]["success"]) for ep in episode_traces)
    print(
        f"[rank {eval_rank}/{eval_world_size}] traces={len(episode_traces)} | "
        f"videos={videos_saved}/{len(rollout_ids)} | successes={successes} | "
        f"SR={successes / max(len(episode_traces), 1):.3f}"
    )
    print(f"Metrics: {metrics_dir}")


if __name__ == "__main__":
    tyro.cli(run_boundary_multi_seed)
