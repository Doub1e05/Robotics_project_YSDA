#!/usr/bin/env python3
"""Evaluate one standard LIBERO Spatial task through a GR00T policy server."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import tyro


@dataclass
class Config:
    task_index: int
    host: str
    port: int
    output: Path
    episodes: int = 10
    environment_seed: int = 0
    max_episode_steps: int = 220
    execution_horizon: int = 8


def main(config: Config) -> None:
    repo = Path(__file__).resolve().parents[2]
    gr00t_root = repo / "external" / "Isaac-GR00T"
    sys.path.insert(0, str(gr00t_root))

    from libero.libero.benchmark import get_benchmark_dict
    from gr00t.eval.rollout_policy import run_gr00t_sim_policy

    suite = get_benchmark_dict()["libero_spatial"]()
    if not 0 <= config.task_index < suite.get_num_tasks():
        raise IndexError(config.task_index)
    task = suite.get_task(config.task_index)
    env_id = f"libero_sim/{task.name}"
    env_name, successes, info = run_gr00t_sim_policy(
        env_name=env_id,
        n_episodes=config.episodes,
        max_episode_steps=config.max_episode_steps,
        policy_client_host=config.host,
        policy_client_port=config.port,
        n_envs=1,
        n_action_steps=config.execution_horizon,
        video_dir=None,
        seed=config.environment_seed,
    )
    result = {
        "benchmark": "LIBERO Spatial",
        "task_index": config.task_index,
        "task_name": task.name,
        "task_description": task.language,
        "env_name": env_name,
        "episodes": len(successes),
        "successes": int(sum(bool(value) for value in successes)),
        "success_rate": float(sum(bool(value) for value in successes) / len(successes)),
        "environment_seed": config.environment_seed,
        "execution_horizon": config.execution_horizon,
        "episode_successes": [bool(value) for value in successes],
        "episode_lengths": [int(value) for value in info.get("episode_lengths", [])],
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
