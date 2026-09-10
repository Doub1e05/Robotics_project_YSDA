#!/usr/bin/env python3
"""Run one deterministic GR00T SIMPLER task through a policy server."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import tyro


@dataclass
class Config:
    env_name: str
    host: str
    port: int
    output: Path
    episodes: int = 24
    environment_seed: int = 0
    max_episode_steps: int = 300
    execution_horizon: int = 4
    video_dir: Path | None = None


def main(config: Config) -> None:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "external" / "Isaac-GR00T"))
    from gr00t.eval.rollout_policy import run_gr00t_sim_policy

    env_name, successes, info = run_gr00t_sim_policy(
        env_name=config.env_name,
        n_episodes=config.episodes,
        max_episode_steps=config.max_episode_steps,
        policy_client_host=config.host,
        policy_client_port=config.port,
        n_envs=1,
        n_action_steps=config.execution_horizon,
        video_dir=str(config.video_dir) if config.video_dir else None,
        seed=config.environment_seed,
    )
    result = {
        "env_name": env_name,
        "episodes": len(successes),
        "successes": int(sum(bool(value) for value in successes)),
        "success_rate": float(sum(bool(value) for value in successes) / len(successes)),
        "environment_seed_start": config.environment_seed,
        "execution_horizon": config.execution_horizon,
        "episode_successes": [bool(value) for value in successes],
        "episode_lengths": [int(value) for value in info.get("episode_lengths", [])],
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
