#!/usr/bin/env python3
"""Run one INT-ACT Object OOD task against a GR00T policy server."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import gymnasium as gym
import tyro


def make_intact_widowx_env_class():
    """Adapt NVIDIA's WidowX wrapper to INT-ACT's TimeLimit return type."""
    from gr00t.eval.sim.SimplerEnv.simpler_env import WidowXBridgeEnv

    class IntActWidowXBridgeEnv(WidowXBridgeEnv):
        def __init__(self, env_name: str, image_size: tuple[int, int]) -> None:
            super().__init__(env_name=env_name, image_size=image_size)
            # INT-ACT's simpler_env.make returns gymnasium.TimeLimit whereas
            # NVIDIA's adapter expects robot_uid directly on the returned env.
            self.env.robot_uid = self.env.unwrapped.robot_uid
            # Restore the official INT-ACT Object OOD horizon. The parent
            # adapter raises this to 10,000 for stock SIMPLER evaluations.
            self.env._max_episode_steps = 60

    return IntActWidowXBridgeEnv


@dataclass
class Config:
    task_key: str
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

    # The stock GR00T SIMPLER adapter registers only the original Bridge tasks.
    # Register the INT-ACT task key as a WidowX environment while letting the
    # INT-ACT SimplerEnv fork resolve it to the corresponding *-v2 gym env.
    env_id = f"simpler_env_widowx/{config.task_key}"
    gym.register(
        id=env_id,
        entry_point=make_intact_widowx_env_class(),
        kwargs={"env_name": config.task_key, "image_size": (256, 256)},
    )

    from gr00t.eval.rollout_policy import run_gr00t_sim_policy

    env_name, successes, info = run_gr00t_sim_policy(
        env_name=env_id,
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
        "task_key": config.task_key,
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
