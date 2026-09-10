#!/usr/bin/env python3
"""Evaluate one LIBERO-PRO Spatial task against a remote GR00T policy server."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys

import gymnasium as gym
import numpy as np
import torch
import tyro


SPLITS = ("lan", "object", "swap", "task")


def read_language(bddl_path: Path) -> str:
    text = bddl_path.read_text(encoding="utf-8")
    match = re.search(r"^\s*\(:language\s+(.+?)\)\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"No :language field in {bddl_path}")
    return match.group(1).strip()


def make_libero_pro_env_class():
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv

    class LiberoProEnv(LiberoEnv):
        """LIBERO adapter that deterministically walks the official PRO init states."""

        def __init__(
            self,
            task_bddl_file: str,
            task_description: str,
            init_states_file: str,
        ) -> None:
            super().__init__(task_bddl_file, task_description)
            states = torch.load(init_states_file, map_location="cpu", weights_only=False)
            if isinstance(states, torch.Tensor):
                states = states.numpy()
            self._init_states = np.asarray(states)
            if len(self._init_states) == 0:
                raise ValueError(f"No init states in {init_states_file}")
            self._next_init_state = 0

        def reset(self, seed=None, options=None):
            if seed is not None:
                self._env.seed(int(seed))
                self._next_init_state = int(seed)
            observation = self._env.reset()
            state_id = self._next_init_state % len(self._init_states)
            observation = self._env.set_init_state(self._init_states[state_id])
            self._next_init_state += 1
            return self._process_observation(observation), {
                "success": self._env.check_success(),
                "init_state_id": state_id,
            }

    return LiberoProEnv


@dataclass
class Config:
    split: str
    task_index: int
    host: str
    port: int
    output: Path
    data_root: Path
    episodes: int = 10
    environment_seed: int = 0
    max_episode_steps: int = 220
    execution_horizon: int = 8
    video_dir: Path | None = None
    smoke_test_only: bool = False


def main(config: Config) -> None:
    if config.split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {config.split!r}")
    repo = Path(__file__).resolve().parents[2]
    pro_python = repo / "external" / "LIBERO-PRO-code"
    groot_root = repo / "external" / "Isaac-GR00T"
    sys.path[:0] = [str(pro_python), str(groot_root)]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    bddl_dir = config.data_root / "bddl_files" / f"libero_spatial_{config.split}"
    init_dir = config.data_root / "init_files" / f"libero_spatial_{config.split}"
    tasks = sorted(bddl_dir.glob("*.bddl"))
    if len(tasks) != 10:
        raise ValueError(f"Expected 10 tasks in {bddl_dir}, found {len(tasks)}")
    if not 0 <= config.task_index < len(tasks):
        raise IndexError(f"task_index={config.task_index}, available=0..{len(tasks) - 1}")
    bddl_path = tasks[config.task_index]
    init_path = init_dir / f"{bddl_path.stem}.pruned_init"
    if not init_path.is_file():
        raise FileNotFoundError(init_path)

    task_description = read_language(bddl_path)
    env_id = f"libero_sim/pro_spatial_{config.split}_{config.task_index:02d}"
    gym.register(
        id=env_id,
        entry_point=make_libero_pro_env_class(),
        kwargs={
            "task_bddl_file": str(bddl_path),
            "task_description": task_description,
            "init_states_file": str(init_path),
        },
    )

    if config.smoke_test_only:
        env = gym.make(env_id)
        try:
            observation, info = env.reset(seed=config.environment_seed)
            action = {
                key: np.zeros(space.shape, dtype=space.dtype)
                for key, space in env.action_space.items()
            }
            _, _, _, _, step_info = env.step(action)
        finally:
            env.close()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "env_name": env_id,
                    "task": bddl_path.stem,
                    "observation_keys": sorted(observation),
                    "reset_info": {
                        "success": bool(info["success"]),
                        "init_state_id": int(info["init_state_id"]),
                    },
                    "step_success": bool(step_info["success"]),
                },
                indent=2,
            ),
            flush=True,
        )
        return

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
    success_count = int(sum(bool(value) for value in successes))
    result = {
        "benchmark": "LIBERO-PRO Spatial",
        "split": config.split,
        "task_index": config.task_index,
        "task_name": bddl_path.stem,
        "task_description": task_description,
        "env_name": env_name,
        "episodes": len(successes),
        "successes": success_count,
        "success_rate": float(success_count / len(successes)),
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
