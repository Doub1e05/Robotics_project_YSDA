#!/usr/bin/env python3
"""Evaluate one official LIBERO-Plus Spatial task with a GR00T policy server."""

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


def init_states_path(init_root: Path, task) -> Path:
    """Mirror LIBERO-Plus's task-to-init mapping without torch.load defaults."""
    name = task.init_states_file
    if "_language_" in name:
        name = name.split("_language_")[0] + Path(name).suffix
        return init_root / task.problem_folder / name
    if "_view_" in name:
        name = name.split("_view_")[0] + Path(name).suffix
        return init_root / task.problem_folder / name
    if "_table_" in name:
        name = re.sub(r"_table_\d+", "", name)
    elif "_tb_" in name:
        name = re.sub(r"_tb_\d+", "", name)
    elif "_light_" in name:
        name = name.split("_light_")[0] + Path(name).suffix
    if "_add_" in task.init_states_file or "_level" in task.init_states_file:
        return init_root / "libero_newobj" / task.problem_folder / name
    return init_root / task.problem_folder / name


def make_env_class():
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv

    class LiberoPlusEnv(LiberoEnv):
        def __init__(
            self,
            task_bddl_file: str,
            task_description: str,
            init_state: list[float],
        ) -> None:
            super().__init__(task_bddl_file, task_description)
            self._fixed_init_state = np.asarray(init_state)

        def reset(self, seed=None, options=None):
            if seed is not None:
                self._env.seed(int(seed))
            observation = self._env.reset()
            observation = self._env.set_init_state(self._fixed_init_state)
            return self._process_observation(observation), {
                "success": self._env.check_success(),
                "init_state_id": 0,
            }

    return LiberoPlusEnv


@dataclass
class Config:
    task_index: int
    host: str
    port: int
    output: Path
    libero_root: Path
    classification_file: Path
    environment_seed: int = 0
    max_episode_steps: int = 220
    execution_horizon: int = 8
    smoke_test_only: bool = False


def main(config: Config) -> None:
    repo = Path(__file__).resolve().parents[2]
    plus_root = config.libero_root.resolve()
    groot_root = repo / "external" / "Isaac-GR00T"
    sys.path[:0] = [str(plus_root), str(groot_root)]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.utils import get_libero_path

    suite = get_benchmark_dict()["libero_spatial"]()
    if not 0 <= config.task_index < suite.get_num_tasks():
        raise IndexError(
            f"task_index={config.task_index}, available=0..{suite.get_num_tasks() - 1}"
        )
    task = suite.get_task(config.task_index)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    state_path = init_states_path(Path(get_libero_path("init_states")), task)
    states = torch.load(state_path, map_location="cpu", weights_only=False)
    if isinstance(states, torch.Tensor):
        states = states.numpy()
    states = np.asarray(states)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    if len(states) == 0:
        raise ValueError(f"No init states in {state_path}")

    classifications = json.loads(config.classification_file.read_text(encoding="utf-8"))
    classification = classifications["libero_spatial"][config.task_index]
    env_id = f"libero_sim/plus_spatial_{config.task_index:04d}"
    gym.register(
        id=env_id,
        entry_point=make_env_class(),
        kwargs={
            "task_bddl_file": str(bddl_path),
            "task_description": task.language,
            "init_state": states[0].tolist(),
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
        print(json.dumps({
            "status": "ok",
            "task_index": config.task_index,
            "task": task.name,
            "category": classification["category"],
            "difficulty_level": classification["difficulty_level"],
            "observation_keys": sorted(observation),
            "reset_success": bool(info["success"]),
            "step_success": bool(step_info["success"]),
        }, indent=2), flush=True)
        return

    from gr00t.eval.rollout_policy import run_gr00t_sim_policy

    env_name, successes, info = run_gr00t_sim_policy(
        env_name=env_id,
        n_episodes=1,
        max_episode_steps=config.max_episode_steps,
        policy_client_host=config.host,
        policy_client_port=config.port,
        n_envs=1,
        n_action_steps=config.execution_horizon,
        video_dir=None,
        seed=config.environment_seed,
    )
    result = {
        "benchmark": "LIBERO-Plus Spatial fixed first-100 slice",
        "task_index": config.task_index,
        "task_name": task.name,
        "task_description": task.language,
        "category": classification["category"],
        "difficulty_level": classification["difficulty_level"],
        "env_name": env_name,
        "episodes": len(successes),
        "successes": int(sum(bool(value) for value in successes)),
        "success_rate": float(sum(bool(value) for value in successes) / len(successes)),
        "environment_seed": config.environment_seed,
        "init_state_id": 0,
        "execution_horizon": config.execution_horizon,
        "episode_successes": [bool(value) for value in successes],
        "episode_lengths": [int(value) for value in info.get("episode_lengths", [])],
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
