#!/usr/bin/env python3
"""Serve GR00T N1.7 with fixed-seed action-space consensus-only selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import tyro


def _add_gr00t_to_path(repo: Path) -> None:
    gr00t_root = repo / "external" / "Isaac-GR00T"
    if not gr00t_root.is_dir():
        raise FileNotFoundError(f"Missing Isaac-GR00T checkout: {gr00t_root}")
    sys.path.insert(0, str(gr00t_root))


class ConsensusOnlySimPolicy:
    """Wrap a GR00T SIMPLER policy with deterministic, per-call K-candidate medoid selection."""

    def __init__(self, policy: Any, candidate_seeds: tuple[int, ...]) -> None:
        if len(candidate_seeds) < 2:
            raise ValueError("Consensus requires at least two candidate seeds.")
        self.policy = policy
        self.candidate_seeds = candidate_seeds

    def get_modality_config(self) -> Any:
        return self.policy.get_modality_config()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.policy.reset(options)

    def _sample_candidate(
        self, seed: int, observation: dict[str, Any], options: dict[str, Any] | None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        # The underlying GR00T flow policy samples initial action noise with torch.randn.
        # Forking restores the outer RNG state, so every control call uses precisely the
        # requested fixed candidate seed rather than inheriting simulator randomness.
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            return self.policy.get_action(observation, options)

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        candidates = [self._sample_candidate(seed, observation, options)[0] for seed in self.candidate_seeds]
        action_keys = tuple(sorted(candidates[0]))
        batch_size = candidates[0][action_keys[0]].shape[0]
        selected = np.empty(batch_size, dtype=np.int64)
        costs = np.empty((batch_size, len(candidates)), dtype=np.float32)

        for batch_idx in range(batch_size):
            distances = np.zeros((len(candidates), len(candidates)), dtype=np.float64)
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    squared = 0.0
                    count = 0
                    for key in action_keys:
                        delta = candidates[i][key][batch_idx] - candidates[j][key][batch_idx]
                        squared += float(np.square(delta, dtype=np.float64).sum())
                        count += int(delta.size)
                    distances[i, j] = distances[j, i] = np.sqrt(squared / max(count, 1))
            costs[batch_idx] = distances.sum(axis=1) / (len(candidates) - 1)
            selected[batch_idx] = int(np.argmin(costs[batch_idx]))

        action = {
            key: np.stack(
                [candidates[selected[b]][key][b] for b in range(batch_size)], axis=0
            ).astype(np.float32, copy=False)
            for key in action_keys
        }
        info = {
            "consensus_mode": "action_medoid_only",
            "candidate_seeds": list(self.candidate_seeds),
            "selected_candidate_indices": selected.tolist(),
            "candidate_mean_pairwise_distances": costs.tolist(),
        }
        return action, info


@dataclass
class Config:
    model_path: str
    embodiment_tag: str = "SIMPLER_ENV_WIDOWX"
    device: str = "cuda"
    host: str = "127.0.0.1"
    port: int = 5555
    candidate_seeds: str = "1,994,995"


def main(config: Config) -> None:
    repo = Path(__file__).resolve().parents[2]
    _add_gr00t_to_path(repo)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    seeds = tuple(int(value.strip()) for value in config.candidate_seeds.split(",") if value.strip())
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"candidate_seeds must be unique, got {seeds}")
    base_policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(config.embodiment_tag),
        model_path=config.model_path,
        device=config.device,
    )
    policy = ConsensusOnlySimPolicy(Gr00tSimPolicyWrapper(base_policy), seeds)
    print(f"GR00T consensus-only ready: seeds={seeds}, port={config.port}", flush=True)
    with PolicyServer(policy=policy, host=config.host, port=config.port) as server:
        server.run()


if __name__ == "__main__":
    main(tyro.cli(Config))
