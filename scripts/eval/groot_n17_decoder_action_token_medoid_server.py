#!/usr/bin/env python3
"""Serve GR00T N1.7 with fixed-seed decoder action-token medoid selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import tyro


class DecoderActionTokenMedoidPolicy:
    """Select a candidate using final DiT action-token trajectories only."""

    def __init__(self, policy: Any, candidate_seeds: tuple[int, ...]) -> None:
        self.policy = policy
        self.candidate_seeds = candidate_seeds
        self._tokens: np.ndarray | None = None
        self._hook = self.policy.policy.model.action_head.model.register_forward_hook(self._capture)

    def _capture(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("Expected GR00T DiT decoder states with shape [B, tokens, hidden].")
        # Position zero is the state token; the remaining positions are action tokens.
        self._tokens = hidden[:, 1:, :].detach().float().cpu().numpy()

    def get_modality_config(self) -> Any:
        return self.policy.get_modality_config()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.policy.reset(options)

    def _sample(
        self, seed: int, observation: dict[str, Any], options: dict[str, Any] | None
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self._tokens = None
            action, _ = self.policy.get_action(observation, options)
            if self._tokens is None:
                raise RuntimeError("No decoder action tokens captured for GR00T candidate.")
            return action, self._tokens

    @staticmethod
    def _costs(tokens: list[np.ndarray]) -> np.ndarray:
        count, batch = len(tokens), tokens[0].shape[0]
        costs = np.zeros((batch, count), dtype=np.float64)
        for batch_idx in range(batch):
            pairwise = np.zeros((count, count), dtype=np.float64)
            for i in range(count):
                for j in range(i + 1, count):
                    first, second = tokens[i][batch_idx], tokens[j][batch_idx]
                    length = min(first.shape[0], second.shape[0])
                    first, second = first[:length], second[:length]
                    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
                    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
                    per_token = 1.0 - np.clip(np.sum(first * second, axis=-1), -1.0, 1.0)
                    weights = np.ones(length, dtype=np.float64)
                    weights[: min(4, length)] = 4.0
                    pairwise[i, j] = pairwise[j, i] = float(np.average(per_token, weights=weights))
            costs[batch_idx] = pairwise.sum(axis=1) / (count - 1)
        return costs

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        samples = [self._sample(seed, observation, options) for seed in self.candidate_seeds]
        actions, tokens = zip(*samples)
        costs = self._costs(list(tokens))
        selected = np.argmin(costs, axis=1)
        keys = tuple(sorted(actions[0]))
        action = {
            key: np.stack([actions[selected[b]][key][b] for b in range(len(selected))]).astype(np.float32)
            for key in keys
        }
        return action, {
            "selector": "decoder_action_token_medoid",
            "candidate_seeds": list(self.candidate_seeds),
            "selected_candidate_indices": selected.tolist(),
            "candidate_token_medoid_costs": costs.tolist(),
        }


@dataclass
class Config:
    model_path: str
    embodiment_tag: str = "SIMPLER_ENV_WIDOWX"
    device: str = "cuda"
    host: str = "127.0.0.1"
    port: int = 5572
    candidate_seeds: str = "1,999,998"


def main(config: Config) -> None:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "external" / "Isaac-GR00T"))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    seeds = tuple(int(value.strip()) for value in config.candidate_seeds.split(",") if value.strip())
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError(f"Expected three unique candidate seeds, received {seeds}.")
    base = Gr00tPolicy(
        EmbodimentTag.resolve(config.embodiment_tag),
        config.model_path,
        device=config.device,
    )
    policy = DecoderActionTokenMedoidPolicy(Gr00tSimPolicyWrapper(base), seeds)
    print(f"GR00T decoder-action-token-medoid ready: seeds={seeds}, port={config.port}", flush=True)
    with PolicyServer(policy, host=config.host, port=config.port) as server:
        server.run()


if __name__ == "__main__":
    main(tyro.cli(Config))
