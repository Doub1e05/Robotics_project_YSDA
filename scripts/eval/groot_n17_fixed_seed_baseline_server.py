#!/usr/bin/env python3
"""Serve an unmodified GR00T policy with a fixed action-noise seed per step."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import tyro


class FixedSeedSimPolicy:
    """Keep the policy unchanged while making every action-chunk sample reproducible."""

    def __init__(self, policy: Any, action_seed: int) -> None:
        self.policy = policy
        self.action_seed = action_seed

    def get_modality_config(self) -> Any:
        return self.policy.get_modality_config()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.policy.reset(options)

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        # Match the fixed-candidate semantics of consensus: seed k produces the same
        # conditional sample at every control step, while the observation still changes.
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(self.action_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.action_seed)
            action, info = self.policy.get_action(observation, options)
        info = dict(info or {})
        info["action_seed"] = self.action_seed
        return action, info


@dataclass
class Config:
    model_path: str
    action_seed: int
    port: int = 5576


def main(config: Config) -> None:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "external" / "Isaac-GR00T"))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    base = Gr00tPolicy(EmbodimentTag.SIMPLER_ENV_WIDOWX, config.model_path, device="cuda")
    policy = FixedSeedSimPolicy(Gr00tSimPolicyWrapper(base), config.action_seed)
    print(f"GR00T fixed-seed baseline ready: action_seed={config.action_seed}, port={config.port}", flush=True)
    with PolicyServer(policy, host="127.0.0.1", port=config.port) as server:
        server.run()


if __name__ == "__main__":
    main(tyro.cli(Config))
