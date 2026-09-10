#!/usr/bin/env python3
"""Serve the LIBERO GR00T N1.7 policy with deterministic diffusion sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import tyro


class FixedSeedSimPolicy:
    """Reset the policy sampling RNG before every action-chunk prediction."""

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
    action_seed: int = 1
    port: int = 5594


def main(config: Config) -> None:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "external" / "Isaac-GR00T"))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    base = Gr00tPolicy(EmbodimentTag.LIBERO_PANDA, config.model_path, device="cuda")
    policy = FixedSeedSimPolicy(Gr00tSimPolicyWrapper(base), config.action_seed)
    print(
        f"GR00T LIBERO fixed-seed baseline ready: action_seed={config.action_seed}, "
        f"port={config.port}",
        flush=True,
    )
    with PolicyServer(policy, host="127.0.0.1", port=config.port) as server:
        server.run()


if __name__ == "__main__":
    main(tyro.cli(Config))
