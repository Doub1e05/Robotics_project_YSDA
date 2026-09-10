#!/usr/bin/env python3
"""Serve the unmodified GR00T N1.7 SIMPLER-Bridge policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import tyro


@dataclass
class Config:
    model_path: str
    port: int = 5574


def main(config: Config) -> None:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "external" / "Isaac-GR00T"))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    base = Gr00tPolicy(EmbodimentTag.SIMPLER_ENV_WIDOWX, config.model_path, device="cuda")
    policy = Gr00tSimPolicyWrapper(base)
    print(f"GR00T baseline ready: port={config.port}", flush=True)
    with PolicyServer(policy, host="127.0.0.1", port=config.port) as server:
        server.run()


if __name__ == "__main__":
    main(tyro.cli(Config))
