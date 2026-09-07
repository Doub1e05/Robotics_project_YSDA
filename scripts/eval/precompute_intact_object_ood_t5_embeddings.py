#!/usr/bin/env python3
"""Compute T5-11B prompt embeddings for all 16 INT-ACT Object Diversity tasks on CPU."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig

INSTRUCTIONS = (
    "put green cube on plate",
    "put the small plate on the green cube",
    "put coke can on plate",
    "put pepsi can on plate",
    "put carrot on sponge",
    "put eggplant on sponge",
    "put carrot on keyboard",
    "put coke can on keyboard",
    "put orange juice on plate",
    "put nut on plate",
    "put eggplant on keyboard",
    "put carrot on ramekin",
    "put carrot on wheel",
    "put coke can on ramekin",
    "put coke can on wheel",
    "put nut on wheel",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--t5-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "model/checkpoints/text_encoder/t5-11b",
    )
    args = parser.parse_args()
    if not (args.t5_dir / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"Missing T5 weights: {args.t5_dir / 'pytorch_model.bin'}")

    config = CosmosT5TextEncoderConfig(ckpt_path=str(args.t5_dir))
    encoder = CosmosT5TextEncoder(config=config, device="cpu", torch_dtype=None)
    embeddings: dict[str, torch.Tensor] = {}
    for instruction in INSTRUCTIONS:
        embedding = encoder.encode_prompts(instruction).cpu().contiguous()
        embeddings[instruction] = embedding
        print(f"embedded {instruction!r}: shape={tuple(embedding.shape)} dtype={embedding.dtype}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, args.output)
    print(f"saved {len(embeddings)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
