#!/usr/bin/env python3
"""Compute the exact T5-11B prompt embeddings for the four official Bridge tasks on CPU."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig

INSTRUCTIONS = (
    "put carrot on plate",
    "put the spoon on the towel",
    "stack the green block on the yellow block",
    "put eggplant into yellow basket",
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
