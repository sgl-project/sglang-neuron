#!/usr/bin/env python3
"""Create a no-copy, text-only view of a Qwen3.5 dense or MoE checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error(
            "source and destination must differ to preserve the source checkpoint"
        )
    args.destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.source / "config.json").read_text())
    text = dict(config["text_config"])
    model_type = text["model_type"]
    architectures = {
        "qwen3_5_moe_text": "Qwen3_5MoeForCausalLM",
        "qwen3_5_text": "Qwen3_5ForCausalLM",
    }
    text["architectures"] = [architectures[model_type]]
    text["tie_word_embeddings"] = config.get("tie_word_embeddings", False)
    target_config = args.destination / "config.json"
    if target_config.is_symlink():
        target_config.unlink()
    target_config.write_text(json.dumps(text, indent=2, sort_keys=True) + "\n")
    for entry in args.source.iterdir():
        if entry.name == "config.json" or entry.name.startswith("model.visual"):
            continue
        target = args.destination / entry.name
        if entry.name == "tokenizer_config.json":
            tokenizer_config = json.loads(entry.read_text())
            tokenizer_config["fix_mistral_regex"] = True
            if target.is_symlink():
                target.unlink()
            target.write_text(
                json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n"
            )
            continue
        if not target.exists():
            target.symlink_to(os.path.relpath(entry, args.destination))


if __name__ == "__main__":
    main()
