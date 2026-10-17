from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a replayable model-only checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    source = Path(arguments.checkpoint)
    destination = Path(arguments.output)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    payload = {
        "format_version": 1,
        "model": checkpoint["model"],
        "model_config": checkpoint["model_config"],
        "training_config": checkpoint.get("training_config"),
        "step": checkpoint.get("step"),
        "source_checkpoint": str(source),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"{destination} sha256={digest}")


if __name__ == "__main__":
    main()
