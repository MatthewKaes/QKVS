from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from qkvs.config import ModelConfig
from qkvs.data import make_address_overrides, make_pointer_batch
from qkvs.model import QKVSDecoder
from qkvs.runtime import environment, write_json
from qkvs.training import TrainConfig, _route_loss


def _audit(retrieval_budget: int) -> dict[str, Any]:
    torch.manual_seed(16101 + retrieval_budget)
    model_config = ModelConfig(
        vocab_size=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        dense_window=32,
        retrieval_budget=retrieval_budget,
        block_size=16,
        address_stride=16,
        attention_kind="qkvs",
        sparse_backend="materialized",
    )
    train_config = TrainConfig(
        name="gradient_audit",
        hops=1,
        train_seq_len=192,
        batch_size=4,
        route_loss_weight=0.1,
        route_loss_space="blocks",
    )
    batch = make_pointer_batch(
        batch_size=4,
        seq_len=192,
        vocab_size=16,
        dense_window=32,
        block_size=16,
        address_stride=16,
        n_layers=2,
        hops=1,
        generator=torch.Generator().manual_seed(16201),
    )
    model = QKVSDecoder(model_config)
    overrides = make_address_overrides(
        batch,
        seq_len=192,
        n_windows=model_config.attention().n_windows,
        device=torch.device("cpu"),
    )
    output = model(
        batch.values,
        batch.pointers,
        batch.token_types,
        address_overrides=overrides,
    )
    answer_loss = F.cross_entropy(output.logits[:, -1], batch.answers)
    route_loss = _route_loss(output, batch, model_config, train_config)
    (answer_loss + train_config.route_loss_weight * route_loss).backward()
    layers = []
    for layer_index, layer in enumerate(model.layers):
        projection = layer.attention.s_proj
        assert isinstance(projection, torch.nn.Linear)
        assert projection.weight.grad is not None
        assert projection.bias.grad is not None
        layers.append(
            {
                "layer": layer_index,
                "weight_gradient_norm_by_window": projection.weight.grad.norm(dim=1).tolist(),
                "bias_gradient_absolute_by_window": projection.bias.grad.abs().tolist(),
            }
        )
    return {
        "retrieval_budget": retrieval_budget,
        "block_size": 16,
        "n_windows": model_config.attention().n_windows,
        "answer_loss": answer_loss.item(),
        "route_loss": route_loss.item(),
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit gradient coverage of S output windows")
    parser.add_argument("--output", default="artifacts/results/router_gradient_audit.json")
    arguments = parser.parse_args()
    result = {
        "environment": environment(),
        "multiwindow": _audit(128),
        "single_window": _audit(16),
        "interpretation": (
            "Hard selection does not propagate answer gradients to S. The current route loss "
            "supervises output zero only; a one-window configuration removes uncovered outputs."
        ),
    }
    write_json(Path(arguments.output), result)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
