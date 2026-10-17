from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from qkvs.config import ModelConfig
from qkvs.data import PointerBatch, make_address_overrides, make_multi_region_batch
from qkvs.model import QKVSDecoder
from qkvs.runtime import environment, write_json
from qkvs.training import TrainConfig, _route_loss


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared = [
        parameter.grad.float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared:
        return 0.0
    return math.sqrt(sum(value.item() for value in squared))


def _parameter_groups(model: QKVSDecoder) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    router = [
        parameter
        for layer in model.layers
        for module in (
            layer.attention.s_proj,
            layer.attention.s_offset_proj,
            layer.attention.router_position_projection,
            layer.attention.router_pointer_projection,
        )
        if module is not None
        for parameter in module.parameters()
    ]
    router_ids = {id(parameter) for parameter in router}
    base = [parameter for parameter in model.parameters() if id(parameter) not in router_ids]
    return router, base


def _output_gradient_norms(model: QKVSDecoder) -> list[list[float]]:
    result: list[list[float]] = []
    for layer in model.layers:
        projection = layer.attention.router_output_projection()
        if projection is None or projection.weight.grad is None:
            result.append([])
        else:
            result.append(projection.weight.grad.norm(dim=1).tolist())
    return result


def _module_gradient_norm(module: nn.Module | None) -> float:
    return 0.0 if module is None else _gradient_norm(module.parameters())


def _loss_backward(
    model: QKVSDecoder,
    model_config: ModelConfig,
    train_config: TrainConfig,
    batch: PointerBatch,
    *,
    loss_kind: str,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    overrides = None
    if loss_kind == "answer_oracle":
        overrides = make_address_overrides(
            batch,
            seq_len=batch.values.shape[1],
            n_windows=model_config.attention().n_windows,
            device=torch.device("cpu"),
        )
    output = model(
        batch.values,
        batch.pointers,
        batch.token_types,
        address_overrides=overrides,
    )
    if loss_kind in {"answer_native", "answer_oracle"}:
        loss = F.cross_entropy(output.logits[:, -1], batch.answers)
    elif loss_kind == "route":
        loss = _route_loss(output, batch, model_config, train_config)
    else:
        raise ValueError(f"unknown audit loss: {loss_kind}")
    loss.backward()
    router_parameters, base_parameters = _parameter_groups(model)
    router_norm = _gradient_norm(router_parameters)
    base_norm = _gradient_norm(base_parameters)
    clip_threshold = 1.0
    assert output.addresses is not None
    addresses = output.addresses.detach()
    logits = torch.logit(addresses.clamp(1e-6, 1.0 - 1e-6))
    return {
        "loss": loss.item(),
        "router_gradient_norm": router_norm,
        "base_gradient_norm": base_norm,
        "router_clip_ratio_at_1": min(1.0, clip_threshold / max(router_norm, 1e-12)),
        "base_clip_ratio_at_1": min(1.0, clip_threshold / max(base_norm, 1e-12)),
        "router_output_gradient_norms_by_layer": _output_gradient_norms(model),
        "categorical_gradient_norm_by_layer": [
            _module_gradient_norm(layer.attention.s_proj) for layer in model.layers
        ],
        "local_offset_gradient_norm_by_layer": [
            _module_gradient_norm(layer.attention.s_offset_proj) for layer in model.layers
        ],
        "pointer_projection_gradient_norm_by_layer": [
            _module_gradient_norm(layer.attention.router_pointer_projection)
            for layer in model.layers
        ],
        "logit_absolute_mean": logits.abs().mean().item(),
        "logit_absolute_max": logits.abs().max().item(),
        "logit_saturation_fraction": (logits.abs() > 4.0).float().mean().item(),
        "address_saturation_fraction": (
            ((addresses < 0.01) | (addresses > 0.99)).float().mean().item()
        ),
    }


def _audit(*, detach: bool, position_encoding: bool) -> dict[str, Any]:
    torch.manual_seed(16101 + 100 * int(detach) + int(position_encoding))
    model_config = ModelConfig(
        vocab_size=9,
        d_model=96,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        dense_window=32,
        retrieval_budget=64,
        block_size=8,
        address_stride=4,
        attention_kind="qkvs",
        router_capacity=8,
        router_detach_input=detach,
        router_mlp_ratio=2,
        router_position_encoding=("absolute_fourier" if position_encoding else "none"),
        pointer_set_encoding="sorted_fourier",
        sparse_backend="materialized",
    )
    train_config = TrainConfig(
        name="gradient_audit",
        task="multi_region",
        region_records=24,
        required_regions=8,
        train_seq_len=512,
        batch_size=8,
        route_loss_weight=1.0,
        route_loss_space="set_coverage",
        route_assignment_weight=1.0,
    )
    batch = make_multi_region_batch(
        batch_size=8,
        seq_len=512,
        dense_window=32,
        block_size=8,
        address_stride=4,
        n_windows=8,
        n_layers=2,
        n_records=24,
        required_regions=8,
        generator=torch.Generator().manual_seed(16201),
    )
    model = QKVSDecoder(model_config)
    return {
        "router_detach_input": detach,
        "router_position_encoding": model_config.router_position_encoding,
        "answer_native": _loss_backward(
            model,
            model_config,
            train_config,
            batch,
            loss_kind="answer_native",
        ),
        "answer_oracle": _loss_backward(
            model,
            model_config,
            train_config,
            batch,
            loss_kind="answer_oracle",
        ),
        "route": _loss_backward(
            model,
            model_config,
            train_config,
            batch,
            loss_kind="route",
        ),
    }


def _soft_neighbor_audit() -> dict[str, Any]:
    torch.manual_seed(16301)
    model_config = ModelConfig(
        vocab_size=9,
        d_model=96,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        dense_window=32,
        retrieval_budget=64,
        block_size=8,
        address_stride=4,
        attention_kind="qkvs",
        router_capacity=8,
        router_detach_input=False,
        router_mlp_ratio=2,
        router_position_encoding="absolute_fourier",
        router_pointer_input=True,
        unrotated_address_matching=True,
        router_output_encoding="hierarchical",
        router_radix=16,
        router_local_offset=True,
        soft_neighbor_training=True,
        pointer_set_encoding="sorted_fourier",
        sparse_backend="materialized",
    )
    train_config = TrainConfig(
        name="soft_gradient_audit",
        task="multi_region",
        region_records=24,
        required_regions=8,
        train_seq_len=512,
        batch_size=8,
        route_loss_weight=1.0,
        route_loss_space="set_coverage",
        route_assignment_weight=0.0,
    )
    batch = make_multi_region_batch(
        batch_size=8,
        seq_len=512,
        dense_window=32,
        block_size=8,
        address_stride=4,
        n_windows=8,
        n_layers=2,
        n_records=24,
        required_regions=8,
        generator=torch.Generator().manual_seed(16401),
    )
    model = QKVSDecoder(model_config)
    return {
        "answer_native": _loss_backward(
            model,
            model_config,
            train_config,
            batch,
            loss_kind="answer_native",
        ),
        "route": _loss_backward(
            model,
            model_config,
            train_config,
            batch,
            loss_kind="route",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate answer and routing gradients into S")
    parser.add_argument("--output", default="artifacts/results/router_gradient_audit_v2.json")
    arguments = parser.parse_args()
    result = {
        "environment": environment(),
        "detached_hidden": _audit(detach=True, position_encoding=False),
        "joint_hidden": _audit(detach=False, position_encoding=False),
        "joint_hidden_with_coordinates": _audit(detach=False, position_encoding=True),
        "joint_hierarchical_soft_neighbors": _soft_neighbor_audit(),
        "interpretation": (
            "Hard selection gives answer loss no gradient to S, with native or oracle windows. "
            "Route loss trains S; detaching its input prevents that loss from adapting the shared "
            "representation. Raw norms and hypothetical clip ratios are reported separately."
        ),
    }
    write_json(Path(arguments.output), result)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
