from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from qkvs.config import ModelConfig
from qkvs.data import (
    PointerBatch,
    make_address_overrides,
    make_memory_anchor_batch,
    make_memory_bank_batch,
    make_multi_region_batch,
    make_oracle_window_masks,
    make_pointer_batch,
    make_semantic_batch,
    make_span_batch,
)
from qkvs.model import DecoderOutput, QKVSDecoder
from qkvs.runtime import environment, seed_everything, write_json


@dataclass(frozen=True)
class TrainConfig:
    name: str
    seed: int = 1
    task: str = "pointer"
    hops: int = 1
    semantic_keys: int = 8
    answer_vocab_size: int = 16
    memory_source_window: int = 8
    memory_anchor_source_offset: int | None = None
    memory_bank_keys: int = 8
    memory_bank_distractors: int = 3
    memory_bank_source_offset: int = 32
    memory_loss_weight: float = 0.0
    region_records: int = 100
    required_regions: int = 8
    route_assignment_weight: float = 0.25
    fixed_training_batch: bool = False
    train_seq_len: int = 64
    train_seq_lengths: tuple[int, ...] = ()
    eval_seq_lens: tuple[int, ...] = (64, 128)
    batch_size: int = 64
    steps: int = 300
    eval_batches: int = 8
    eval_batch_size: int | None = None
    learning_rate: float = 3e-4
    router_learning_rate: float | None = None
    weight_decay: float = 0.01
    route_loss_weight: float = 4.0
    route_loss_space: str = "normalized"
    teacher_decay_fraction: float = 0.8
    teacher_floor: float = 0.0
    teacher_only_oracle_window: bool = False
    hop_curriculum: tuple[int, ...] = ()
    query_only_retrieval: bool = False
    checkpoint_every: int = 100
    log_every: int = 50

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainConfig:
        copied = dict(value)
        copied["eval_seq_lens"] = tuple(copied.get("eval_seq_lens", (64, 128)))
        copied["train_seq_lengths"] = tuple(copied.get("train_seq_lengths", ()))
        copied["hop_curriculum"] = tuple(copied.get("hop_curriculum", ()))
        return cls(**copied)


def _batch(
    model_config: ModelConfig,
    train_config: TrainConfig,
    seq_len: int,
    generator: torch.Generator,
    hops: int | None = None,
    batch_size: int | None = None,
) -> PointerBatch:
    if train_config.task in {"semantic_fixed", "semantic_random"}:
        return make_semantic_batch(
            batch_size=(train_config.batch_size if batch_size is None else batch_size),
            seq_len=seq_len,
            dense_window=model_config.dense_window,
            block_size=model_config.block_size,
            address_stride=model_config.attention().stride,
            n_layers=model_config.n_layers,
            n_keys=train_config.semantic_keys,
            answer_vocab_size=train_config.answer_vocab_size,
            fixed_layout=train_config.task == "semantic_fixed",
            generator=generator,
        )
    if train_config.task == "span":
        return make_span_batch(
            batch_size=(train_config.batch_size if batch_size is None else batch_size),
            seq_len=seq_len,
            vocab_size=model_config.vocab_size,
            dense_window=model_config.dense_window,
            block_size=model_config.block_size,
            address_stride=model_config.attention().stride,
            n_layers=model_config.n_layers,
            generator=generator,
        )
    if train_config.task == "memory_anchor":
        return make_memory_anchor_batch(
            batch_size=(train_config.batch_size if batch_size is None else batch_size),
            seq_len=seq_len,
            vocab_size=model_config.vocab_size,
            dense_window=model_config.dense_window,
            address_stride=model_config.attention().stride,
            n_layers=model_config.n_layers,
            source_window=train_config.memory_source_window,
            source_offset=train_config.memory_anchor_source_offset,
            generator=generator,
        )
    if train_config.task == "memory_bank":
        return make_memory_bank_batch(
            batch_size=(train_config.batch_size if batch_size is None else batch_size),
            seq_len=seq_len,
            dense_window=model_config.dense_window,
            address_stride=model_config.attention().stride,
            n_layers=model_config.n_layers,
            n_keys=train_config.memory_bank_keys,
            answer_vocab_size=train_config.answer_vocab_size,
            distractors_per_marker=train_config.memory_bank_distractors,
            source_offset=train_config.memory_bank_source_offset,
            generator=generator,
        )
    if train_config.task == "multi_region":
        return make_multi_region_batch(
            batch_size=(train_config.batch_size if batch_size is None else batch_size),
            seq_len=seq_len,
            dense_window=model_config.dense_window,
            block_size=model_config.block_size,
            address_stride=model_config.attention().stride,
            n_windows=model_config.attention().n_windows,
            n_layers=model_config.n_layers,
            n_records=train_config.region_records,
            required_regions=train_config.required_regions,
            generator=generator,
        )
    if train_config.task != "pointer":
        raise ValueError(f"unknown task: {train_config.task}")
    return make_pointer_batch(
        batch_size=train_config.batch_size if batch_size is None else batch_size,
        seq_len=seq_len,
        vocab_size=model_config.vocab_size,
        dense_window=model_config.dense_window,
        block_size=model_config.block_size,
        address_stride=model_config.attention().stride,
        n_layers=model_config.n_layers,
        hops=train_config.hops if hops is None else hops,
        generator=generator,
    )


def _teacher_controls(
    batch: PointerBatch,
    model_config: ModelConfig,
    probability: float,
    generator: torch.Generator,
    device: torch.device,
    only_oracle_window: bool,
) -> tuple[Tensor | None, Tensor | None]:
    if model_config.attention_kind != "qkvs" or probability <= 0.0:
        return None, None
    overrides = make_address_overrides(
        batch,
        seq_len=batch.values.shape[1],
        n_windows=model_config.attention().n_windows,
        device=device,
    )
    teacher_mask = torch.rand(batch.values.shape[0], generator=generator) < probability
    overrides[:, ~teacher_mask.to(device), :, :] = torch.nan
    if not only_oracle_window:
        return overrides, None
    window_masks = torch.ones_like(overrides, dtype=torch.bool)
    teacher_on_device = teacher_mask.to(device)
    window_masks[:, teacher_on_device, -1] = False
    if batch.route_target_addresses is None:
        window_masks[:, teacher_on_device, -1, 0] = True
    else:
        window_masks[:, teacher_on_device, -1] = True
    return overrides, window_masks


def _window_target_intervals(
    positions: Tensor,
    *,
    history_end: int,
    block_size: int,
    stride: int,
) -> tuple[Tensor, Tensor]:
    n_addresses = math.ceil(history_end / stride)
    first_valid = torch.div(
        (positions - block_size + 1).clamp_min(0) + stride - 1,
        stride,
        rounding_mode="floor",
    )
    last_valid = torch.div(positions, stride, rounding_mode="floor").clamp_max(n_addresses - 1)
    return first_valid.float(), (last_valid + 1).float()


def _route_loss(
    output: DecoderOutput,
    batch: PointerBatch,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Tensor:
    if output.addresses is None or model_config.router_mode != "learned":
        return output.logits.new_zeros(())
    predicted = output.addresses[:, :, -1].float()
    if model_config.block_size <= 0:
        raise ValueError("block size must be positive")
    if train_config.route_loss_space == "set_coverage":
        if batch.route_target_positions is None:
            raise ValueError("set coverage requires route target positions")
        history_end = batch.values.shape[1] - model_config.dense_window
        stride = model_config.attention().stride
        n_addresses = math.ceil(history_end / stride)
        target_lower, target_upper = _window_target_intervals(
            batch.route_target_positions,
            history_end=history_end,
            block_size=model_config.block_size,
            stride=stride,
        )
        scaled = predicted.unsqueeze(-1) * n_addresses
        distance = (target_lower.unsqueeze(-2) - scaled).clamp_min(0)
        distance += (scaled - target_upper.unsqueeze(-2)).clamp_min(0)
        coverage_cost = F.smooth_l1_loss(
            distance,
            torch.zeros_like(distance),
            beta=0.25,
            reduction="none",
        )
        target_centers = (target_lower + target_upper) / 2.0
        sorted_predictions = (predicted * n_addresses).sort(dim=-1).values
        sorted_targets = target_centers.sort(dim=-1).values
        output_count = predicted.shape[-1]
        target_count = target_centers.shape[-1]
        quantiles = torch.div(
            (torch.arange(output_count, device=predicted.device) * 2 + 1) * target_count,
            2 * output_count,
            rounding_mode="floor",
        ).clamp_max(target_count - 1)
        matched_targets = sorted_targets[..., quantiles]
        coverage = coverage_cost.amin(dim=-2).mean()
        assignment = F.smooth_l1_loss(
            sorted_predictions,
            matched_targets,
            beta=0.25,
        )
        return coverage + train_config.route_assignment_weight * assignment
    predicted_first = predicted[..., 0]
    if train_config.route_loss_space == "normalized":
        return F.smooth_l1_loss(predicted_first, batch.route_addresses.float(), beta=0.02)
    if train_config.route_loss_space == "blocks":
        history_end = batch.values.shape[1] - model_config.dense_window
        stride = model_config.attention().stride
        n_addresses = math.ceil(history_end / stride)
        target_centers = (
            torch.div(
                batch.route_positions,
                stride,
                rounding_mode="floor",
            ).float()
            + 0.5
        )
        return F.smooth_l1_loss(predicted_first * n_addresses, target_centers, beta=0.25)
    raise ValueError(f"unknown route loss space: {train_config.route_loss_space}")


def _memory_loss(output: DecoderOutput, batch: PointerBatch) -> Tensor:
    if batch.memory_targets is None:
        return output.logits.new_zeros(())
    valid = batch.memory_targets >= 0
    if not valid.any():
        return output.logits.new_zeros(())
    return F.cross_entropy(output.logits[valid].float(), batch.memory_targets[valid])


def _retrieval_mask(batch: PointerBatch, train_config: TrainConfig) -> Tensor | None:
    if not train_config.query_only_retrieval:
        return None
    mask = torch.zeros_like(batch.values, dtype=torch.bool)
    mask[:, -1] = True
    return mask


def _route_accuracies(
    addresses: Tensor | None,
    batch: PointerBatch,
    model_config: ModelConfig,
) -> list[float] | None:
    if addresses is None:
        return None
    history_end = batch.values.shape[1] - model_config.dense_window
    stride = model_config.attention().stride
    n_addresses = math.ceil(history_end / stride)
    predicted = addresses[:, :, -1].float().clamp(0.0, 1.0 - torch.finfo(torch.float32).eps)
    predicted_starts = torch.floor(predicted * n_addresses).to(torch.long) * stride
    if batch.route_target_positions is None:
        targets = batch.route_positions.unsqueeze(-1)
        correct = (
            (predicted_starts <= targets) & (targets < predicted_starts + model_config.block_size)
        ).any(dim=-1)
    else:
        targets = batch.route_target_positions.unsqueeze(-2)
        contains = (predicted_starts.unsqueeze(-1) <= targets) & (
            targets < predicted_starts.unsqueeze(-1) + model_config.block_size
        )
        correct = contains.any(dim=-2).float().mean(dim=-1)
    return [value.item() for value in correct.float().mean(dim=1)]


def _cascade_sensitivity(
    model: QKVSDecoder,
    batch: PointerBatch,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> list[float] | None:
    if model_config.attention_kind != "qkvs" or batch.hops <= 1:
        return None
    with torch.no_grad():
        retrieval_mask = _retrieval_mask(batch, train_config)
        original = model(
            batch.values,
            batch.pointers,
            batch.token_types,
            retrieval_mask=retrieval_mask,
            semantic_ids=batch.semantic_ids,
        ).addresses
        if original is None:
            return None
        sensitivities: list[float] = []
        for layer_index in range(1, min(batch.hops, model_config.n_layers)):
            ablated_pointers = batch.pointers.clone()
            batch_indices = torch.arange(batch.values.shape[0], device=batch.values.device)
            source_positions = batch.route_positions[layer_index - 1]
            ablated_pointers[batch_indices, source_positions] = -1.0
            ablated = model(
                batch.values,
                ablated_pointers,
                batch.token_types,
                retrieval_mask=retrieval_mask,
                semantic_ids=batch.semantic_ids,
            ).addresses
            assert ablated is not None
            change = (
                original[layer_index, :, -1].float() - ablated[layer_index, :, -1].float()
            ).abs()
            sensitivities.append(change.mean().item())
    return sensitivities


def evaluate(
    model: QKVSDecoder,
    model_config: ModelConfig,
    train_config: TrainConfig,
    seq_len: int,
    *,
    oracle: bool,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total = 0
    correct = 0
    losses: list[float] = []
    route_sums: list[float] | None = None
    sensitivity_sums: list[float] | None = None

    for batch_index in range(train_config.eval_batches):
        batch = _batch(
            model_config,
            train_config,
            seq_len,
            generator,
            batch_size=train_config.eval_batch_size,
        ).to(device)
        overrides = None
        window_masks = None
        if oracle and model_config.attention_kind == "qkvs":
            overrides = make_address_overrides(
                batch,
                seq_len=seq_len,
                n_windows=model_config.attention().n_windows,
                device=device,
            )
            window_masks = make_oracle_window_masks(
                batch,
                seq_len=seq_len,
                n_windows=model_config.attention().n_windows,
                device=device,
            )
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            output = model(
                batch.values,
                batch.pointers,
                batch.token_types,
                address_overrides=overrides,
                retrieval_mask=_retrieval_mask(batch, train_config),
                semantic_ids=batch.semantic_ids,
                historical_window_masks=window_masks,
            )
            logits = output.logits[:, -1].float()
        losses.append(F.cross_entropy(logits, batch.answers).item())
        correct += (logits.argmax(dim=-1) == batch.answers).sum().item()
        total += batch.answers.numel()
        accuracies = _route_accuracies(output.addresses, batch, model_config)
        if accuracies is not None:
            route_sums = (
                accuracies
                if route_sums is None
                else [left + right for left, right in zip(route_sums, accuracies, strict=True)]
            )
        if batch_index == 0 and not oracle:
            sensitivity_sums = _cascade_sensitivity(model, batch, model_config, train_config)

    result: dict[str, Any] = {
        "answer_accuracy": correct / total,
        "answer_loss": sum(losses) / len(losses),
        "examples": total,
    }
    if route_sums is not None:
        result["route_accuracy_by_layer"] = [
            value / train_config.eval_batches for value in route_sums
        ]
        result["route_accuracy_mean"] = sum(result["route_accuracy_by_layer"]) / len(route_sums)
    if sensitivity_sums is not None:
        result["cascade_address_sensitivity"] = sensitivity_sums
    return result


def train_run(
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    output_directory: Path,
    device: torch.device,
    evaluation_device: torch.device | None = None,
    resume: bool = False,
    max_steps_this_run: int | None = None,
) -> dict[str, Any]:
    seed_everything(train_config.seed)
    model = QKVSDecoder(model_config).to(device)
    if (
        train_config.router_learning_rate is None
        or model_config.attention_kind != "qkvs"
        or model_config.router_mode != "learned"
    ):
        parameter_groups: Any = model.parameters()
    else:
        router_parameters = [
            parameter
            for layer in model.layers
            if layer.attention.s_proj is not None
            for parameter in layer.attention.s_proj.parameters()
        ]
        router_ids = {id(parameter) for parameter in router_parameters}
        base_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in router_ids
        ]
        parameter_groups = [
            {"params": base_parameters, "lr": train_config.learning_rate},
            {"params": router_parameters, "lr": train_config.router_learning_rate},
        ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        fused=device.type == "cuda",
    )
    generator = torch.Generator().manual_seed(train_config.seed + 101)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    start_step = 1
    checkpoint_directory = output_directory.parent / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / f"{train_config.name}.pt"

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "optimizer" not in checkpoint or "step" not in checkpoint:
            raise ValueError(f"checkpoint cannot resume training: {checkpoint_path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["data_generator_state"].cpu())
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        history = checkpoint.get("history", [])
        start_step = int(checkpoint["step"]) + 1

    def save_checkpoint(step: int) -> None:
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "model_config": model_config.as_dict(),
                "training_config": train_config.__dict__,
                "history": history,
                "data_generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
            },
            temporary,
        )
        temporary.replace(checkpoint_path)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    fixed_batch: PointerBatch | None = None
    if train_config.fixed_training_batch:
        if train_config.train_seq_lengths or train_config.hop_curriculum:
            raise ValueError("fixed training batches do not support length or hop curricula")
        fixed_batch = _batch(
            model_config,
            train_config,
            train_config.train_seq_len,
            torch.Generator().manual_seed(train_config.seed + 404),
        )

    end_step = train_config.steps
    if max_steps_this_run is not None:
        end_step = min(end_step, start_step + max_steps_this_run - 1)

    for step in range(start_step, end_step + 1):
        model.train()
        train_seq_len = train_config.train_seq_len
        if train_config.train_seq_lengths:
            train_seq_len = train_config.train_seq_lengths[
                (step - 1) % len(train_config.train_seq_lengths)
            ]
        training_hops = train_config.hops
        if train_config.hop_curriculum:
            phase = min(
                len(train_config.hop_curriculum) - 1,
                (step - 1) * len(train_config.hop_curriculum) // train_config.steps,
            )
            training_hops = train_config.hop_curriculum[phase]
        batch = (
            fixed_batch
            if fixed_batch is not None
            else _batch(
                model_config,
                train_config,
                train_seq_len,
                generator,
                hops=training_hops,
            )
        ).to(device)
        decay_steps = max(1, int(train_config.steps * train_config.teacher_decay_fraction))
        teacher_probability = train_config.teacher_floor + (1.0 - train_config.teacher_floor) * max(
            0.0, 1.0 - (step - 1) / decay_steps
        )
        overrides, window_masks = _teacher_controls(
            batch,
            model_config,
            teacher_probability,
            generator,
            device,
            train_config.teacher_only_oracle_window,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch.values,
                batch.pointers,
                batch.token_types,
                address_overrides=overrides,
                retrieval_mask=_retrieval_mask(batch, train_config),
                semantic_ids=batch.semantic_ids,
                historical_window_masks=window_masks,
            )
            answer_loss = F.cross_entropy(output.logits[:, -1].float(), batch.answers)
            route_loss = _route_loss(output, batch, model_config, train_config)
            memory_loss = _memory_loss(output, batch)
            loss = (
                answer_loss
                + train_config.route_loss_weight * route_loss
                + train_config.memory_loss_weight * memory_loss
            )
        should_log = step == 1 or step % train_config.log_every == 0 or step == train_config.steps
        loss.backward()
        router_gradient_norms: list[list[float]] | None = None
        if should_log and model_config.attention_kind == "qkvs":
            router_gradient_norms = []
            for layer in model.layers:
                projection = layer.attention.router_output_projection()
                if projection is None or projection.weight.grad is None:
                    router_gradient_norms.append([])
                else:
                    router_gradient_norms.append(
                        projection.weight.grad.norm(dim=1)[
                            : model_config.attention().n_windows
                        ].tolist()
                    )
        if model_config.router_detach_input:
            router_parameters = [
                parameter
                for layer in model.layers
                if layer.attention.s_proj is not None
                for parameter in layer.attention.s_proj.parameters()
            ]
            router_ids = {id(parameter) for parameter in router_parameters}
            base_parameters = [
                parameter for parameter in model.parameters() if id(parameter) not in router_ids
            ]
            torch.nn.utils.clip_grad_norm_(base_parameters, 1.0)
            torch.nn.utils.clip_grad_norm_(router_parameters, 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if should_log:
            accuracies = _route_accuracies(output.addresses, batch, model_config)
            entry: dict[str, Any] = {
                "step": step,
                "loss": loss.item(),
                "answer_loss": answer_loss.item(),
                "route_loss": route_loss.item(),
                "memory_loss": memory_loss.item(),
                "teacher_probability": teacher_probability,
                "training_hops": training_hops,
                "train_seq_len": train_seq_len,
            }
            if accuracies is not None:
                entry["route_accuracy_mean"] = sum(accuracies) / len(accuracies)
            if router_gradient_norms is not None:
                entry["router_gradient_norms_by_layer"] = router_gradient_norms
            history.append(entry)
            print(
                f"{train_config.name}: step={step}/{train_config.steps} "
                f"answer={answer_loss.item():.4f} route={route_loss.item():.4f} "
                f"memory={memory_loss.item():.4f} "
                f"teacher={teacher_probability:.2f}",
                flush=True,
            )
        if step % train_config.checkpoint_every == 0 or step == end_step:
            save_checkpoint(step)

    if start_step > train_config.steps:
        save_checkpoint(train_config.steps)

    peak_cuda_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    training_complete = end_step >= train_config.steps or start_step > train_config.steps
    elapsed = time.perf_counter() - started
    if not training_complete:
        result = {
            "name": train_config.name,
            "model": model_config.as_dict(),
            "training": {
                **train_config.__dict__,
                "eval_seq_lens": list(train_config.eval_seq_lens),
            },
            "history": history,
            "evaluations": {},
            "elapsed_seconds": elapsed,
            "peak_training_cuda_memory_bytes": peak_cuda_memory,
            "training_complete": False,
            "completed_step": end_step,
            "environment": environment(),
        }
        output_directory.mkdir(parents=True, exist_ok=True)
        write_json(output_directory / f"{train_config.name}.json", result)
        return result

    evaluation_device = device if evaluation_device is None else evaluation_device
    if evaluation_device != device:
        model = model.to(evaluation_device)
        del optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    evaluations: dict[str, Any] = {}
    for seq_len in train_config.eval_seq_lens:
        evaluations[str(seq_len)] = {
            "native": evaluate(
                model,
                model_config,
                train_config,
                seq_len,
                oracle=False,
                seed=train_config.seed + 10_000 + seq_len,
                device=evaluation_device,
            )
        }
        if model_config.attention_kind == "qkvs":
            evaluations[str(seq_len)]["oracle"] = evaluate(
                model,
                model_config,
                train_config,
                seq_len,
                oracle=True,
                seed=train_config.seed + 10_000 + seq_len,
                device=evaluation_device,
            )

    elapsed = time.perf_counter() - started
    result = {
        "name": train_config.name,
        "model": model_config.as_dict(),
        "training": {
            **train_config.__dict__,
            "eval_seq_lens": list(train_config.eval_seq_lens),
        },
        "history": history,
        "evaluations": evaluations,
        "elapsed_seconds": elapsed,
        "peak_training_cuda_memory_bytes": peak_cuda_memory,
        "training_complete": True,
        "completed_step": train_config.steps,
        "environment": environment(),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / f"{train_config.name}.json", result)
    return result
