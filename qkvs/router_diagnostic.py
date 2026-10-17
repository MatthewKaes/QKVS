from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from qkvs.runtime import environment, load_json, seed_everything, write_json

RouterOutput = Literal["scalar_cell", "scalar_normalized", "hierarchical"]


@dataclass(frozen=True)
class RouterBatch:
    pointers: Tensor
    address_counts: Tensor
    target_cells: Tensor
    target_positions: Tensor

    def to(self, device: torch.device) -> RouterBatch:
        return RouterBatch(
            self.pointers.to(device),
            self.address_counts.to(device),
            self.target_cells.to(device),
            self.target_positions.to(device),
        )


def make_router_batch(
    *,
    batch_size: int,
    n_windows: int,
    context_lengths: tuple[int, ...],
    dense_window: int,
    address_stride: int,
    generator: torch.Generator,
) -> RouterBatch:
    """Generate shuffled pointer sets without constructing token sequences."""
    length_indices = torch.randint(len(context_lengths), (batch_size,), generator=generator)
    lengths = torch.tensor(context_lengths, dtype=torch.long)[length_indices]
    history_end = lengths - dense_window
    address_counts = torch.div(
        history_end + address_stride - 1,
        address_stride,
        rounding_mode="floor",
    )
    if int(address_counts.min()) < n_windows:
        raise ValueError("every context needs at least one distinct cell per router output")

    target_cells = torch.floor(
        torch.rand((batch_size, n_windows), generator=generator) * address_counts.unsqueeze(-1)
    ).to(torch.long)
    duplicate_rows = torch.ones(batch_size, dtype=torch.bool)
    while duplicate_rows.any():
        sorted_cells = target_cells.sort(dim=-1).values
        duplicate_rows = (sorted_cells[:, 1:] == sorted_cells[:, :-1]).any(dim=-1)
        duplicate_count = int(duplicate_rows.sum())
        if duplicate_count:
            replacements = torch.floor(
                torch.rand((duplicate_count, n_windows), generator=generator)
                * address_counts[duplicate_rows].unsqueeze(-1)
            ).to(torch.long)
            target_cells[duplicate_rows] = replacements
    target_cells = target_cells.sort(dim=-1).values

    offsets = torch.randint(
        address_stride,
        (batch_size, n_windows),
        generator=generator,
    )
    target_positions = target_cells * address_stride + offsets
    target_positions = torch.minimum(target_positions, history_end.unsqueeze(-1) - 1)
    pointers = (target_cells.to(torch.float32) + 0.5) / address_counts.unsqueeze(-1)
    order = torch.rand((batch_size, n_windows), generator=generator).argsort(dim=-1)
    pointers = pointers.gather(1, order)
    return RouterBatch(pointers, address_counts, target_cells, target_positions)


class StandaloneRouter(nn.Module):
    """Permutation-equivariant address-copying gate independent of transformer attention."""

    def __init__(
        self,
        *,
        output: RouterOutput,
        hidden_width: int,
        max_address_count: int,
        radix: int,
    ) -> None:
        super().__init__()
        if max_address_count > radix * radix:
            raise ValueError("two radix digits cannot represent the configured address space")
        self.output_kind = output
        self.max_address_count = max_address_count
        self.radix = radix
        feature_width = 4 + 2 * 6 + 2 * 4
        self.trunk = nn.Sequential(
            nn.Linear(feature_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        if output in {"scalar_cell", "scalar_normalized"}:
            self.output = nn.Linear(hidden_width, 1)
            self.coarse_output = None
            self.fine_output = None
        elif output == "hierarchical":
            self.output = None
            self.coarse_output = nn.Linear(hidden_width, radix)
            self.fine_output = nn.Linear(hidden_width, radix)
        else:
            raise ValueError(f"unknown router output: {output}")

    def _features(self, pointers: Tensor, address_counts: Tensor) -> Tensor:
        counts = address_counts.to(torch.float32).unsqueeze(-1)
        cell_indices = pointers * counts - 0.5
        cell_periods = 2.0 ** torch.arange(1, 7, device=pointers.device)
        cell_angles = 2.0 * torch.pi * cell_indices.unsqueeze(-1) / cell_periods
        address_frequencies = 2.0 ** torch.arange(4, device=pointers.device)
        address_angles = 2.0 * torch.pi * pointers.unsqueeze(-1) * address_frequencies
        return torch.cat(
            (
                pointers.unsqueeze(-1),
                (cell_indices / self.max_address_count).unsqueeze(-1),
                (counts / self.max_address_count).expand_as(pointers).unsqueeze(-1),
                (torch.log2(counts + 1.0) / math.log2(self.max_address_count + 1.0))
                .expand_as(pointers)
                .unsqueeze(-1),
                torch.sin(cell_angles),
                torch.cos(cell_angles),
                torch.sin(address_angles),
                torch.cos(address_angles),
            ),
            dim=-1,
        )

    def forward(self, pointers: Tensor, address_counts: Tensor) -> tuple[Tensor, ...]:
        sorted_pointers = pointers.sort(dim=-1).values
        hidden = self.trunk(self._features(sorted_pointers, address_counts))
        if self.output_kind in {"scalar_cell", "scalar_normalized"}:
            assert self.output is not None
            return (self.output(hidden).squeeze(-1),)
        assert self.coarse_output is not None and self.fine_output is not None
        return self.coarse_output(hidden), self.fine_output(hidden)

    def loss(self, batch: RouterBatch) -> Tensor:
        outputs = self(batch.pointers, batch.address_counts)
        if self.output_kind in {"scalar_cell", "scalar_normalized"}:
            predicted_cells = torch.sigmoid(outputs[0]) * batch.address_counts.unsqueeze(-1)
            if self.output_kind == "scalar_normalized":
                predicted = predicted_cells / batch.address_counts.unsqueeze(-1)
                targets = (
                    batch.target_cells.to(torch.float32) + 0.5
                ) / batch.address_counts.unsqueeze(-1)
                return F.smooth_l1_loss(predicted, targets, beta=0.01)
            return F.smooth_l1_loss(
                predicted_cells,
                batch.target_cells.to(torch.float32) + 0.5,
                beta=0.25,
            )
        coarse, fine = outputs
        return F.cross_entropy(
            coarse.flatten(0, 1),
            torch.div(batch.target_cells, self.radix, rounding_mode="floor").flatten(),
        ) + F.cross_entropy(
            fine.flatten(0, 1),
            (batch.target_cells % self.radix).flatten(),
        )

    def decoded_cells(self, pointers: Tensor, address_counts: Tensor) -> Tensor:
        outputs = self(pointers, address_counts)
        if self.output_kind in {"scalar_cell", "scalar_normalized"}:
            cells = torch.floor(torch.sigmoid(outputs[0]) * address_counts.unsqueeze(-1))
        else:
            coarse, fine = outputs
            cells = coarse.argmax(dim=-1) * self.radix + fine.argmax(dim=-1)
        return torch.minimum(cells.to(torch.long), address_counts.unsqueeze(-1) - 1)


def _evaluate(
    model: StandaloneRouter,
    *,
    context_length: int,
    examples: int,
    batch_size: int,
    n_windows: int,
    dense_window: int,
    address_stride: int,
    window_width: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    exact = 0
    covered = 0
    all_covered = 0
    total = 0
    with torch.no_grad():
        while total < examples:
            current_batch = min(batch_size, examples - total)
            batch = make_router_batch(
                batch_size=current_batch,
                n_windows=n_windows,
                context_lengths=(context_length,),
                dense_window=dense_window,
                address_stride=address_stride,
                generator=generator,
            ).to(device)
            predicted = model.decoded_cells(batch.pointers, batch.address_counts)
            predicted = predicted.sort(dim=-1).values
            exact += (predicted == batch.target_cells).sum().item()
            starts = predicted.unsqueeze(-1) * address_stride
            targets = batch.target_positions.unsqueeze(-2)
            contains = (starts <= targets) & (targets < starts + window_width)
            target_covered = contains.any(dim=-2)
            covered += target_covered.sum().item()
            all_covered += target_covered.all(dim=-1).sum().item()
            total += current_batch
    return {
        "context_length": context_length,
        "examples": total,
        "exact_cell_accuracy": exact / (total * n_windows),
        "target_coverage": covered / (total * n_windows),
        "all_targets_covered": all_covered / total,
    }


def _train_model(
    specification: dict[str, Any],
    output_kind: RouterOutput,
    device: torch.device,
) -> dict[str, Any]:
    seed_offset = {
        "scalar_cell": 0,
        "scalar_normalized": 1,
        "hierarchical": 2,
    }[output_kind]
    seed = specification["seed"] + seed_offset
    seed_everything(seed)
    dense_window = specification["dense_window"]
    address_stride = specification["address_stride"]
    maximum_context = max(specification["train_context_lengths"])
    max_address_count = math.ceil((maximum_context - dense_window) / address_stride)
    model = StandaloneRouter(
        output=output_kind,
        hidden_width=specification["hidden_width"],
        max_address_count=max_address_count,
        radix=specification["radix"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=specification["learning_rate"],
        weight_decay=specification["weight_decay"],
    )
    generator = torch.Generator().manual_seed(seed + 100)
    history = []
    started = time.perf_counter()
    for step in range(1, specification["steps"] + 1):
        model.train()
        batch = make_router_batch(
            batch_size=specification["batch_size"],
            n_windows=specification["n_windows"],
            context_lengths=tuple(specification["train_context_lengths"]),
            dense_window=dense_window,
            address_stride=address_stride,
            generator=generator,
        ).to(device)
        should_log = (
            step == 1 or step % specification["log_every"] == 0 or step == specification["steps"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(batch)
        loss.backward()
        raw_gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            specification["gradient_clip"],
        )
        before_step = (
            [parameter.detach().clone() for parameter in model.parameters()] if should_log else []
        )
        optimizer.step()
        if should_log:
            raw_gradient = float(raw_gradient_norm)
            update_norm = math.sqrt(
                sum(
                    (parameter.detach() - before).square().sum().item()
                    for parameter, before in zip(model.parameters(), before_step, strict=True)
                )
            )
            entry = {
                "step": step,
                "loss": loss.item(),
                "raw_gradient_norm": raw_gradient,
                "gradient_clip_ratio": min(
                    1.0,
                    specification["gradient_clip"] / max(raw_gradient, 1e-12),
                ),
                "parameter_update_norm": update_norm,
            }
            if output_kind in {"scalar_cell", "scalar_normalized"}:
                with torch.no_grad():
                    logits = model(batch.pointers, batch.address_counts)[0]
                    addresses = torch.sigmoid(logits)
                entry["logit_absolute_mean"] = logits.abs().mean().item()
                entry["logit_absolute_max"] = logits.abs().max().item()
                entry["logit_saturation_fraction"] = (logits.abs() > 4.0).float().mean().item()
                entry["address_saturation_fraction"] = (
                    ((addresses < 0.01) | (addresses > 0.99)).float().mean().item()
                )
            history.append(entry)
            print(
                f"router-{output_kind}: step={step}/{specification['steps']} "
                f"loss={loss.item():.6f} grad={float(raw_gradient_norm):.4f}",
                flush=True,
            )
    evaluations = {
        str(context_length): _evaluate(
            model,
            context_length=context_length,
            examples=specification["eval_examples"],
            batch_size=specification["eval_batch_size"],
            n_windows=specification["n_windows"],
            dense_window=dense_window,
            address_stride=address_stride,
            window_width=specification["window_width"],
            seed=seed + 10_000 + context_length,
            device=device,
        )
        for context_length in specification["eval_context_lengths"]
    }
    return {
        "output": output_kind,
        "seed": seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "evaluations": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train standalone multi-address router gates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mode",
        choices=("scalar_cell", "scalar_normalized", "hierarchical", "all"),
        default="all",
    )
    arguments = parser.parse_args()
    specification = load_json(arguments.config)
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    modes: tuple[RouterOutput, ...] = (
        ("scalar_cell", "scalar_normalized", "hierarchical")
        if arguments.mode == "all"
        else (arguments.mode,)
    )
    result = {
        "specification": specification,
        "environment": environment(),
        "device": str(device),
        "models": {mode: _train_model(specification, mode, device) for mode in modes},
    }
    write_json(Path(arguments.output), result)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
