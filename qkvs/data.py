from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from qkvs.attention import normalized_block_centers

FILLER = 0
RECORD = 1
QUERY = 2


@dataclass(frozen=True)
class PointerBatch:
    values: Tensor
    pointers: Tensor
    token_types: Tensor
    answers: Tensor
    route_addresses: Tensor  # [layers, batch]
    route_positions: Tensor  # [layers, batch]
    hops: int

    def to(self, device: torch.device | str) -> PointerBatch:
        return PointerBatch(
            values=self.values.to(device),
            pointers=self.pointers.to(device),
            token_types=self.token_types.to(device),
            answers=self.answers.to(device),
            route_addresses=self.route_addresses.to(device),
            route_positions=self.route_positions.to(device),
            hops=self.hops,
        )


def make_pointer_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dense_window: int,
    block_size: int,
    n_layers: int,
    hops: int,
    generator: torch.Generator,
) -> PointerBatch:
    """Generate pointer chains whose records are all outside the rolling window."""
    if not 1 <= hops <= n_layers:
        raise ValueError("hops must be between one and the layer count")
    history_end = seq_len - dense_window
    if history_end < hops:
        raise ValueError("sequence does not contain enough historical positions")

    values = torch.randint(vocab_size, (batch_size, seq_len), generator=generator)
    pointers = torch.full((batch_size, seq_len), -1.0)
    token_types = torch.full((batch_size, seq_len), FILLER, dtype=torch.long)
    route_positions = torch.empty((n_layers, batch_size), dtype=torch.long)
    answers = torch.empty(batch_size, dtype=torch.long)

    for batch_index in range(batch_size):
        permutation = torch.randperm(history_end, generator=generator)
        chain = permutation[:hops]
        token_types[batch_index, chain] = RECORD
        for hop_index in range(hops - 1):
            next_address = normalized_block_centers(chain[hop_index + 1], history_end, block_size)
            pointers[batch_index, chain[hop_index]] = next_address
        answers[batch_index] = values[batch_index, chain[-1]]
        start_address = normalized_block_centers(chain[0], history_end, block_size)
        pointers[batch_index, -1] = start_address
        token_types[batch_index, -1] = QUERY
        for layer_index in range(n_layers):
            route_positions[layer_index, batch_index] = chain[min(layer_index, hops - 1)]

    route_addresses = normalized_block_centers(route_positions, history_end, block_size)
    return PointerBatch(
        values=values,
        pointers=pointers,
        token_types=token_types,
        answers=answers,
        route_addresses=route_addresses,
        route_positions=route_positions,
        hops=hops,
    )


def semantic_vocabulary_size(n_keys: int, answer_vocab_size: int) -> int:
    """Vocabulary containing answers/fillers, key-answer records, and key queries."""
    return answer_vocab_size + n_keys * answer_vocab_size + n_keys


def make_semantic_batch(
    *,
    batch_size: int,
    seq_len: int,
    dense_window: int,
    block_size: int,
    n_layers: int,
    n_keys: int,
    answer_vocab_size: int,
    fixed_layout: bool,
    generator: torch.Generator,
) -> PointerBatch:
    """Generate key-value lookup without exposing a numerical address to the query.

    Random layout makes the address information-theoretically unavailable to a direct router;
    fixed layout tests whether a semantic key can learn a stable direct-address map.
    """
    history_end = seq_len - dense_window
    if history_end < n_keys:
        raise ValueError("sequence does not contain enough historical positions for all keys")
    vocab_size = semantic_vocabulary_size(n_keys, answer_vocab_size)
    values = torch.randint(answer_vocab_size, (batch_size, seq_len), generator=generator)
    pointers = torch.full((batch_size, seq_len), -1.0)
    token_types = torch.full((batch_size, seq_len), FILLER, dtype=torch.long)
    route_positions = torch.empty((n_layers, batch_size), dtype=torch.long)
    answers = torch.empty(batch_size, dtype=torch.long)
    record_base = answer_vocab_size
    query_base = answer_vocab_size + n_keys * answer_vocab_size

    fixed_positions = torch.div(
        (torch.arange(n_keys) * 2 + 1) * history_end,
        2 * n_keys,
        rounding_mode="floor",
    )
    for batch_index in range(batch_size):
        positions = (
            fixed_positions
            if fixed_layout
            else torch.randperm(history_end, generator=generator)[:n_keys]
        )
        record_answers = torch.randint(answer_vocab_size, (n_keys,), generator=generator)
        keys = torch.arange(n_keys)
        record_tokens = record_base + keys * answer_vocab_size + record_answers
        values[batch_index, positions] = record_tokens
        token_types[batch_index, positions] = RECORD
        query_key = torch.randint(n_keys, (), generator=generator)
        values[batch_index, -1] = query_base + query_key
        token_types[batch_index, -1] = QUERY
        answers[batch_index] = record_answers[query_key]
        route_positions[:, batch_index] = positions[query_key]

    if values.max().item() >= vocab_size:
        raise AssertionError("semantic token encoding exceeded its vocabulary")
    route_addresses = normalized_block_centers(route_positions, history_end, block_size)
    return PointerBatch(
        values=values,
        pointers=pointers,
        token_types=token_types,
        answers=answers,
        route_addresses=route_addresses,
        route_positions=route_positions,
        hops=1,
    )


def make_address_overrides(
    batch: PointerBatch,
    *,
    seq_len: int,
    n_windows: int,
    device: torch.device,
) -> Tensor:
    """Teacher-route only the first window of the final query at each layer."""
    layers, batch_size = batch.route_addresses.shape
    overrides = torch.full(
        (layers, batch_size, seq_len, n_windows),
        torch.nan,
        device=device,
    )
    overrides[:, :, -1, 0] = batch.route_addresses.to(device)
    return overrides
