from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from qkvs.attention import normalized_window_addresses

FILLER = 0
RECORD = 1
QUERY = 2
MEMORY = 3


@dataclass(frozen=True)
class PointerBatch:
    values: Tensor
    pointers: Tensor
    token_types: Tensor
    answers: Tensor
    route_addresses: Tensor  # [layers, batch]
    route_positions: Tensor  # [layers, batch]
    hops: int
    semantic_ids: Tensor | None = None
    memory_targets: Tensor | None = None

    def to(self, device: torch.device | str) -> PointerBatch:
        return PointerBatch(
            values=self.values.to(device),
            pointers=self.pointers.to(device),
            token_types=self.token_types.to(device),
            answers=self.answers.to(device),
            route_addresses=self.route_addresses.to(device),
            route_positions=self.route_positions.to(device),
            hops=self.hops,
            semantic_ids=(None if self.semantic_ids is None else self.semantic_ids.to(device)),
            memory_targets=(
                None if self.memory_targets is None else self.memory_targets.to(device)
            ),
        )


def make_pointer_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dense_window: int,
    block_size: int,
    address_stride: int | None = None,
    n_layers: int,
    hops: int,
    generator: torch.Generator,
) -> PointerBatch:
    """Generate pointer chains whose records are all outside the rolling window."""
    if not 1 <= hops <= n_layers:
        raise ValueError("hops must be between one and the layer count")
    history_end = seq_len - dense_window
    stride = block_size if address_stride is None else address_stride
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
            next_address = normalized_window_addresses(chain[hop_index + 1], history_end, stride)
            pointers[batch_index, chain[hop_index]] = next_address
        answers[batch_index] = values[batch_index, chain[-1]]
        start_address = normalized_window_addresses(chain[0], history_end, stride)
        pointers[batch_index, -1] = start_address
        token_types[batch_index, -1] = QUERY
        for layer_index in range(n_layers):
            route_positions[layer_index, batch_index] = chain[min(layer_index, hops - 1)]

    route_addresses = normalized_window_addresses(route_positions, history_end, stride)
    return PointerBatch(
        values=values,
        pointers=pointers,
        token_types=token_types,
        answers=answers,
        route_addresses=route_addresses,
        route_positions=route_positions,
        hops=hops,
    )


def make_span_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dense_window: int,
    block_size: int,
    address_stride: int,
    n_layers: int,
    generator: torch.Generator,
) -> PointerBatch:
    """Retrieve a marked token immediately after an addressed span start."""
    history_end = seq_len - dense_window
    if history_end < 2:
        raise ValueError("sequence does not contain a two-token historical span")
    values = torch.randint(vocab_size, (batch_size, seq_len), generator=generator)
    pointers = torch.full((batch_size, seq_len), -1.0)
    token_types = torch.full((batch_size, seq_len), FILLER, dtype=torch.long)
    starts = torch.randint(history_end - 1, (batch_size,), generator=generator)
    answers = torch.empty(batch_size, dtype=torch.long)
    for item, start in enumerate(starts):
        token_types[item, start + 1] = RECORD
        token_types[item, -1] = QUERY
        answers[item] = values[item, start + 1]
        pointers[item, -1] = normalized_window_addresses(start, history_end, address_stride)
    route_positions = starts.unsqueeze(0).expand(n_layers, -1).clone()
    route_addresses = normalized_window_addresses(route_positions, history_end, address_stride)
    return PointerBatch(
        values,
        pointers,
        token_types,
        answers,
        route_addresses,
        route_positions,
        1,
    )


def make_memory_anchor_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dense_window: int,
    address_stride: int,
    n_layers: int,
    source_window: int,
    source_offset: int | None = None,
    generator: torch.Generator,
) -> PointerBatch:
    """Place a record near a pointer-free memory sink retrieved much later."""
    marker = seq_len // 2
    if source_window < 2 or marker < source_window:
        raise ValueError("source window must fit before the memory marker")
    history_end = seq_len - dense_window
    if marker >= history_end:
        raise ValueError("memory marker must be outside the final dense window")
    values = torch.randint(vocab_size, (batch_size, seq_len), generator=generator)
    pointers = torch.full((batch_size, seq_len), -1.0)
    token_types = torch.full((batch_size, seq_len), FILLER, dtype=torch.long)
    answers = torch.empty(batch_size, dtype=torch.long)
    memory_targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    marker_positions = torch.full((n_layers, batch_size), marker, dtype=torch.long)
    for item in range(batch_size):
        distance = (
            torch.randint(1, source_window, (), generator=generator)
            if source_offset is None
            else source_offset
        )
        if not 0 < distance < source_window:
            raise ValueError("source offset must fall inside the dense source window")
        source = marker - distance
        token_types[item, source] = RECORD
        token_types[item, marker] = MEMORY
        token_types[item, -1] = QUERY
        values[item, marker] = 0
        values[item, -1] = 0
        answers[item] = values[item, source]
        memory_targets[item, marker] = answers[item]
    route_addresses = normalized_window_addresses(marker_positions, history_end, address_stride)
    return PointerBatch(
        values,
        pointers,
        token_types,
        answers,
        route_addresses,
        marker_positions,
        1,
        memory_targets=memory_targets,
    )


def semantic_vocabulary_size(n_keys: int, answer_vocab_size: int) -> int:
    """Vocabulary containing answers/fillers, key-answer records, and key queries."""
    return answer_vocab_size + n_keys * answer_vocab_size + n_keys


def memory_bank_vocabulary_size(n_keys: int, answer_vocab_size: int) -> int:
    """Answers/fillers, key-value records, key markers, and key queries."""
    return answer_vocab_size + n_keys * answer_vocab_size + 2 * n_keys


def make_memory_bank_batch(
    *,
    batch_size: int,
    seq_len: int,
    dense_window: int,
    address_stride: int,
    n_layers: int,
    n_keys: int,
    answer_vocab_size: int,
    distractors_per_marker: int,
    source_offset: int,
    generator: torch.Generator,
) -> PointerBatch:
    """Build several pointer-free memory sinks with competing local records."""
    history_end = seq_len - dense_window
    values = torch.randint(answer_vocab_size, (batch_size, seq_len), generator=generator)
    pointers = torch.full((batch_size, seq_len), -1.0)
    token_types = torch.full((batch_size, seq_len), FILLER, dtype=torch.long)
    answers = torch.empty(batch_size, dtype=torch.long)
    route_positions = torch.empty((n_layers, batch_size), dtype=torch.long)
    semantic_ids = torch.full((batch_size, seq_len), -1, dtype=torch.long)
    memory_targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    fractions = torch.linspace(0.30, 0.45, n_keys)
    marker_positions = torch.floor(fractions * history_end).to(torch.long)
    minimum_spacing = int(torch.diff(marker_positions).min().item())
    if minimum_spacing <= distractors_per_marker + 2:
        raise ValueError("memory markers are too close for their distractor regions")
    if source_offset <= distractors_per_marker + 1:
        raise ValueError("source offset must leave room for distractors")

    for item in range(batch_size):
        record_answers = torch.randint(answer_vocab_size, (n_keys,), generator=generator)
        for key, marker in enumerate(marker_positions):
            source = marker - source_offset
            values[item, source] = record_answers[key]
            token_types[item, source] = RECORD
            semantic_ids[item, source] = key
            for distractor in range(distractors_per_marker):
                position = source - 1 - distractor
                distractor_key = (key + distractor + 1) % n_keys
                distractor_answer = torch.randint(answer_vocab_size, (), generator=generator)
                values[item, position] = distractor_answer
                token_types[item, position] = RECORD
                semantic_ids[item, position] = distractor_key
            values[item, marker] = 0
            token_types[item, marker] = MEMORY
            semantic_ids[item, marker] = key
            memory_targets[item, marker] = record_answers[key]

        query_key = torch.randint(n_keys, (), generator=generator)
        values[item, -1] = 0
        token_types[item, -1] = QUERY
        semantic_ids[item, -1] = query_key
        answers[item] = record_answers[query_key]
        route_positions[:, item] = marker_positions[query_key]

    route_addresses = normalized_window_addresses(route_positions, history_end, address_stride)
    return PointerBatch(
        values,
        pointers,
        token_types,
        answers,
        route_addresses,
        route_positions,
        1,
        semantic_ids,
        memory_targets,
    )


def make_semantic_batch(
    *,
    batch_size: int,
    seq_len: int,
    dense_window: int,
    block_size: int,
    address_stride: int | None = None,
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
    stride = block_size if address_stride is None else address_stride
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
    route_addresses = normalized_window_addresses(route_positions, history_end, stride)
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


def make_oracle_window_masks(
    batch: PointerBatch,
    *,
    seq_len: int,
    n_windows: int,
    device: torch.device,
) -> Tensor:
    """Enable only the oracle-addressed historical window for the final query."""
    layers, batch_size = batch.route_addresses.shape
    masks = torch.ones(
        (layers, batch_size, seq_len, n_windows),
        dtype=torch.bool,
        device=device,
    )
    masks[:, :, -1] = False
    masks[:, :, -1, 0] = True
    return masks
