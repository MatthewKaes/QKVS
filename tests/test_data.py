import torch

from qkvs.data import (
    MEMORY,
    QUERY,
    RECORD,
    make_memory_anchor_batch,
    make_memory_bank_batch,
    make_multi_region_batch,
    make_multi_region_counterfactual_pair,
    make_pointer_batch,
    make_semantic_batch,
    make_span_batch,
    semantic_vocabulary_size,
)


def test_pointer_chain_targets_match_embedded_pointers() -> None:
    batch = make_pointer_batch(
        batch_size=4,
        seq_len=24,
        vocab_size=16,
        dense_window=6,
        block_size=1,
        n_layers=4,
        hops=3,
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.all(batch.token_types[:, -1] == QUERY)
    for item in range(4):
        positions = batch.route_positions[:, item]
        assert torch.all(batch.token_types[item, positions[:3]] == RECORD)
        torch.testing.assert_close(batch.pointers[item, -1], batch.route_addresses[0, item])
        torch.testing.assert_close(
            batch.pointers[item, positions[0]], batch.route_addresses[1, item]
        )
        torch.testing.assert_close(
            batch.pointers[item, positions[1]], batch.route_addresses[2, item]
        )
        assert batch.answers[item] == batch.values[item, positions[2]]


def test_semantic_batch_hides_addresses_but_encodes_matching_records() -> None:
    n_keys = 4
    answer_vocab = 8
    batch = make_semantic_batch(
        batch_size=3,
        seq_len=24,
        dense_window=4,
        block_size=1,
        n_layers=3,
        n_keys=n_keys,
        answer_vocab_size=answer_vocab,
        fixed_layout=False,
        generator=torch.Generator().manual_seed(31),
    )
    query_base = answer_vocab + n_keys * answer_vocab
    for item in range(3):
        query_key = batch.values[item, -1] - query_base
        position = batch.route_positions[0, item]
        record_token = batch.values[item, position]
        encoded_key = (record_token - answer_vocab) // answer_vocab
        encoded_answer = (record_token - answer_vocab) % answer_vocab
        assert encoded_key == query_key
        assert encoded_answer == batch.answers[item]
        assert batch.pointers[item, -1] == -1
        assert batch.token_types[item, position] == RECORD
    assert batch.values.max() < semantic_vocabulary_size(n_keys, answer_vocab)


def test_fixed_semantic_layout_preserves_normalized_key_address_across_lengths() -> None:
    short = make_semantic_batch(
        batch_size=1,
        seq_len=32,
        dense_window=1,
        block_size=1,
        n_layers=2,
        n_keys=4,
        answer_vocab_size=8,
        fixed_layout=True,
        generator=torch.Generator().manual_seed(2),
    )
    long = make_semantic_batch(
        batch_size=1,
        seq_len=128,
        dense_window=1,
        block_size=1,
        n_layers=2,
        n_keys=4,
        answer_vocab_size=8,
        fixed_layout=True,
        generator=torch.Generator().manual_seed(2),
    )

    def key_addresses(batch, history_end: int) -> torch.Tensor:
        positions = torch.nonzero(batch.token_types[0] == RECORD).flatten()
        tokens = batch.values[0, positions]
        keys = (tokens - 8) // 8
        addresses = (positions.float() + 0.5) / history_end
        result = torch.empty(4)
        result[keys] = addresses
        return result

    torch.testing.assert_close(
        key_addresses(short, 31), key_addresses(long, 127), atol=0.01, rtol=0.0
    )


def test_span_answer_is_marked_token_after_addressed_start() -> None:
    batch = make_span_batch(
        batch_size=8,
        seq_len=32,
        vocab_size=16,
        dense_window=1,
        block_size=4,
        address_stride=1,
        n_layers=3,
        generator=torch.Generator().manual_seed(73),
    )
    for item in range(8):
        start = batch.route_positions[0, item]
        assert batch.answers[item] == batch.values[item, start + 1]
        assert batch.token_types[item, start + 1] == RECORD


def test_memory_anchor_has_no_pointer_and_source_is_in_configured_window() -> None:
    batch = make_memory_anchor_batch(
        batch_size=8,
        seq_len=64,
        vocab_size=32,
        dense_window=8,
        address_stride=1,
        n_layers=4,
        source_window=8,
        generator=torch.Generator().manual_seed(81),
    )
    marker = 32
    assert torch.all(batch.token_types[:, marker] == MEMORY)
    assert torch.all(batch.pointers[:, marker] == -1)
    for item in range(8):
        sources = torch.nonzero(batch.token_types[item] == RECORD).flatten()
        assert sources.numel() == 1
        source = sources.item()
        assert marker - 8 < source < marker
        assert batch.answers[item] == batch.values[item, source]


def test_memory_anchor_supports_exact_non_leaking_source_offset() -> None:
    batch = make_memory_anchor_batch(
        batch_size=4,
        seq_len=512,
        vocab_size=16,
        dense_window=64,
        address_stride=16,
        n_layers=4,
        source_window=64,
        source_offset=32,
        generator=torch.Generator().manual_seed(82),
    )
    marker = 256
    source = marker - 32
    assert batch.memory_targets is not None
    assert torch.all(batch.token_types[:, source] == RECORD)
    assert torch.all(batch.memory_targets[:, marker] == batch.answers)
    assert source < marker - 16


def test_scaled_memory_bank_is_beyond_rolling_receptive_field() -> None:
    batch = make_memory_bank_batch(
        batch_size=2,
        seq_len=1024,
        dense_window=128,
        address_stride=4,
        n_layers=4,
        n_keys=4,
        answer_vocab_size=16,
        distractors_per_marker=3,
        source_offset=32,
        generator=torch.Generator().manual_seed(91),
    )
    effective_receptive_field = 1 + 4 * (128 - 1)
    for item in range(2):
        assert batch.semantic_ids is not None
        assert batch.memory_targets is not None
        markers = torch.nonzero(batch.token_types[item] == MEMORY).flatten()
        assert markers.numel() == 4
        assert 1023 - markers.max().item() > effective_receptive_field
        assert torch.all(batch.pointers[item] == -1)
        assert batch.token_types[item, -1] == QUERY
        assert batch.route_positions[0, item] in markers
        assert batch.values[item].max() < 16
        assert torch.all(batch.semantic_ids[item, markers] >= 0)
        assert torch.all(batch.memory_targets[item, markers] >= 0)
        assert torch.all(batch.memory_targets[item, batch.token_types[item] != MEMORY] == -100)
        for marker in markers:
            source = marker - 32
            assert batch.token_types[item, source] == RECORD
            assert source < marker - 16


def test_multi_region_answers_require_balanced_requested_records() -> None:
    batch = make_multi_region_batch(
        batch_size=16,
        seq_len=512,
        dense_window=32,
        block_size=32,
        address_stride=8,
        n_windows=8,
        n_layers=4,
        n_records=24,
        required_regions=8,
        generator=torch.Generator().manual_seed(93),
    )
    assert batch.route_target_positions is not None
    assert batch.route_target_addresses is not None
    assert batch.route_target_positions.shape == (4, 16, 8)
    assert batch.route_target_addresses.shape == (4, 16, 8)
    rolling_receptive_field = 1 + 4 * (32 - 1)
    assert torch.all(batch.route_target_positions < 511 - rolling_receptive_field)
    for item in range(16):
        targets = batch.route_target_positions[0, item]
        assert batch.values[item, targets].sum() == batch.answers[item]
        assert torch.all(batch.token_types[item, targets] == RECORD)
        records = torch.nonzero(batch.token_types[item] == RECORD).flatten()
        assert records.numel() == 24
        assert batch.values[item, records].sum() == 12
        assert batch.values[item].sum() > batch.answers[item]
        expected_record_addresses = (
            torch.div(records, 8, rounding_mode="floor").float() + 0.5
        ) / 60
        torch.testing.assert_close(batch.pointers[item, records, 0], expected_record_addresses)
        assert torch.all(batch.pointers[item, records, 1:] == -1)
        expected_pointers = (torch.div(targets, 8, rounding_mode="floor").float() + 0.5) / 60
        torch.testing.assert_close(batch.pointers[item, -1], expected_pointers)
    assert 0 <= batch.answers.min() <= batch.answers.max() <= 8
    assert torch.unique(batch.answers).numel() > 4


def test_multi_region_batch_supports_zero_sparse_windows_for_controls() -> None:
    batch = make_multi_region_batch(
        batch_size=2,
        seq_len=256,
        dense_window=16,
        block_size=32,
        address_stride=16,
        n_windows=0,
        n_layers=2,
        n_records=12,
        required_regions=4,
        generator=torch.Generator().manual_seed(94),
    )
    assert batch.route_target_addresses is not None
    assert batch.route_target_addresses.shape == (2, 2, 0)
    assert batch.route_addresses.shape == (2, 2)


def test_multi_region_counterfactual_changes_only_the_requested_pointer_set() -> None:
    first, second = make_multi_region_counterfactual_pair(
        batch_size=8,
        seq_len=512,
        dense_window=32,
        block_size=32,
        address_stride=8,
        n_windows=8,
        n_layers=4,
        n_records=24,
        required_regions=8,
        generator=torch.Generator().manual_seed(95),
    )
    torch.testing.assert_close(first.values, second.values)
    torch.testing.assert_close(first.token_types, second.token_types)
    torch.testing.assert_close(first.pointers[:, :-1], second.pointers[:, :-1])
    assert torch.all(first.answers != second.answers)
    assert torch.all((first.pointers[:, -1] != second.pointers[:, -1]).any(dim=1))
    for batch in (first, second):
        assert batch.route_target_positions is not None
        selected = batch.route_target_positions[0]
        torch.testing.assert_close(batch.values.gather(1, selected).sum(dim=1), batch.answers)


def test_multi_region_rejects_ambiguous_candidate_addresses() -> None:
    try:
        make_multi_region_batch(
            batch_size=1,
            seq_len=128,
            dense_window=16,
            block_size=32,
            address_stride=16,
            n_windows=0,
            n_layers=2,
            n_records=12,
            required_regions=4,
            generator=torch.Generator().manual_seed(96),
        )
    except ValueError as error:
        assert "distinct pointer addresses" in str(error)
    else:
        raise AssertionError("ambiguous record addresses should be rejected")
