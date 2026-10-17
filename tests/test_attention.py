import torch

from qkvs.attention import (
    CausalSelfAttention,
    Selection,
    build_selection,
    build_soft_neighbor_selection,
    chunked_selected_attention,
    duplicate_attention_reference,
    log_multiplicity_reference,
    router_coordinate_features,
    selected_attention,
    streaming_selected_attention,
)
from qkvs.config import AttentionConfig


def test_full_attention_selection_below_threshold() -> None:
    addresses = torch.rand(2, 8, 2)
    selection = build_selection(addresses, dense_window=4, retrieval_budget=4, block_size=2)
    for query in range(8):
        actual = selection.indices[0, query][selection.valid[0, query]]
        torch.testing.assert_close(actual, torch.arange(query + 1))


def test_selected_attention_matches_dense_math_below_threshold() -> None:
    generator = torch.Generator().manual_seed(41)
    q = torch.randn(1, 2, 6, 4, generator=generator)
    k = torch.randn(1, 2, 6, 4, generator=generator)
    v = torch.randn(1, 2, 6, 4, generator=generator)
    selection = build_selection(
        torch.rand(1, 6, 2, generator=generator),
        dense_window=4,
        retrieval_budget=2,
        block_size=1,
    )
    actual = selected_attention(q, k, v, selection)
    expected = torch.empty_like(actual)
    for position in range(6):
        logits = torch.einsum("bhd,bhtd->bht", q[:, :, position], k[:, :, : position + 1]) / 2.0
        expected[:, :, position] = torch.einsum(
            "bht,bhtd->bhd",
            torch.softmax(logits, dim=-1),
            v[:, :, : position + 1],
        )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_direct_block_addressing_and_recent_window() -> None:
    addresses = torch.full((1, 12, 2), 0.5)
    addresses[0, -1] = torch.tensor([0.01, 0.76])
    selection = build_selection(addresses, dense_window=4, retrieval_budget=4, block_size=2)
    assert selection.indices[0, -1].tolist() == [0, 1, 6, 7, 8, 9, 10, 11]
    assert selection.valid[0, -1].all()


def test_selection_is_causal_and_bounded() -> None:
    addresses = torch.rand(3, 31, 4)
    selection = build_selection(addresses, dense_window=7, retrieval_budget=8, block_size=2)
    assert selection.indices.shape == (3, 31, 15)
    positions = torch.arange(31).view(1, 31, 1)
    assert torch.all(
        selection.indices[selection.valid]
        <= positions.expand_as(selection.indices)[selection.valid]
    )


def test_overlapping_windows_preserve_duplicate_occurrences() -> None:
    addresses = torch.full((1, 20, 2), 0.2)
    selection = build_selection(addresses, dense_window=4, retrieval_budget=4, block_size=2)
    assert selection.indices[0, -1, :2].tolist() == selection.indices[0, -1, 2:4].tolist()


def test_address_stride_allows_partial_window_overlap() -> None:
    addresses = torch.full((1, 13, 2), 0.5)
    addresses[0, -1] = torch.tensor([0.21, 0.41])
    selection = build_selection(
        addresses,
        dense_window=4,
        retrieval_budget=8,
        block_size=4,
        address_stride=2,
    )
    assert selection.indices[0, -1].tolist() == [
        2,
        3,
        4,
        5,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
    ]


def test_duplicate_occurrences_equal_log_multiplicity_bias() -> None:
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(8, generator=generator)
    k = torch.randn(6, 8, generator=generator)
    v = torch.randn(6, 5, generator=generator)
    indices = torch.tensor([0, 2, 2, 2, 5, 5])
    duplicated = duplicate_attention_reference(q, k, v, indices)
    deduplicated = log_multiplicity_reference(q, k, v, indices)
    torch.testing.assert_close(duplicated, deduplicated, rtol=1e-6, atol=1e-6)


def test_streaming_attention_matches_materialized_forward_and_backward() -> None:
    generator = torch.Generator().manual_seed(101)
    addresses = torch.rand(2, 13, 2, generator=generator)
    selection = build_selection(addresses, dense_window=4, retrieval_budget=4, block_size=2)
    materialized_inputs = [
        torch.randn(2, 2, 13, 4, generator=generator, requires_grad=True) for _ in range(3)
    ]
    streaming_inputs = [value.detach().clone().requires_grad_() for value in materialized_inputs]
    expected = selected_attention(*materialized_inputs, selection)
    actual = streaming_selected_attention(
        *streaming_inputs,
        selection,
        query_chunk_size=3,
        candidate_tile_size=2,
        checkpoint_chunks=True,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for expected_input, actual_input in zip(materialized_inputs, streaming_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=2e-5, atol=2e-6)


def test_chunked_sdpa_matches_materialized_forward_and_backward() -> None:
    generator = torch.Generator().manual_seed(102)
    addresses = torch.rand(2, 13, 2, generator=generator)
    selection = build_selection(addresses, dense_window=4, retrieval_budget=4, block_size=2)
    materialized_inputs = [
        torch.randn(2, 2, 13, 4, generator=generator, requires_grad=True) for _ in range(3)
    ]
    chunked_inputs = [value.detach().clone().requires_grad_() for value in materialized_inputs]
    expected = selected_attention(*materialized_inputs, selection)
    actual = chunked_selected_attention(
        *chunked_inputs,
        selection,
        query_chunk_size=3,
        checkpoint_chunks=True,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for expected_input, actual_input in zip(materialized_inputs, chunked_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=2e-5, atol=2e-6)


def test_selected_attention_bias_is_differentiable_across_backends() -> None:
    generator = torch.Generator().manual_seed(103)
    addresses = torch.rand(2, 13, 2, generator=generator)
    base = build_selection(addresses, dense_window=4, retrieval_budget=4, block_size=2)
    first_bias = torch.randn(base.indices.shape, generator=generator, requires_grad=True)
    biases = [first_bias, first_bias.detach().clone().requires_grad_()]
    inputs = [torch.randn(2, 2, 13, 4, generator=generator) for _ in range(3)]
    expected = selected_attention(*inputs, Selection(base.indices, base.valid, biases[0]))
    actual = chunked_selected_attention(
        *inputs,
        Selection(base.indices, base.valid, biases[1]),
        query_chunk_size=3,
        checkpoint_chunks=True,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert biases[0].grad is not None and biases[1].grad is not None
    torch.testing.assert_close(biases[1].grad, biases[0].grad, rtol=2e-5, atol=2e-6)


def test_soft_neighbor_selection_has_bounded_adjacent_windows_and_offset_gradient() -> None:
    addresses = torch.full((1, 10, 1), 0.5)
    offsets = torch.full((1, 10, 1), 0.25, requires_grad=True)
    selection = build_soft_neighbor_selection(
        addresses,
        offsets,
        dense_window=4,
        retrieval_budget=2,
        block_size=2,
        address_stride=2,
    )
    assert selection.indices.shape == (1, 10, 8)
    assert selection.logit_bias is not None
    torch.testing.assert_close(
        selection.logit_bias[0, -1, :4].exp(),
        torch.tensor([0.75, 0.75, 0.25, 0.25]),
    )
    q = torch.randn(1, 2, 10, 4)
    k = torch.randn(1, 2, 10, 4)
    v = torch.randn(1, 2, 10, 4)
    selected_attention(q, k, v, selection).square().sum().backward()
    assert offsets.grad is not None and offsets.grad.abs().sum() > 0


def test_router_receives_auxiliary_gradient() -> None:
    torch.manual_seed(3)
    attention = CausalSelfAttention(
        AttentionConfig(
            d_model=16,
            n_heads=4,
            dense_window=4,
            retrieval_budget=2,
            block_size=1,
        )
    )
    hidden = torch.randn(2, 10, 16)
    output = attention(hidden)
    assert output.addresses is not None
    loss = (output.addresses[:, -1, 0] - torch.tensor([0.2, 0.8])).square().mean()
    loss.backward()
    assert attention.s_proj is not None
    assert attention.s_proj.weight.grad is not None
    assert attention.s_proj.weight.grad.abs().sum() > 0


def test_detached_router_trains_s_without_changing_its_input() -> None:
    torch.manual_seed(4)
    attention = CausalSelfAttention(
        AttentionConfig(
            d_model=16,
            n_heads=4,
            dense_window=4,
            retrieval_budget=2,
            block_size=1,
            router_detach_input=True,
        )
    )
    hidden = torch.randn(2, 10, 16, requires_grad=True)
    addresses = attention._addresses(hidden)
    assert addresses is not None
    addresses.square().sum().backward()
    assert hidden.grad is None
    assert attention.s_proj is not None
    assert attention.s_proj.weight.grad is not None
    assert attention.s_proj.weight.grad.abs().sum() > 0


def test_router_coordinate_features_expose_position_and_address_scale() -> None:
    positions = torch.tensor([255, 511, 1023])
    features = router_coordinate_features(
        positions,
        dense_window=128,
        address_stride=16,
    )
    assert features.shape == (3, 20)
    assert torch.unique(features, dim=0).shape[0] == 3


def test_router_position_channel_trains_while_hidden_input_is_detached() -> None:
    torch.manual_seed(5)
    attention = CausalSelfAttention(
        AttentionConfig(
            d_model=16,
            n_heads=4,
            dense_window=4,
            retrieval_budget=2,
            block_size=1,
            router_detach_input=True,
            router_position_encoding="absolute_fourier",
        )
    )
    hidden = torch.randn(2, 10, 16, requires_grad=True)
    addresses = attention._addresses(hidden)
    assert addresses is not None
    addresses.square().sum().backward()
    assert hidden.grad is None
    assert attention.router_position_projection is not None
    gradients = [parameter.grad for parameter in attention.router_position_projection.parameters()]
    assert all(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)


def test_hierarchical_router_decodes_coarse_and_fine_address_digits() -> None:
    attention = CausalSelfAttention(
        AttentionConfig(
            d_model=16,
            n_heads=4,
            dense_window=4,
            retrieval_budget=1,
            block_size=1,
            router_output_encoding="hierarchical",
            router_radix=4,
        )
    )
    projection = attention.router_output_projection()
    assert projection is not None
    with torch.no_grad():
        projection.weight.zero_()
        projection.bias.zero_()
        projection.bias[1] = 10.0
        projection.bias[4 + 2] = 10.0
    hidden = torch.zeros(1, 16, 16)
    addresses = attention._addresses(hidden)
    assert addresses is not None
    torch.testing.assert_close(addresses[0, -1, 0], torch.tensor(6.5 / 12.0))


def test_retrieval_mask_disables_only_sparse_historical_slots() -> None:
    attention = CausalSelfAttention(
        AttentionConfig(
            d_model=16,
            n_heads=4,
            dense_window=4,
            retrieval_budget=2,
            block_size=1,
        )
    ).eval()
    hidden = torch.randn(1, 10, 16)
    output = attention(hidden, retrieval_mask=torch.zeros(1, 10, dtype=torch.bool))
    assert output.selection is not None
    assert output.selection.valid[0, 3, :4].all()
    assert not output.selection.valid[0, -1, :2].any()
    assert output.selection.valid[0, -1, 2:].all()


def test_window_mask_can_disable_unrelated_historical_windows() -> None:
    attention = CausalSelfAttention(
        AttentionConfig(
            d_model=16,
            n_heads=4,
            dense_window=4,
            retrieval_budget=4,
            block_size=2,
        )
    ).eval()
    hidden = torch.randn(1, 10, 16)
    window_mask = torch.ones(1, 10, 2, dtype=torch.bool)
    window_mask[:, -1, 1] = False
    output = attention(hidden, historical_window_mask=window_mask)
    assert output.selection is not None
    assert output.selection.valid[0, -1, :2].all()
    assert not output.selection.valid[0, -1, 2:4].any()
    assert output.selection.valid[0, -1, 4:].all()
