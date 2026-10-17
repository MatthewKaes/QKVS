import torch

from qkvs.config import ModelConfig
from qkvs.model import QKVSDecoder


def _inputs(seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.randint(0, 16, (2, seq_len), generator=torch.Generator().manual_seed(4))
    pointers = torch.full((2, seq_len), -1.0)
    pointers[:, -1] = torch.tensor([0.2, 0.7])
    token_types = torch.zeros((2, seq_len), dtype=torch.long)
    token_types[:, -1] = 2
    return values, pointers, token_types


def test_each_layer_has_independent_s_projection() -> None:
    model = QKVSDecoder(
        ModelConfig(
            vocab_size=16,
            d_model=32,
            n_heads=4,
            n_layers=4,
            dense_window=4,
            retrieval_budget=2,
            block_size=1,
        )
    )
    weights = [layer.attention.s_proj.weight for layer in model.layers]
    assert len({weight.data_ptr() for weight in weights}) == 4


def test_incremental_decode_matches_batched_reference() -> None:
    torch.manual_seed(12)
    model = QKVSDecoder(
        ModelConfig(
            vocab_size=16,
            d_model=32,
            n_heads=4,
            n_layers=3,
            dense_window=4,
            retrieval_budget=2,
            block_size=1,
            dropout=0.0,
        )
    ).eval()
    values, pointers, token_types = _inputs(11)
    with torch.no_grad():
        batched = model(values, pointers, token_types).logits
        cache = model.empty_cache(max_length=values.shape[1])
        steps = []
        for position in range(values.shape[1]):
            output, cache = model.step(
                values[:, position],
                pointers[:, position],
                token_types[:, position],
                cache,
            )
            steps.append(output.logits)
        incremental = torch.cat(steps, dim=1)
    torch.testing.assert_close(batched, incremental, rtol=2e-5, atol=2e-5)


def test_unrotated_address_matching_incremental_decode_matches_batched() -> None:
    torch.manual_seed(13)
    model = QKVSDecoder(
        ModelConfig(
            vocab_size=16,
            d_model=32,
            n_heads=4,
            n_layers=2,
            dense_window=4,
            retrieval_budget=2,
            block_size=1,
            router_pointer_input=True,
            unrotated_address_matching=True,
            sparse_backend="chunked",
            checkpoint_sparse=False,
            dropout=0.0,
        )
    ).eval()
    values, pointers, token_types = _inputs(11)
    with torch.no_grad():
        batched = model(values, pointers, token_types).logits
        cache = model.empty_cache(max_length=values.shape[1])
        steps = []
        for position in range(values.shape[1]):
            output, cache = model.step(
                values[:, position],
                pointers[:, position],
                token_types[:, position],
                cache,
            )
            steps.append(output.logits)
        incremental = torch.cat(steps, dim=1)
    torch.testing.assert_close(batched, incremental, rtol=2e-5, atol=2e-5)


def test_layers_change_router_decisions_through_hidden_state() -> None:
    torch.manual_seed(23)
    model = QKVSDecoder(
        ModelConfig(
            vocab_size=16,
            d_model=32,
            n_heads=4,
            n_layers=3,
            dense_window=4,
            retrieval_budget=2,
            block_size=1,
        )
    ).eval()
    values, pointers, token_types = _inputs(11)
    addresses = model(values, pointers, token_types).addresses
    assert addresses is not None
    assert not torch.equal(addresses[0, :, -1], addresses[1, :, -1])


def test_incremental_cache_appends_without_reallocating_or_mutating_history() -> None:
    model = QKVSDecoder(
        ModelConfig(
            vocab_size=16,
            d_model=32,
            n_heads=4,
            n_layers=2,
            dense_window=4,
            retrieval_budget=2,
            block_size=1,
        )
    ).eval()
    values, pointers, token_types = _inputs(3)
    cache = model.empty_cache(max_length=3)
    with torch.no_grad():
        _, cache = model.step(values[:, 0], pointers[:, 0], token_types[:, 0], cache)
        first_layer = cache.layers[0]
        assert first_layer is not None
        storage_pointer = first_layer.keys.data_ptr()
        historical_key = first_layer.keys[:, :, 0].clone()
        _, cache = model.step(values[:, 1], pointers[:, 1], token_types[:, 1], cache)
    updated = cache.layers[0]
    assert updated is not None
    assert updated.length == 2
    assert updated.keys.shape[2] == 3
    assert updated.keys.data_ptr() == storage_pointer
    torch.testing.assert_close(updated.keys[:, :, 0], historical_key)


def test_raw_pointer_encoding_has_no_learned_projection() -> None:
    model = QKVSDecoder(
        ModelConfig(
            vocab_size=16,
            d_model=32,
            n_heads=4,
            n_layers=2,
            dense_window=4,
            retrieval_budget=2,
            block_size=1,
            pointer_encoding="raw",
        )
    )
    assert model.pointer_encoder.projection is None
    encoded = model.pointer_encoder(torch.tensor([[0.25, -1.0]]))
    assert encoded[0, 0, 0] == 0.25
    assert encoded[0, 0, 1] == 1.0
    assert not encoded[0, 1].any()


def test_router_capacity_keeps_parameter_shape_across_active_geometries() -> None:
    output_shapes = []
    for windows in (1, 2, 4, 8):
        model = QKVSDecoder(
            ModelConfig(
                vocab_size=9,
                d_model=32,
                n_heads=4,
                n_layers=1,
                dense_window=8,
                retrieval_budget=32,
                block_size=32 // windows,
                router_capacity=8,
            )
        )
        projection = model.layers[0].attention.s_proj
        assert isinstance(projection, torch.nn.Linear)
        output_shapes.append(projection.weight.shape)
    assert output_shapes == [(8, 32)] * 4


def test_router_mlp_preserves_requested_output_capacity() -> None:
    model = QKVSDecoder(
        ModelConfig(
            vocab_size=9,
            d_model=32,
            n_heads=4,
            n_layers=1,
            dense_window=8,
            retrieval_budget=32,
            block_size=4,
            router_capacity=8,
            router_mlp_ratio=2,
        )
    )
    router = model.layers[0].attention.s_proj
    assert isinstance(router, torch.nn.Sequential)
    assert router[-1].weight.shape == (8, 64)


def test_pointer_set_embedding_is_permutation_invariant() -> None:
    torch.manual_seed(32)
    model = QKVSDecoder(
        ModelConfig(
            vocab_size=9,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dense_window=4,
            retrieval_budget=8,
            block_size=2,
            pointer_set_encoding="sorted_fourier",
        )
    ).eval()
    values = torch.zeros((1, 12), dtype=torch.long)
    pointers = torch.full((1, 12, 4), -1.0)
    pointers[0, -1] = torch.tensor([0.1, 0.3, 0.6, 0.8])
    permuted = pointers.clone()
    permuted[0, -1] = pointers[0, -1, torch.tensor([2, 0, 3, 1])]
    token_types = torch.zeros((1, 12), dtype=torch.long)
    with torch.no_grad():
        expected = model(values, pointers, token_types)
        actual = model(values, permuted, token_types)
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.addresses, expected.addresses)


def test_direct_router_pointer_input_changes_s_without_changing_hidden_input() -> None:
    attention = (
        QKVSDecoder(
            ModelConfig(
                vocab_size=9,
                d_model=32,
                n_heads=4,
                n_layers=1,
                dense_window=4,
                retrieval_budget=2,
                block_size=1,
                router_pointer_input=True,
            )
        )
        .layers[0]
        .attention
    )
    hidden = torch.zeros(1, 8, 32)
    first_context = torch.zeros_like(hidden)
    second_context = first_context.clone()
    second_context[:, -1, 0] = 1.0
    with torch.no_grad():
        first = attention._addresses(hidden, router_context=first_context)
        second = attention._addresses(hidden, router_context=second_context)
    assert first is not None and second is not None
    assert not torch.equal(first[:, -1], second[:, -1])


def test_soft_neighbor_answer_loss_reaches_local_offset_head() -> None:
    torch.manual_seed(41)
    model = QKVSDecoder(
        ModelConfig(
            vocab_size=16,
            d_model=32,
            n_heads=4,
            n_layers=1,
            dense_window=4,
            retrieval_budget=4,
            block_size=2,
            router_output_encoding="hierarchical",
            router_radix=8,
            router_local_offset=True,
            soft_neighbor_training=True,
            sparse_backend="materialized",
        )
    ).train()
    values, pointers, token_types = _inputs(12)
    output = model(values, pointers, token_types)
    torch.nn.functional.cross_entropy(
        output.logits[:, -1],
        torch.tensor([1, 2]),
    ).backward()
    offset = model.layers[0].attention.s_offset_proj
    assert isinstance(offset, torch.nn.Linear)
    assert offset.weight.grad is not None and offset.weight.grad.abs().sum() > 0
    router = model.layers[0].attention.router_output_projection()
    assert router is not None and router.weight.grad is None
