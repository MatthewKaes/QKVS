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
