import torch

from qkvs.config import ModelConfig
from qkvs.data import PointerBatch, make_address_overrides, make_multi_region_batch
from qkvs.model import DecoderOutput, QKVSDecoder
from qkvs.training import TrainConfig, _route_loss, _router_parameters


def _set_routing_fixture() -> tuple[ModelConfig, TrainConfig, PointerBatch]:
    model_config = ModelConfig(
        vocab_size=9,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dense_window=8,
        retrieval_budget=32,
        block_size=4,
        address_stride=4,
        semantic_vocab_size=0,
        router_capacity=8,
        sparse_backend="chunked",
        query_chunk_size=16,
    )
    train_config = TrainConfig(
        name="set_routing",
        task="multi_region",
        region_records=16,
        required_regions=8,
        route_loss_space="set_coverage",
    )
    batch = make_multi_region_batch(
        batch_size=3,
        seq_len=128,
        dense_window=model_config.dense_window,
        block_size=model_config.block_size,
        address_stride=model_config.attention().stride,
        n_windows=model_config.attention().n_windows,
        n_layers=model_config.n_layers,
        n_records=train_config.region_records,
        required_regions=train_config.required_regions,
        generator=torch.Generator().manual_seed(111),
    )
    return model_config, train_config, batch


def test_set_route_loss_is_permutation_invariant() -> None:
    model_config, train_config, batch = _set_routing_fixture()
    addresses = torch.rand(2, 3, 1, 8, generator=torch.Generator().manual_seed(112))
    output = DecoderOutput(torch.zeros(3, 1, 9), addresses)
    permuted = DecoderOutput(torch.zeros(3, 1, 9), addresses[..., [5, 2, 7, 0, 4, 1, 6, 3]])
    torch.testing.assert_close(
        _route_loss(output, batch, model_config, train_config),
        _route_loss(permuted, batch, model_config, train_config),
    )


def test_set_route_loss_reaches_every_active_s_output() -> None:
    model_config, train_config, batch = _set_routing_fixture()
    model = QKVSDecoder(model_config)
    output = model(
        batch.values,
        batch.pointers,
        batch.token_types,
        semantic_ids=batch.semantic_ids,
    )
    _route_loss(output, batch, model_config, train_config).backward()
    for layer in model.layers:
        projection = layer.attention.s_proj
        assert isinstance(projection, torch.nn.Linear)
        assert projection.weight.grad is not None
        gradients = projection.weight.grad.norm(dim=1)
        assert gradients.shape == (8,)
        assert torch.all(gradients > 0)


def test_hierarchical_set_route_loss_reaches_every_digit_output() -> None:
    model_config, train_config, batch = _set_routing_fixture()
    values = model_config.as_dict()
    values["router_output_encoding"] = "hierarchical"
    values["router_radix"] = 16
    hierarchical_config = ModelConfig(**values)
    train_config = TrainConfig(
        **{
            **train_config.__dict__,
            "route_assignment_weight": 0.0,
        }
    )
    model = QKVSDecoder(hierarchical_config)
    output = model(
        batch.values,
        batch.pointers,
        batch.token_types,
        semantic_ids=batch.semantic_ids,
    )
    loss = _route_loss(output, batch, hierarchical_config, train_config)
    assert torch.isfinite(loss)
    loss.backward()
    for layer in model.layers:
        projection = layer.attention.router_output_projection()
        assert projection is not None and projection.weight.grad is not None
        gradients = projection.weight.grad.view(8, 2, 16, -1).norm(dim=-1)
        assert torch.all(gradients > 0)


def test_router_parameter_group_owns_coordinate_pointer_and_offset_projections() -> None:
    config = ModelConfig(
        vocab_size=9,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dense_window=8,
        retrieval_budget=8,
        block_size=2,
        router_position_encoding="absolute_fourier",
        router_pointer_input=True,
        router_local_offset=True,
    )
    model = QKVSDecoder(config)
    grouped = {id(parameter) for parameter in _router_parameters(model)}
    for layer in model.layers:
        for module in (
            layer.attention.s_proj,
            layer.attention.s_offset_proj,
            layer.attention.router_position_projection,
            layer.attention.router_pointer_projection,
        ):
            assert module is not None
            assert all(id(parameter) in grouped for parameter in module.parameters())


def test_multi_region_oracle_assigns_every_window_to_a_required_target() -> None:
    model_config, _, batch = _set_routing_fixture()
    overrides = make_address_overrides(
        batch,
        seq_len=128,
        n_windows=model_config.attention().n_windows,
        device=torch.device("cpu"),
    )
    history_end = 128 - model_config.dense_window
    n_addresses = history_end // model_config.attention().stride
    starts = torch.floor(overrides[:, :, -1] * n_addresses).to(torch.long)
    starts *= model_config.attention().stride
    assert batch.route_target_positions is not None
    targets = batch.route_target_positions
    contains = (starts.unsqueeze(-1) <= targets.unsqueeze(-2)) & (
        targets.unsqueeze(-2) < starts.unsqueeze(-1) + model_config.block_size
    )
    assert contains.any(dim=-2).all()
