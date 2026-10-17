import torch

from qkvs.router_diagnostic import StandaloneRouter, make_router_batch


def test_router_batch_has_unique_decodable_cells_at_mixed_lengths() -> None:
    batch = make_router_batch(
        batch_size=32,
        n_windows=8,
        context_lengths=(4096, 8192, 16384),
        dense_window=256,
        address_stride=16,
        generator=torch.Generator().manual_seed(17),
    )
    decoded = torch.floor(batch.pointers * batch.address_counts.unsqueeze(-1)).to(torch.long)
    torch.testing.assert_close(decoded.sort(dim=-1).values, batch.target_cells)
    assert torch.all(batch.target_cells[:, 1:] != batch.target_cells[:, :-1])


def test_standalone_router_is_invariant_to_pointer_order() -> None:
    model = StandaloneRouter(
        output="hierarchical",
        hidden_width=32,
        max_address_count=4096,
        radix=64,
    ).eval()
    pointers = torch.tensor([[0.1, 0.7, 0.3, 0.5]])
    permuted = pointers[:, [2, 0, 3, 1]]
    counts = torch.tensor([240])
    with torch.no_grad():
        expected = model(pointers, counts)
        actual = model(permuted, counts)
    for expected_output, actual_output in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_output, expected_output)
