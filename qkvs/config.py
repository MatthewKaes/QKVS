from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

RouterMode = Literal["learned", "unconditioned", "random"]
AttentionKind = Literal["qkvs", "dense", "rolling"]
PointerEncoding = Literal["fourier", "raw"]
PointerSetEncoding = Literal["sum", "sorted_raw", "sorted_fourier"]
RouterPositionEncoding = Literal["none", "absolute_fourier"]
RouterOutputEncoding = Literal["scalar", "hierarchical"]
SparseBackend = Literal["materialized", "streaming", "chunked"]


@dataclass(frozen=True)
class AttentionConfig:
    d_model: int = 128
    n_heads: int = 4
    dense_window: int = 32
    retrieval_budget: int = 8
    block_size: int = 2
    address_stride: int | None = None
    dropout: float = 0.0
    kind: AttentionKind = "qkvs"
    router_mode: RouterMode = "learned"
    router_capacity: int | None = None
    router_detach_input: bool = False
    router_mlp_ratio: int = 0
    router_position_encoding: RouterPositionEncoding = "none"
    router_pointer_input: bool = False
    unrotated_address_matching: bool = False
    router_output_encoding: RouterOutputEncoding = "scalar"
    router_radix: int = 64
    router_local_offset: bool = False
    soft_neighbor_training: bool = False
    sparse_backend: SparseBackend = "streaming"
    query_chunk_size: int = 32
    candidate_tile_size: int = 32
    checkpoint_sparse: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.dense_window <= 0:
            raise ValueError("dense_window must be positive")
        if self.retrieval_budget < 0:
            raise ValueError("retrieval_budget cannot be negative")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.address_stride is not None and self.address_stride <= 0:
            raise ValueError("address_stride must be positive")
        if self.address_stride is not None and self.address_stride > self.block_size:
            raise ValueError("address_stride cannot exceed window width")
        if self.kind == "qkvs":
            if self.retrieval_budget <= 0:
                raise ValueError("QKVS requires a positive retrieval budget")
            if self.retrieval_budget % self.block_size:
                raise ValueError("retrieval_budget must be divisible by block_size")
            if self.router_capacity is not None and self.router_capacity < self.n_windows:
                raise ValueError("router capacity cannot be smaller than the active window count")
            if self.router_mlp_ratio < 0:
                raise ValueError("router MLP ratio cannot be negative")
            if self.router_position_encoding not in {"none", "absolute_fourier"}:
                raise ValueError("unknown router position encoding")
            if self.router_output_encoding not in {"scalar", "hierarchical"}:
                raise ValueError("unknown router output encoding")
            if self.router_radix <= 1:
                raise ValueError("router radix must be greater than one")
            if self.soft_neighbor_training and not self.router_local_offset:
                raise ValueError("soft neighbor training requires a local router offset")
            if self.unrotated_address_matching and not self.router_pointer_input:
                raise ValueError("unrotated address matching requires direct pointer input")
        if self.query_chunk_size <= 0 or self.candidate_tile_size <= 0:
            raise ValueError("sparse attention tile sizes must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_windows(self) -> int:
        if self.kind != "qkvs":
            return 0
        return self.retrieval_budget // self.block_size

    @property
    def stride(self) -> int:
        return self.block_size if self.address_stride is None else self.address_stride

    @property
    def selection_budget(self) -> int:
        if self.kind == "dense":
            raise ValueError("dense attention has no fixed selection budget")
        return self.dense_window + (self.retrieval_budget if self.kind == "qkvs" else 0)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    mlp_ratio: int = 4
    dense_window: int = 32
    retrieval_budget: int = 8
    block_size: int = 2
    address_stride: int | None = None
    dropout: float = 0.0
    attention_kind: AttentionKind = "qkvs"
    router_mode: RouterMode = "learned"
    router_capacity: int | None = None
    router_detach_input: bool = False
    router_mlp_ratio: int = 0
    router_position_encoding: RouterPositionEncoding = "none"
    router_pointer_input: bool = False
    unrotated_address_matching: bool = False
    router_output_encoding: RouterOutputEncoding = "scalar"
    router_radix: int = 64
    router_local_offset: bool = False
    soft_neighbor_training: bool = False
    n_token_types: int = 3
    pointer_encoding: PointerEncoding = "fourier"
    pointer_set_encoding: PointerSetEncoding = "sum"
    semantic_vocab_size: int = 0
    sparse_backend: SparseBackend = "streaming"
    query_chunk_size: int = 32
    candidate_tile_size: int = 32
    checkpoint_sparse: bool = True

    def __post_init__(self) -> None:
        if self.semantic_vocab_size < 0:
            raise ValueError("semantic_vocab_size cannot be negative")

    def attention(self) -> AttentionConfig:
        return AttentionConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dense_window=self.dense_window,
            retrieval_budget=self.retrieval_budget,
            block_size=self.block_size,
            address_stride=self.address_stride,
            dropout=self.dropout,
            kind=self.attention_kind,
            router_mode=self.router_mode,
            router_capacity=self.router_capacity,
            router_detach_input=self.router_detach_input,
            router_mlp_ratio=self.router_mlp_ratio,
            router_position_encoding=self.router_position_encoding,
            router_pointer_input=self.router_pointer_input,
            unrotated_address_matching=self.unrotated_address_matching,
            router_output_encoding=self.router_output_encoding,
            router_radix=self.router_radix,
            router_local_offset=self.router_local_offset,
            soft_neighbor_training=self.soft_neighbor_training,
            sparse_backend=self.sparse_backend,
            query_chunk_size=self.query_chunk_size,
            candidate_tile_size=self.candidate_tile_size,
            checkpoint_sparse=self.checkpoint_sparse,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
