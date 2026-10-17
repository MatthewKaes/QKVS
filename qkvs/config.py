from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

RouterMode = Literal["learned", "unconditioned", "random"]
AttentionKind = Literal["qkvs", "dense", "rolling"]
PointerEncoding = Literal["fourier", "raw"]
SparseBackend = Literal["materialized", "streaming"]


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
    n_token_types: int = 3
    pointer_encoding: PointerEncoding = "fourier"
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
            sparse_backend=self.sparse_backend,
            query_chunk_size=self.query_chunk_size,
            candidate_tile_size=self.candidate_tile_size,
            checkpoint_sparse=self.checkpoint_sparse,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
