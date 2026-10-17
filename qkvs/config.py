from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

RouterMode = Literal["learned", "unconditioned", "random"]
AttentionKind = Literal["qkvs", "dense", "rolling"]


@dataclass(frozen=True)
class AttentionConfig:
    d_model: int = 128
    n_heads: int = 4
    dense_window: int = 32
    retrieval_budget: int = 8
    block_size: int = 2
    dropout: float = 0.0
    kind: AttentionKind = "qkvs"
    router_mode: RouterMode = "learned"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.dense_window <= 0:
            raise ValueError("dense_window must be positive")
        if self.retrieval_budget < 0:
            raise ValueError("retrieval_budget cannot be negative")
        if self.kind == "qkvs":
            if self.retrieval_budget <= 0:
                raise ValueError("QKVS requires a positive retrieval budget")
            if self.retrieval_budget % self.block_size:
                raise ValueError("retrieval_budget must be divisible by block_size")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_windows(self) -> int:
        if self.kind != "qkvs":
            return 0
        return self.retrieval_budget // self.block_size

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
    dropout: float = 0.0
    attention_kind: AttentionKind = "qkvs"
    router_mode: RouterMode = "learned"
    n_token_types: int = 3

    def attention(self) -> AttentionConfig:
        return AttentionConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dense_window=self.dense_window,
            retrieval_budget=self.retrieval_budget,
            block_size=self.block_size,
            dropout=self.dropout,
            kind=self.attention_kind,
            router_mode=self.router_mode,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
