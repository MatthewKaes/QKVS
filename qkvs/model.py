from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from qkvs.attention import AttentionOutput, CausalSelfAttention, KVCache
from qkvs.config import ModelConfig


@dataclass(frozen=True)
class DecoderOutput:
    logits: Tensor
    addresses: Tensor | None  # [layers, batch, time, windows]


@dataclass(frozen=True)
class DecoderCache:
    layers: tuple[KVCache | None, ...]
    max_length: int


class PointerEncoder(nn.Module):
    def __init__(self, d_model: int, mode: str = "fourier", n_frequencies: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.n_frequencies = n_frequencies
        if mode == "fourier":
            self.projection: nn.Module | None = nn.Sequential(
                nn.Linear(2 + 2 * n_frequencies, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        elif mode == "raw":
            self.projection = None
        else:
            raise ValueError(f"unknown pointer encoding: {mode}")

    def forward(self, pointers: Tensor) -> Tensor:
        valid = (pointers >= 0).to(torch.float32)
        addresses = pointers.clamp(0.0, 1.0).to(torch.float32)
        if self.mode == "raw":
            encoded = torch.zeros(
                *pointers.shape,
                self.d_model,
                device=pointers.device,
                dtype=torch.float32,
            )
            encoded[..., 0] = addresses * valid
            encoded[..., 1] = valid
            return encoded
        frequencies = 2.0 ** torch.arange(
            self.n_frequencies, device=pointers.device, dtype=torch.float32
        )
        angles = 2.0 * torch.pi * addresses.unsqueeze(-1) * frequencies
        features = torch.cat(
            (
                addresses.unsqueeze(-1),
                valid.unsqueeze(-1),
                torch.sin(angles) * valid.unsqueeze(-1),
                torch.cos(angles) * valid.unsqueeze(-1),
            ),
            dim=-1,
        )
        assert self.projection is not None
        return self.projection(features) * valid.unsqueeze(-1)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config.attention())
        self.mlp_norm = nn.RMSNorm(config.d_model)
        hidden = config.d_model * config.mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, config.d_model, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden: Tensor,
        address_override: Tensor | None = None,
        retrieval_mask: Tensor | None = None,
    ) -> tuple[Tensor, AttentionOutput]:
        attention = self.attention(self.attention_norm(hidden), address_override, retrieval_mask)
        hidden = hidden + attention.hidden
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden, attention

    def step(
        self,
        hidden: Tensor,
        cache: KVCache | None,
        cache_capacity: int,
        address_override: Tensor | None = None,
    ) -> tuple[Tensor, AttentionOutput, KVCache]:
        attention, updated_cache = self.attention.step(
            self.attention_norm(hidden), cache, cache_capacity, address_override
        )
        hidden = hidden + attention.hidden
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden, attention, updated_cache


class QKVSDecoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.value_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.type_embedding = nn.Embedding(config.n_token_types, config.d_model)
        self.semantic_embedding = (
            nn.Embedding(config.semantic_vocab_size, config.d_model)
            if config.semantic_vocab_size > 0
            else None
        )
        self.pointer_encoder = PointerEncoder(config.d_model, mode=config.pointer_encoding)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _embed(
        self,
        values: Tensor,
        pointers: Tensor,
        token_types: Tensor,
        semantic_ids: Tensor | None = None,
    ) -> Tensor:
        hidden = (
            self.value_embedding(values)
            + self.type_embedding(token_types)
            + self.pointer_encoder(pointers)
        )
        if self.semantic_embedding is not None:
            if semantic_ids is None:
                raise ValueError("semantic IDs are required by this model configuration")
            valid = semantic_ids >= 0
            hidden = hidden + self.semantic_embedding(semantic_ids.clamp_min(0)) * valid.unsqueeze(
                -1
            )
        return self.dropout(hidden)

    def forward(
        self,
        values: Tensor,
        pointers: Tensor,
        token_types: Tensor,
        address_overrides: Tensor | None = None,
        retrieval_mask: Tensor | None = None,
        semantic_ids: Tensor | None = None,
    ) -> DecoderOutput:
        if address_overrides is not None and address_overrides.shape[0] != len(self.layers):
            raise ValueError("address overrides need one leading entry per layer")
        hidden = self._embed(values, pointers, token_types, semantic_ids)
        layer_addresses: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            override = None if address_overrides is None else address_overrides[index]
            hidden, attention = layer(hidden, override, retrieval_mask)
            if attention.addresses is not None:
                layer_addresses.append(attention.addresses)
        logits = self.output(self.final_norm(hidden))
        addresses = torch.stack(layer_addresses) if layer_addresses else None
        return DecoderOutput(logits=logits, addresses=addresses)

    def empty_cache(self, max_length: int) -> DecoderCache:
        if max_length <= 0:
            raise ValueError("maximum cache length must be positive")
        return DecoderCache(layers=tuple(None for _ in self.layers), max_length=max_length)

    def step(
        self,
        values: Tensor,
        pointers: Tensor,
        token_types: Tensor,
        cache: DecoderCache,
        address_overrides: Tensor | None = None,
        semantic_ids: Tensor | None = None,
    ) -> tuple[DecoderOutput, DecoderCache]:
        if values.ndim != 1:
            raise ValueError("incremental inputs must have shape [batch]")
        hidden = self._embed(
            values.unsqueeze(1),
            pointers.unsqueeze(1),
            token_types.unsqueeze(1),
            None if semantic_ids is None else semantic_ids.unsqueeze(1),
        )
        layer_addresses: list[Tensor] = []
        updated_caches: list[KVCache] = []
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache.layers, strict=True)):
            override = None
            if address_overrides is not None:
                override = address_overrides[index].unsqueeze(1)
            hidden, attention, updated = layer.step(hidden, layer_cache, cache.max_length, override)
            updated_caches.append(updated)
            if attention.addresses is not None:
                layer_addresses.append(attention.addresses)
        logits = self.output(self.final_norm(hidden))
        addresses = torch.stack(layer_addresses) if layer_addresses else None
        return DecoderOutput(logits, addresses), DecoderCache(
            tuple(updated_caches), cache.max_length
        )
