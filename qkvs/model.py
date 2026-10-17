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
    router_logits: Tensor | None = None
    router_offsets: Tensor | None = None


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
        historical_window_mask: Tensor | None = None,
        router_context: Tensor | None = None,
    ) -> tuple[Tensor, AttentionOutput]:
        attention = self.attention(
            self.attention_norm(hidden),
            address_override,
            retrieval_mask,
            historical_window_mask,
            router_context,
        )
        hidden = hidden + attention.hidden
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden, attention

    def step(
        self,
        hidden: Tensor,
        cache: KVCache | None,
        cache_capacity: int,
        address_override: Tensor | None = None,
        router_context: Tensor | None = None,
    ) -> tuple[Tensor, AttentionOutput, KVCache]:
        attention, updated_cache = self.attention.step(
            self.attention_norm(hidden),
            cache,
            cache_capacity,
            address_override,
            router_context,
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
    ) -> tuple[Tensor, Tensor]:
        hidden = self.value_embedding(values) + self.type_embedding(token_types)
        if pointers.ndim == values.ndim + 1:
            if self.config.pointer_set_encoding in {"sorted_raw", "sorted_fourier"}:
                valid = pointers >= 0
                sort_values = torch.where(valid, pointers, torch.inf)
                sorted_values, order = sort_values.sort(dim=-1)
                sorted_valid = valid.gather(-1, order)
                sorted_values = torch.where(sorted_valid, sorted_values, 0.0)
                if self.config.pointer_set_encoding == "sorted_fourier":
                    frequencies = 2.0 ** torch.arange(
                        4, device=pointers.device, dtype=torch.float32
                    )
                    angles = 2.0 * torch.pi * sorted_values.unsqueeze(-1) * frequencies
                    valid_feature = sorted_valid.unsqueeze(-1).to(torch.float32)
                    features = torch.cat(
                        (
                            sorted_values.unsqueeze(-1),
                            valid_feature,
                            torch.sin(angles) * valid_feature,
                            torch.cos(angles) * valid_feature,
                        ),
                        dim=-1,
                    ).flatten(-2)
                else:
                    features = torch.cat((sorted_values, sorted_valid.to(torch.float32)), dim=-1)
                if features.shape[-1] > self.config.d_model:
                    raise ValueError("sorted pointer set does not fit in the model width")
                pointer_hidden = torch.zeros(
                    *values.shape,
                    self.config.d_model,
                    device=values.device,
                    dtype=torch.float32,
                )
                pointer_hidden[..., : features.shape[-1]] = features
            else:
                pointer_hidden = self.pointer_encoder(pointers)
                pointer_count = (pointers >= 0).sum(dim=-1, keepdim=True).clamp_min(1).sqrt()
                pointer_hidden = pointer_hidden.sum(dim=-2) / pointer_count
        elif pointers.ndim == values.ndim:
            pointer_hidden = self.pointer_encoder(pointers)
        else:
            raise ValueError("pointers need zero or one set dimension")
        hidden = hidden + pointer_hidden
        if self.semantic_embedding is not None:
            if semantic_ids is None:
                raise ValueError("semantic IDs are required by this model configuration")
            valid = semantic_ids >= 0
            hidden = hidden + self.semantic_embedding(semantic_ids.clamp_min(0)) * valid.unsqueeze(
                -1
            )
        return self.dropout(hidden), pointer_hidden

    def forward(
        self,
        values: Tensor,
        pointers: Tensor,
        token_types: Tensor,
        address_overrides: Tensor | None = None,
        retrieval_mask: Tensor | None = None,
        semantic_ids: Tensor | None = None,
        historical_window_masks: Tensor | None = None,
    ) -> DecoderOutput:
        if address_overrides is not None and address_overrides.shape[0] != len(self.layers):
            raise ValueError("address overrides need one leading entry per layer")
        hidden, pointer_hidden = self._embed(values, pointers, token_types, semantic_ids)
        router_context = pointer_hidden if self.config.router_pointer_input else None
        layer_addresses: list[Tensor] = []
        layer_router_logits: list[Tensor] = []
        layer_router_offsets: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            override = None if address_overrides is None else address_overrides[index]
            window_mask = (
                None if historical_window_masks is None else historical_window_masks[index]
            )
            hidden, attention = layer(
                hidden,
                override,
                retrieval_mask,
                window_mask,
                router_context,
            )
            if attention.addresses is not None:
                layer_addresses.append(attention.addresses)
            if attention.router_logits is not None:
                layer_router_logits.append(attention.router_logits)
            if attention.router_offsets is not None:
                layer_router_offsets.append(attention.router_offsets)
        logits = self.output(self.final_norm(hidden))
        addresses = torch.stack(layer_addresses) if layer_addresses else None
        router_logits = torch.stack(layer_router_logits) if layer_router_logits else None
        router_offsets = torch.stack(layer_router_offsets) if layer_router_offsets else None
        return DecoderOutput(
            logits=logits,
            addresses=addresses,
            router_logits=router_logits,
            router_offsets=router_offsets,
        )

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
        hidden, pointer_hidden = self._embed(
            values.unsqueeze(1),
            pointers.unsqueeze(1),
            token_types.unsqueeze(1),
            None if semantic_ids is None else semantic_ids.unsqueeze(1),
        )
        router_context = pointer_hidden if self.config.router_pointer_input else None
        layer_addresses: list[Tensor] = []
        layer_router_logits: list[Tensor] = []
        layer_router_offsets: list[Tensor] = []
        updated_caches: list[KVCache] = []
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache.layers, strict=True)):
            override = None
            if address_overrides is not None:
                override = address_overrides[index].unsqueeze(1)
            hidden, attention, updated = layer.step(
                hidden,
                layer_cache,
                cache.max_length,
                override,
                router_context,
            )
            updated_caches.append(updated)
            if attention.addresses is not None:
                layer_addresses.append(attention.addresses)
            if attention.router_logits is not None:
                layer_router_logits.append(attention.router_logits)
            if attention.router_offsets is not None:
                layer_router_offsets.append(attention.router_offsets)
        logits = self.output(self.final_norm(hidden))
        addresses = torch.stack(layer_addresses) if layer_addresses else None
        router_logits = torch.stack(layer_router_logits) if layer_router_logits else None
        router_offsets = torch.stack(layer_router_offsets) if layer_router_offsets else None
        return DecoderOutput(logits, addresses, router_logits, router_offsets), DecoderCache(
            tuple(updated_caches), cache.max_length
        )
