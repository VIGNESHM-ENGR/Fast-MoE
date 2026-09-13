"""Generic MoE architecture descriptor: normalizes a HF config.json across MoE
families and computes parameter counts and KV-cache size from it.

Each family names the same concepts differently (e.g. `num_experts` vs
`num_local_experts` vs `n_routed_experts`), so parsing is explicit per
`model_type`; unknown architectures are rejected rather than guessed, because
a wrong guess produces an out-of-memory crash at load time.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class UnsupportedArchitecture(ValueError):
    pass


@dataclass(frozen=True)
class GQAAttention:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    bias: bool = False

    def params(self, hidden: int) -> int:
        q, kv = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        bias = (q + 2 * kv) if self.bias else 0
        return hidden * q + 2 * hidden * kv + q * hidden + bias

    def kv_bytes_per_token_per_layer(self, dtype_bytes: int) -> int:
        return 2 * self.num_kv_heads * self.head_dim * dtype_bytes


@dataclass(frozen=True)
class MLAAttention:
    """DeepSeek multi-head latent attention (arXiv:2405.04434)."""

    num_heads: int
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    def params(self, hidden: int) -> int:
        qk_head = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.q_lora_rank is None:
            q = hidden * self.num_heads * qk_head
        else:
            q = hidden * self.q_lora_rank + self.q_lora_rank + self.q_lora_rank * self.num_heads * qk_head
        kv_a = hidden * (self.kv_lora_rank + self.qk_rope_head_dim) + self.kv_lora_rank
        kv_b = self.kv_lora_rank * self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        o = self.num_heads * self.v_head_dim * hidden
        return q + kv_a + kv_b + o

    def kv_bytes_per_token_per_layer(self, dtype_bytes: int) -> int:
        # Only the compressed latent and the decoupled RoPE key are cached.
        return (self.kv_lora_rank + self.qk_rope_head_dim) * dtype_bytes


@dataclass(frozen=True)
class MoEArchDescriptor:
    model_type: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    tie_word_embeddings: bool
    max_position_embeddings: int
    attention: GQAAttention | MLAAttention
    num_experts: int
    experts_per_token: int
    expert_intermediate_size: int
    moe_layers: tuple[int, ...]
    dense_intermediate_size: int
    num_shared_experts: int = 0

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layers)

    @property
    def expert_params(self) -> int:
        """Parameters of one routed expert (SwiGLU: gate, up, down projections)."""
        return 3 * self.hidden_size * self.expert_intermediate_size

    @property
    def routed_expert_params(self) -> int:
        return self.num_moe_layers * self.num_experts * self.expert_params

    @property
    def dense_params(self) -> int:
        """Everything except routed experts: embeddings, attention, norms, routers,
        shared experts and dense MLP layers. These run for every token."""
        h = self.hidden_size
        embeddings = self.vocab_size * h * (1 if self.tie_word_embeddings else 2)
        per_layer = self.attention.params(h) + 2 * h
        moe_extra = h * self.num_experts + self.num_shared_experts * self.expert_params
        dense_mlp = 3 * h * self.dense_intermediate_size
        num_dense = self.num_layers - self.num_moe_layers
        return (embeddings + h + self.num_layers * per_layer
                + self.num_moe_layers * moe_extra + num_dense * dense_mlp)

    @property
    def total_params(self) -> int:
        return self.dense_params + self.routed_expert_params

    @property
    def active_params(self) -> int:
        return self.dense_params + self.num_moe_layers * self.experts_per_token * self.expert_params

    def kv_cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        return self.num_layers * self.attention.kv_bytes_per_token_per_layer(dtype_bytes)


def _qwen3_moe(c: dict) -> MoEArchDescriptor:
    n_layers = c["num_hidden_layers"]
    skip, step = set(c.get("mlp_only_layers", [])), c.get("decoder_sparse_step", 1)
    return MoEArchDescriptor(
        model_type=c["model_type"],
        num_layers=n_layers,
        hidden_size=c["hidden_size"],
        vocab_size=c["vocab_size"],
        tie_word_embeddings=c.get("tie_word_embeddings", False),
        max_position_embeddings=c["max_position_embeddings"],
        attention=GQAAttention(
            num_heads=c["num_attention_heads"],
            num_kv_heads=c["num_key_value_heads"],
            head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
            bias=c.get("attention_bias", False),
        ),
        num_experts=c["num_experts"],
        experts_per_token=c["num_experts_per_tok"],
        expert_intermediate_size=c["moe_intermediate_size"],
        moe_layers=tuple(i for i in range(n_layers) if i not in skip and (i + 1) % step == 0),
        dense_intermediate_size=c["intermediate_size"],
    )


def _mixtral(c: dict) -> MoEArchDescriptor:
    n_layers = c["num_hidden_layers"]
    return MoEArchDescriptor(
        model_type=c["model_type"],
        num_layers=n_layers,
        hidden_size=c["hidden_size"],
        vocab_size=c["vocab_size"],
        tie_word_embeddings=c.get("tie_word_embeddings", False),
        max_position_embeddings=c["max_position_embeddings"],
        attention=GQAAttention(
            num_heads=c["num_attention_heads"],
            num_kv_heads=c["num_key_value_heads"],
            head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
        ),
        num_experts=c["num_local_experts"],
        experts_per_token=c["num_experts_per_tok"],
        # Mixtral experts use the model-wide intermediate size; there are no dense MLP layers.
        expert_intermediate_size=c["intermediate_size"],
        moe_layers=tuple(range(n_layers)),
        dense_intermediate_size=c["intermediate_size"],
    )


def _deepseek(c: dict) -> MoEArchDescriptor:
    n_layers = c["num_hidden_layers"]
    first_dense, freq = c.get("first_k_dense_replace", 0), c.get("moe_layer_freq", 1)
    return MoEArchDescriptor(
        model_type=c["model_type"],
        num_layers=n_layers,
        hidden_size=c["hidden_size"],
        vocab_size=c["vocab_size"],
        tie_word_embeddings=c.get("tie_word_embeddings", False),
        max_position_embeddings=c["max_position_embeddings"],
        attention=MLAAttention(
            num_heads=c["num_attention_heads"],
            q_lora_rank=c.get("q_lora_rank"),
            kv_lora_rank=c["kv_lora_rank"],
            qk_nope_head_dim=c["qk_nope_head_dim"],
            qk_rope_head_dim=c["qk_rope_head_dim"],
            v_head_dim=c["v_head_dim"],
        ),
        num_experts=c["n_routed_experts"],
        experts_per_token=c["num_experts_per_tok"],
        expert_intermediate_size=c["moe_intermediate_size"],
        # Same rule sglang-kt uses to count MoE layers (kt_ep_wrapper.py).
        moe_layers=tuple(i for i in range(n_layers) if i >= first_dense and i % freq == 0),
        dense_intermediate_size=c["intermediate_size"],
        num_shared_experts=c.get("n_shared_experts") or 0,
    )


PARSERS: dict[str, Callable[[dict], MoEArchDescriptor]] = {
    "qwen3_moe": _qwen3_moe,
    "mixtral": _mixtral,
    "deepseek_v2": _deepseek,
    "deepseek_v3": _deepseek,
}


def describe(config: dict) -> MoEArchDescriptor:
    model_type = config.get("model_type")
    parser = PARSERS.get(model_type)
    if parser is None:
        raise UnsupportedArchitecture(
            f"Unsupported model_type {model_type!r}; supported: {', '.join(sorted(PARSERS))}"
        )
    return parser(config)


def describe_file(config_path: str | Path) -> MoEArchDescriptor:
    return describe(json.loads(Path(config_path).read_text()))
