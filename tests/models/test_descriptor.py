from pathlib import Path

import pytest

from engine.models.descriptor import (
    GQAAttention,
    MLAAttention,
    UnsupportedArchitecture,
    describe,
    describe_file,
)

FIXTURES = Path(__file__).parent / "fixtures"


def within(actual: int, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) / expected <= tolerance


# Published sizes from each model card / paper.
@pytest.mark.parametrize("fixture,total,active", [
    ("qwen3_30b_a3b", 30.5e9, 3.3e9),
    ("mixtral_8x7b", 46.7e9, 12.9e9),
    ("deepseek_v3", 671e9, 37e9),
])
def test_parameter_counts_match_published_sizes(fixture, total, active):
    d = describe_file(FIXTURES / f"{fixture}.json")
    assert within(d.total_params, total, 0.015)
    assert within(d.active_params, active, 0.03)


def test_deepseek_v2_lite_total():
    # DeepSeek reports 2.4B "activated" excluding the input embedding, so only total is compared.
    d = describe_file(FIXTURES / "deepseek_v2_lite.json")
    assert within(d.total_params, 15.7e9, 0.015)


def test_qwen3_30b_a3b_shape():
    d = describe_file(FIXTURES / "qwen3_30b_a3b.json")
    assert (d.num_experts, d.experts_per_token, d.num_moe_layers) == (128, 8, 48)
    assert d.attention == GQAAttention(num_heads=32, num_kv_heads=4, head_dim=128)
    # 48 layers x 2 (K and V) x 4 KV heads x 128 dims x 2 bytes
    assert d.kv_cache_bytes_per_token() == 98_304


def test_deepseek_dense_prefix_layers_are_not_moe():
    v3 = describe_file(FIXTURES / "deepseek_v3.json")
    assert v3.moe_layers[0] == 3 and v3.num_moe_layers == 58
    assert v3.num_shared_experts == 1
    assert isinstance(v3.attention, MLAAttention)
    # MLA caches only kv_lora_rank + rope dims per layer: 61 x (512 + 64) x 2 bytes
    assert v3.kv_cache_bytes_per_token() == 61 * 576 * 2


def test_mixtral_every_layer_is_moe():
    d = describe_file(FIXTURES / "mixtral_8x7b.json")
    assert d.moe_layers == tuple(range(32))
    assert d.expert_intermediate_size == 14336


def test_qwen3_mlp_only_layers_and_sparse_step():
    config = {
        "model_type": "qwen3_moe", "num_hidden_layers": 6, "hidden_size": 16, "vocab_size": 10,
        "max_position_embeddings": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 8,
        "intermediate_size": 32, "mlp_only_layers": [1], "decoder_sparse_step": 2,
    }
    assert describe(config).moe_layers == (3, 5)


def test_unknown_architecture_is_rejected():
    with pytest.raises(UnsupportedArchitecture, match="qwen3_moe"):
        describe({"model_type": "llama"})
