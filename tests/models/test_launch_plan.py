from dataclasses import replace
from pathlib import Path

import pytest

from engine.hardware.allocator import plan_budget
from engine.hardware.profiler import CPUInfo, GiB
from engine.ktx_bridge.launch_plan import InfeasiblePlan, plan_launch
from engine.models.descriptor import describe_file
from tests.factories import gpu, make_profile

FIXTURES = Path(__file__).parent / "fixtures"
QWEN3 = describe_file(FIXTURES / "qwen3_30b_a3b.json")


def plan_for(profile, desc=QWEN3):
    return plan_launch(desc, profile, plan_budget(profile), model_path="/m/bf16", expert_weight_path="/m/gguf")


def flag(plan, name):
    return next(f.value for f in plan.flags if f.name == name)


def test_laptop_6gb_puts_all_experts_on_cpu_and_fills_vram_with_kv():
    profile = make_profile([gpu(6, 5.66)], ram_total=38, ram_avail=31)
    plan = plan_for(profile)
    assert plan.gpu_experts_per_layer == 0
    assert plan.gpu_dense_bytes == QWEN3.dense_params * 2
    assert 4096 <= plan.context_tokens < 32768
    assert flag(plan, "--kt-method") == "LLAMAFILE"
    assert flag(plan, "--kt-cpuinfer") == "6"
    assert flag(plan, "--kt-threadpool-count") == "1"
    assert flag(plan, "--chunked-prefill-size") == "2048"
    static = plan.gpu_dense_bytes + plan.kv_cache_bytes
    assert float(flag(plan, "--mem-fraction-static")) <= static / (6 * GiB)
    assert plan.notes == ()


def test_24gb_gpu_gets_full_context_then_gpu_experts():
    profile = make_profile([gpu(24, 23.5)], ram_total=64, ram_avail=60)
    plan = plan_for(profile)
    assert plan.gpu_experts_per_layer > 0
    assert plan.context_tokens >= 32768
    assert flag(plan, "--kt-expert-placement-strategy") == "uniform"
    budget = plan_budget(profile)
    used = plan.gpu_dense_bytes + plan.gpu_expert_bytes + plan.kv_cache_bytes
    assert used <= budget.vram_bytes


def test_argv_is_a_runnable_sglang_command():
    argv = plan_for(make_profile([gpu(24, 23.5)])).argv()
    assert argv[:3] == ["python", "-m", "sglang.launch_server"]
    assert argv[argv.index("--kt-num-gpu-experts") + 1].isdigit()


def test_ram_shortfall_is_reported():
    plan = plan_for(make_profile([gpu(6, 5.66)], ram_total=16, ram_avail=14))
    assert "exceed the RAM budget" in plan.notes[0]


@pytest.mark.parametrize("profile,message", [
    (make_profile(), "needs a CUDA GPU"),
    (make_profile([gpu(8, 7.5, cc=(7, 5))]), "compute capability 7.5"),
    (make_profile([gpu(8, 7.5)], cpu=CPUInfo(4, 8, 1, frozenset({"sse4_2"}))), "AVX2"),
    (make_profile([gpu(3, 2.9)]), "Dense weights need"),
    (make_profile([gpu(4, 3.9)]), "tokens of KV cache fit"),
])
def test_infeasible_hardware_explains_why(profile, message):
    with pytest.raises(InfeasiblePlan, match=message):
        plan_for(profile)


def test_deepseek_v3_dense_weights_do_not_fit_small_gpu():
    v3 = describe_file(FIXTURES / "deepseek_v3.json")
    with pytest.raises(InfeasiblePlan, match="Dense weights need"):
        plan_for(make_profile([gpu(24, 23.5)]), v3)


def test_context_never_exceeds_model_maximum():
    small_ctx = replace(QWEN3, max_position_embeddings=8192)
    plan = plan_for(make_profile([gpu(24, 23.5)]), small_ctx)
    assert plan.context_tokens == 8192
