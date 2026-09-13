"""Launch plan: turns a MoE descriptor + hardware profile + tier budget into
`sglang.launch_server` flags for sglang-kt / kt-kernel, each with a reason.

VRAM placement priority: dense weights (touched by every token) → KV cache up
to a target context → GPU experts → any remainder extends the context.
Flag semantics were checked against sglang-kt 0.7.0.post3 `server_args.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engine.hardware.allocator import TierBudget
from engine.hardware.profiler import GiB, HardwareProfile
from engine.models.descriptor import MoEArchDescriptor

# Measured from Qwen/Qwen3-30B-A3B-GGUF file sizes (file bytes * 8 / total params).
GGUF_BITS_PER_WEIGHT = {"Q4_K_M": 4.86, "Q5_K_M": 5.69, "Q6_K": 6.58, "Q8_0": 8.51}
# sglang-kt loads GPU-side weights (dense layers and GPU experts) from the BF16 checkpoint.
GPU_WEIGHT_BYTES = 2
KV_DTYPE_BYTES = 2
TARGET_CONTEXT_TOKENS = 32_768
MIN_CONTEXT_TOKENS = 4_096
MIN_COMPUTE_CAPABILITY = (8, 0)  # kt-kernel GPU requirement (Ampere or newer)
SMALL_GPU_BYTES = 8 * GiB


class InfeasiblePlan(RuntimeError):
    pass


@dataclass(frozen=True)
class Flag:
    name: str
    value: str | None
    reason: str


@dataclass(frozen=True)
class LaunchPlan:
    flags: tuple[Flag, ...]
    gpu_dense_bytes: int
    gpu_expert_bytes: int
    kv_cache_bytes: int
    context_tokens: int
    gpu_experts_per_layer: int
    cpu_expert_bytes: int
    notes: tuple[str, ...]

    def argv(self) -> list[str]:
        args = ["python", "-m", "sglang.launch_server"]
        for flag in self.flags:
            args.append(flag.name)
            if flag.value is not None:
                args.append(flag.value)
        return args


def _gib(n: float) -> str:
    return f"{n / GiB:.2f} GiB"


def plan_launch(
    desc: MoEArchDescriptor,
    profile: HardwareProfile,
    budget: TierBudget,
    *,
    model_path: str | Path,
    expert_weight_path: str | Path,
    expert_quant: str = "Q4_K_M",
) -> LaunchPlan:
    gpu = budget.gpu
    if gpu is None:
        raise InfeasiblePlan("sglang-kt needs a CUDA GPU; the CPU-only fallback is TASKS.md M7.")
    if gpu.compute_capability < MIN_COMPUTE_CAPABILITY:
        cc = ".".join(map(str, gpu.compute_capability))
        raise InfeasiblePlan(f"{gpu.name} has compute capability {cc}; kt-kernel needs 8.0 or newer.")
    if "avx2" not in profile.cpu.flags:
        raise InfeasiblePlan("The LLAMAFILE expert backend needs a CPU with AVX2.")
    if expert_quant not in GGUF_BITS_PER_WEIGHT:
        raise InfeasiblePlan(f"Unknown GGUF quantization {expert_quant!r}; known: {sorted(GGUF_BITS_PER_WEIGHT)}")

    notes: list[str] = []
    dense = desc.dense_params * GPU_WEIGHT_BYTES
    if dense > budget.vram_bytes:
        raise InfeasiblePlan(
            f"Dense weights need {_gib(dense)} of VRAM but only {_gib(budget.vram_bytes)} is usable."
        )

    kv_per_token = desc.kv_cache_bytes_per_token(KV_DTYPE_BYTES)
    remaining = budget.vram_bytes - dense
    target_context = min(desc.max_position_embeddings, TARGET_CONTEXT_TOKENS)
    context = min(target_context, remaining // kv_per_token)
    if context < MIN_CONTEXT_TOKENS:
        raise InfeasiblePlan(
            f"Only {context} tokens of KV cache fit after dense weights; at least "
            f"{MIN_CONTEXT_TOKENS} are needed."
        )

    expert_bytes = desc.expert_params * GPU_WEIGHT_BYTES
    leftover = remaining - context * kv_per_token
    per_layer = min(desc.num_experts, leftover // (desc.num_moe_layers * expert_bytes))
    gpu_experts = per_layer * desc.num_moe_layers * expert_bytes
    context = min(desc.max_position_embeddings, (remaining - gpu_experts) // kv_per_token)
    kv_cache = context * kv_per_token

    cpu_experts = int(desc.routed_expert_params * GGUF_BITS_PER_WEIGHT[expert_quant] / 8)
    if cpu_experts > budget.ram_bytes:
        notes.append(
            f"{expert_quant} experts ({_gib(cpu_experts)}) exceed the RAM budget "
            f"({_gib(budget.ram_bytes)}); loading may fail or page from disk (TASKS.md M8)."
        )

    static = dense + gpu_experts + kv_cache
    mem_fraction = int(static / gpu.total_bytes * 1000) / 1000
    cpu = profile.cpu

    flags = [
        Flag("--model", str(model_path),
             "HF checkpoint; sglang-kt loads attention, embeddings and GPU experts from it."),
        Flag("--kt-weight-path", str(expert_weight_path),
             f"{expert_quant} GGUF experts run on the CPU by kt-kernel."),
        Flag("--kt-method", "LLAMAFILE",
             "GGUF backend: no conversion step, runs on any AVX2 CPU (upstream default AMXINT4 "
             "needs Intel AMX)."),
        Flag("--kt-cpuinfer", str(cpu.physical_cores),
             f"One expert thread per physical core ({cpu.physical_cores}); hyperthreads share "
             "memory bandwidth and slow the kernels down."),
        Flag("--kt-threadpool-count", str(cpu.numa_nodes),
             f"One thread pool per NUMA node ({cpu.numa_nodes}); upstream default is 2."),
        Flag("--kt-num-gpu-experts", str(per_layer),
             f"Experts per MoE layer kept on GPU: {_gib(leftover)} left after dense weights and "
             f"the KV cache; one expert in every layer costs "
             f"{_gib(expert_bytes * desc.num_moe_layers)}."),
        Flag("--mem-fraction-static", f"{mem_fraction:.3f}",
             f"Weights + KV pool = {_gib(static)} of {_gib(gpu.total_bytes)}; the rest "
             f"({_gib(budget.vram_reserve_bytes)} reserve + other processes) stays free for "
             "activations and CUDA context."),
        Flag("--context-length", str(context),
             f"KV cache of {_gib(kv_cache)} at {kv_per_token // 1024} KiB/token"
             + (" (model maximum)." if context == desc.max_position_embeddings else ".")),
        Flag("--chunked-prefill-size", "2048" if gpu.total_bytes < SMALL_GPU_BYTES else "4096",
             "Smaller prefill chunks bound activation memory on GPUs under 8 GB "
             "(heuristic; to be tuned with TASKS.md M5 benchmarks)."),
    ]
    if per_layer > 0:
        flags.append(Flag("--kt-expert-placement-strategy", "uniform",
                          "Equal GPU experts per layer; 'frequency' needs recorded activation "
                          "statistics that don't exist on first launch."))

    return LaunchPlan(
        flags=tuple(flags),
        gpu_dense_bytes=dense,
        gpu_expert_bytes=gpu_experts,
        kv_cache_bytes=kv_cache,
        context_tokens=context,
        gpu_experts_per_layer=per_layer,
        cpu_expert_bytes=cpu_experts,
        notes=tuple(notes),
    )
