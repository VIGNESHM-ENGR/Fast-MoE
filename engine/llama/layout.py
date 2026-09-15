"""Model layout from GGUF metadata, and where each layer's expert weights live.

Tensor names and sizes come from llama.cpp's `gguf` package. Placement follows
llama.cpp's own rules: `-ngl N` offloads the last N layers
(`i_gpu_start = n_layer + 1 - N` in llama-model.cpp) and `-ot` patterns are
matched with a regex search on tensor names (llama-model-loader.cpp).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

EXPERT_TENSOR = re.compile(r"^blk\.(\d+)\.ffn_(gate_up|gate|up|down)_(?:ch)?exps\.")


@dataclass(frozen=True)
class ExpertTensor:
    name: str
    layer: int
    n_bytes: int


@dataclass(frozen=True)
class ModelLayout:
    architecture: str
    block_count: int
    expert_count: int
    experts_per_token: int
    context_length: int
    total_bytes: int
    expert_tensors: tuple[ExpertTensor, ...]
    # llama.cpp always keeps input embeddings on the CPU (dev_input = cpu_dev in llama-model.cpp).
    embedding_bytes: int = 0


@dataclass(frozen=True)
class LayerPlacement:
    layer: int
    gpu_bytes: int
    cpu_bytes: int


def _field(reader, key: str):
    field = reader.fields.get(key)
    if field is None:
        return None
    value = field.parts[field.data[0]]
    return bytes(value).decode() if field.types and field.types[0].name == "STRING" else int(value[0])


def read_layout(path: Path) -> ModelLayout:
    from gguf import GGUFReader

    reader = GGUFReader(path)
    arch = _field(reader, "general.architecture")
    experts = []
    embedding_bytes = 0
    for tensor in reader.tensors:
        match = EXPERT_TENSOR.match(tensor.name)
        if match:
            experts.append(ExpertTensor(tensor.name, int(match.group(1)), int(tensor.n_bytes)))
        elif tensor.name.startswith("token_embd."):
            embedding_bytes += int(tensor.n_bytes)
    return ModelLayout(
        architecture=arch,
        block_count=_field(reader, f"{arch}.block_count"),
        expert_count=_field(reader, f"{arch}.expert_count") or 0,
        experts_per_token=_field(reader, f"{arch}.expert_used_count") or 0,
        context_length=_field(reader, f"{arch}.context_length"),
        total_bytes=sum(int(t.n_bytes) for t in reader.tensors),
        expert_tensors=tuple(experts),
        embedding_bytes=embedding_bytes,
    )


def layers_fully_on_cpu(layout: ModelLayout, gpu_layers: int) -> list[int]:
    """Layers `-ngl` leaves off the GPU entirely: attention and KV cache run on the CPU too."""
    return list(range(max(layout.block_count + 1 - gpu_layers, 0)))


def expert_placement(layout: ModelLayout, gpu_layers: int, cpu_patterns: tuple[str, ...]) -> tuple[LayerPlacement, ...]:
    first_gpu_layer = max(layout.block_count + 1 - gpu_layers, 0)
    patterns = [re.compile(p) for p in cpu_patterns]
    gpu = dict.fromkeys(range(layout.block_count), 0)
    cpu = dict.fromkeys(range(layout.block_count), 0)
    for t in layout.expert_tensors:
        on_cpu = t.layer < first_gpu_layer or any(p.search(t.name) for p in patterns)
        (cpu if on_cpu else gpu)[t.layer] += t.n_bytes
    return tuple(LayerPlacement(i, gpu[i], cpu[i]) for i in range(layout.block_count))


def describe_ranges(layers: list[int]) -> str:
    """[0,1,2,5,6] -> '0-2, 5-6'"""
    ranges: list[str] = []
    for layer in sorted(layers):
        if ranges and int(ranges[-1].split("-")[-1]) == layer - 1:
            ranges[-1] = f"{ranges[-1].split('-')[0]}-{layer}"
        else:
            ranges.append(str(layer))
    return ", ".join(ranges) or "none"
