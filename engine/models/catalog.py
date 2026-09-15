"""Models Fast-MoE has been tested with, and their model cards' recommended sampling settings.

Any MoE GGUF that llama.cpp supports can still be used by path or repo id; the
catalog only adds a friendly name, facts for the UI, and official defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    top_k: int
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repeat_penalty: float = 1.0


@dataclass(frozen=True)
class CatalogModel:
    name: str
    repo: str
    quant: str
    file_prefix: str  # GGUF file names start with this (case-insensitive)
    download_gib: float
    params: str
    experts: str
    recommended: Sampling
    card_url: str


CATALOG: tuple[CatalogModel, ...] = (
    CatalogModel(
        name="Qwen3.6-35B-A3B",
        repo="ggml-org/Qwen3.6-35B-A3B-GGUF",
        quant="Q4_K_M",
        file_prefix="Qwen3.6-35B-A3B",
        download_gib=19.0,
        params="35B total · 3B active",
        experts="256 experts · 8 active per token",
        # Model card, thinking mode for general tasks.
        recommended=Sampling(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5),
        card_url="https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
    ),
    CatalogModel(
        name="Gemma 4 26B-A4B",
        repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        quant="Q4_K_M",
        file_prefix="gemma-4-26B-A4B",
        download_gib=15.8,
        params="25.2B total · 3.8B active",
        experts="128 experts · 8 active + 1 shared",
        recommended=Sampling(temperature=1.0, top_p=0.95, top_k=64),
        card_url="https://huggingface.co/google/gemma-4-26B-A4B-it",
    ),
    CatalogModel(
        name="Qwen3-30B-A3B",
        repo="Qwen/Qwen3-30B-A3B-GGUF",
        quant="Q4_K_M",
        file_prefix="Qwen3-30B-A3B",
        download_gib=17.3,
        params="30.5B total · 3.3B active",
        experts="128 experts · 8 active per token",
        # Model card, thinking mode.
        recommended=Sampling(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0),
        card_url="https://huggingface.co/Qwen/Qwen3-30B-A3B",
    ),
)


def find(model: str | Path | None) -> CatalogModel | None:
    """Catalog entry for a repo id, a GGUF path, or a models/<repo-name>/ folder."""
    if not model:
        return None
    text = str(model)
    for entry in CATALOG:
        if text == entry.repo:
            return entry
    path = Path(text)
    names = [path.name.lower(), path.parent.name.lower()]
    for entry in CATALOG:
        prefix = entry.file_prefix.lower()
        if any(n.startswith(prefix) for n in names) or path.parent.name == entry.repo.split("/")[-1]:
            return entry
    return None
