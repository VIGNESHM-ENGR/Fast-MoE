"""Live resource readings: NVML VRAM, system RAM, and llama-server's Prometheus /metrics."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class Usage:
    vram_used: int | None
    vram_total: int | None
    ram_used: int
    ram_total: int
    prompt_tps: float | None = None
    gen_tps: float | None = None
    busy: bool = False


def parse_prometheus(text: str) -> dict[str, float]:
    values = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        try:
            values[name] = float(value)
        except ValueError:
            continue
    return values


def _vram(gpu_index: int) -> tuple[int | None, int | None]:
    try:
        import pynvml
    except ImportError:
        return None, None
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None, None
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(gpu_index))
        return mem.used, mem.total
    finally:
        pynvml.nvmlShutdown()


def read_usage(gpu_index: int | None, server_url: str | None) -> Usage:
    vram_used, vram_total = _vram(gpu_index) if gpu_index is not None else (None, None)
    vm = psutil.virtual_memory()
    prompt_tps = gen_tps = None
    busy = False
    if server_url:
        try:
            with urllib.request.urlopen(f"{server_url}/metrics", timeout=1) as resp:
                m = parse_prometheus(resp.read().decode())
            prompt_tps = m.get("llamacpp:prompt_tokens_seconds")
            gen_tps = m.get("llamacpp:predicted_tokens_seconds")
            busy = m.get("llamacpp:requests_processing", 0) > 0
        except OSError:
            pass
    return Usage(vram_used, vram_total, vm.total - vm.available, vm.total, prompt_tps, gen_tps, busy)
