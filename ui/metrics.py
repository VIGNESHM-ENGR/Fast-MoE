"""Live resource readings: NVML VRAM, system RAM, and llama-server's Prometheus /metrics."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import psutil

from engine.hardware.profiler import read_cgroup_memory


@dataclass(frozen=True)
class Usage:
    vram_used: int | None
    vram_total: int | None
    ram_used: int
    ram_total: int
    prompt_tps: float | None = None
    gen_tps: float | None = None
    busy: bool = False
    # Model file pages llama-server holds in RAM. Linux reports these as page cache, not "used".
    model_cached: int | None = None


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


def process_file_rss(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    """Bytes of memory-mapped files a process has in RAM (RssFile), or None if unreadable."""
    try:
        for line in (proc_root / str(pid) / "status").read_text().splitlines():
            if line.startswith("RssFile:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def ram_usage(cgroup_root: Path = Path("/sys/fs/cgroup")) -> tuple[int, int]:
    """(used, total) RAM, excluding reclaimable page cache; honours a container memory limit."""
    vm = psutil.virtual_memory()
    limit, _ = read_cgroup_memory(cgroup_root)
    if limit is None:
        return vm.total - vm.available, vm.total
    stat = cgroup_root / "memory.stat"
    if not stat.exists():
        stat = cgroup_root / "memory" / "memory.stat"
    fields = dict(line.split() for line in stat.read_text().splitlines() if line.count(" ") == 1)
    used = int(fields.get("anon", fields.get("rss", 0)))
    return used, min(limit, vm.total)


def read_usage(gpu_index: int | None, server_url: str | None, server_pid: int | None = None) -> Usage:
    vram_used, vram_total = _vram(gpu_index) if gpu_index is not None else (None, None)
    ram_used, ram_total = ram_usage()
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
    model_cached = process_file_rss(server_pid) if server_pid else None
    return Usage(vram_used, vram_total, ram_used, ram_total, prompt_tps, gen_tps, busy, model_cached)
