"""Budget allocator: turns a HardwareProfile into per-tier byte budgets.

This produces tier *totals* only. Splitting VRAM between dense weights, KV
cache and hot experts needs model sizes, so that happens once the MoE
descriptor exists (TASKS.md M2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engine.hardware.profiler import GiB, GPUInfo, HardwareProfile, MiB

# A CUDA context + cuBLAS/cuDNN workspaces alone take several hundred MiB, so a
# pure percentage reserve under-reserves on small (6-8GB) GPUs.
MIN_VRAM_RESERVE_BYTES = 1 * GiB
# Below this, offloading dense layers to the GPU costs more in PCIe transfers than it saves.
MIN_USEFUL_VRAM_BYTES = 1 * GiB

RAM_HEADROOM_FRACTION = 0.10
RAM_HEADROOM_MIN_BYTES = 2 * GiB
# Page-locked memory can't be swapped by the OS; pinning too much starves the system.
PINNED_FRACTION = 0.10
PINNED_MAX_BYTES = 4 * GiB

DISK_HEADROOM_FRACTION = 0.10
DISK_HEADROOM_MIN_BYTES = 5 * GiB

SLOW_COLD_TIER_BYTES_PER_S = 500 * MiB


@dataclass(frozen=True)
class AllocatorSettings:
    """User-facing knobs (config file / Gradio UI). `max_*` are caps; None = no cap."""

    vram_reserve_fraction: float = 0.15
    max_vram_bytes: int | None = None
    max_ram_bytes: int | None = None
    max_disk_bytes: int | None = None
    cpu_only: bool = False


@dataclass(frozen=True)
class TierBudget:
    gpu: GPUInfo | None
    vram_reserve_bytes: int
    vram_bytes: int
    ram_bytes: int
    pinned_ram_bytes: int  # carved out of ram_bytes, used for async host<->device copies
    disk_path: Path
    disk_bytes: int
    notes: tuple[str, ...]


def _cap(value: int, cap: int | None) -> int:
    return value if cap is None else min(value, cap)


def plan_budget(profile: HardwareProfile, settings: AllocatorSettings | None = None) -> TierBudget:
    settings = settings or AllocatorSettings()
    notes: list[str] = []

    gpu = None
    vram_reserve = vram = 0
    if settings.cpu_only:
        notes.append("CPU-only mode requested; VRAM tier disabled.")
    elif not profile.gpus:
        notes.append("No GPU detected; running CPU-only.")
    else:
        # v1 targets a single GPU: use the one with the most free memory.
        candidate = max(profile.gpus, key=lambda g: g.free_bytes)
        reserve = max(int(candidate.total_bytes * settings.vram_reserve_fraction), MIN_VRAM_RESERVE_BYTES)
        usable = _cap(max(0, candidate.free_bytes - reserve), settings.max_vram_bytes)
        if usable < MIN_USEFUL_VRAM_BYTES:
            notes.append(
                f"GPU {candidate.name} has only {usable / GiB:.2f} GiB usable after reserves; "
                "running CPU-only."
            )
        else:
            gpu, vram_reserve, vram = candidate, reserve, usable

    mem = profile.memory
    ram_headroom = max(int(mem.total_bytes * RAM_HEADROOM_FRACTION), RAM_HEADROOM_MIN_BYTES)
    ram = _cap(max(0, mem.available_bytes - ram_headroom), settings.max_ram_bytes)
    pinned = min(int(ram * PINNED_FRACTION), PINNED_MAX_BYTES) if gpu else 0

    storage = profile.storage
    disk_headroom = max(int(storage.free_bytes * DISK_HEADROOM_FRACTION), DISK_HEADROOM_MIN_BYTES)
    disk = _cap(max(0, storage.free_bytes - disk_headroom), settings.max_disk_bytes)
    # A measured speed beats the device-type guess (overlay/network mounts classify as "unknown").
    if storage.read_bytes_per_s is not None:
        if storage.read_bytes_per_s < SLOW_COLD_TIER_BYTES_PER_S:
            notes.append(
                f"Cold tier {storage.path} reads at only {storage.read_bytes_per_s / MiB:.0f} MiB/s; "
                "expect slow swaps."
            )
    elif storage.kind not in ("nvme", "ssd"):
        notes.append(f"Cold tier {storage.path} is on a device of type '{storage.kind}'; expect slow swaps.")

    return TierBudget(
        gpu=gpu,
        vram_reserve_bytes=vram_reserve,
        vram_bytes=vram,
        ram_bytes=ram,
        pinned_ram_bytes=pinned,
        disk_path=storage.path,
        disk_bytes=disk,
        notes=tuple(notes),
    )
