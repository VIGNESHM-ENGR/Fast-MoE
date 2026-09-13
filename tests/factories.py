from pathlib import Path

from engine.hardware.profiler import (
    CPUInfo,
    GiB,
    GPUInfo,
    HardwareProfile,
    MemoryInfo,
    MiB,
    StorageInfo,
)

LAPTOP_CPU = CPUInfo(6, 12, 1, frozenset({"avx2", "avx512f", "avx512_vnni"}))


def gpu(total_gib: float, free_gib: float, index: int = 0, cc: tuple[int, int] = (8, 6)) -> GPUInfo:
    return GPUInfo(index, f"gpu{index}", f"GPU-{index}", int(total_gib * GiB), int(free_gib * GiB), cc)


def make_profile(gpus=(), ram_total=32, ram_avail=28, disk_free=500, kind="nvme", read_mib=3000,
                 cpu=LAPTOP_CPU):
    return HardwareProfile(
        gpus=tuple(gpus),
        memory=MemoryInfo(int(ram_total * GiB), int(ram_avail * GiB), None),
        storage=StorageInfo(Path("/mnt/nvme_cache"), "nvme0n1", kind, int(disk_free * GiB),
                            read_mib * MiB, read_mib * MiB),
        cpu=cpu,
    )
