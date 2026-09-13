"""Hardware prober: detects GPU VRAM, system RAM (cgroup-aware), and the cold-tier
storage directory's backing device + measured sequential throughput.

GPU memory is read through NVML rather than `torch.cuda.mem_get_info`, which
would create a CUDA context (and consume VRAM) just to measure VRAM.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

log = logging.getLogger(__name__)

MiB = 1024**2
GiB = 1024**3

DEFAULT_CACHE_DIR_ENV = "FAST_MOE_NVME_CACHE_DIR"
CONVENTIONAL_NVME_DIR = Path("/mnt/nvme_cache")
DEFAULT_PROBE_BYTES = 256 * MiB
_PROBE_CHUNK = 4 * MiB
# cgroup v1 reports "unlimited" as a huge page-aligned number instead of "max".
_CGROUP_V1_UNLIMITED = 1 << 60


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    uuid: str
    total_bytes: int
    free_bytes: int
    compute_capability: tuple[int, int]


@dataclass(frozen=True)
class CPUInfo:
    physical_cores: int
    logical_threads: int
    numa_nodes: int
    flags: frozenset[str]


@dataclass(frozen=True)
class MemoryInfo:
    total_bytes: int
    available_bytes: int
    cgroup_limit_bytes: int | None


@dataclass(frozen=True)
class StorageInfo:
    path: Path
    device: str | None
    kind: str  # "nvme" | "ssd" | "hdd" | "unknown"
    free_bytes: int
    read_bytes_per_s: float | None = None
    write_bytes_per_s: float | None = None


@dataclass(frozen=True)
class HardwareProfile:
    gpus: tuple[GPUInfo, ...]
    memory: MemoryInfo
    storage: StorageInfo
    cpu: CPUInfo


def probe_gpus() -> tuple[GPUInfo, ...]:
    try:
        import pynvml
    except ImportError:
        log.info("nvidia-ml-py not installed; assuming no GPU")
        return ()
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        log.info("NVML unavailable (%s); assuming no GPU", exc)
        return ()
    try:
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                GPUInfo(
                    index=i,
                    name=pynvml.nvmlDeviceGetName(handle),
                    uuid=pynvml.nvmlDeviceGetUUID(handle),
                    total_bytes=mem.total,
                    free_bytes=mem.free,
                    compute_capability=tuple(pynvml.nvmlDeviceGetCudaComputeCapability(handle)),
                )
            )
    finally:
        pynvml.nvmlShutdown()
    return filter_visible_gpus(tuple(gpus), os.environ.get("CUDA_VISIBLE_DEVICES"))


def parse_cpu_flags(cpuinfo: str) -> frozenset[str]:
    for line in cpuinfo.splitlines():
        if line.startswith("flags"):
            return frozenset(line.split(":", 1)[1].split())
    return frozenset()


def count_numa_nodes(sysfs_root: Path = Path("/sys")) -> int:
    node_dir = sysfs_root / "devices" / "system" / "node"
    nodes = [p for p in node_dir.glob("node*") if p.name[4:].isdigit()] if node_dir.is_dir() else []
    return max(1, len(nodes))


def probe_cpu() -> CPUInfo:
    logical = len(os.sched_getaffinity(0))
    # Physical cores, not hyperthreads: kt-kernel's CPU expert kernels are memory-bandwidth
    # bound and slow down when two threads share a core. Clamped to the affinity mask so
    # `docker --cpuset-cpus` is respected.
    physical = min(psutil.cpu_count(logical=False) or logical, logical)
    return CPUInfo(
        physical_cores=physical,
        logical_threads=logical,
        numa_nodes=count_numa_nodes(),
        flags=parse_cpu_flags(Path("/proc/cpuinfo").read_text()),
    )


def filter_visible_gpus(gpus: tuple[GPUInfo, ...], visible: str | None) -> tuple[GPUInfo, ...]:
    """Apply CUDA_VISIBLE_DEVICES, which NVML itself ignores.

    Mirrors CUDA's parsing: entries are indices or UUID prefixes, and parsing
    stops at the first entry that matches nothing.
    """
    if visible is None:
        return gpus
    selected = []
    for token in (t.strip() for t in visible.split(",")):
        if token.isdigit():
            match = next((g for g in gpus if g.index == int(token)), None)
        elif token.startswith(("GPU-", "MIG-")):
            match = next((g for g in gpus if g.uuid.startswith(token)), None)
        else:
            match = None
        if match is None:
            break
        selected.append(match)
    return tuple(selected)


def read_cgroup_memory(cgroup_root: Path = Path("/sys/fs/cgroup")) -> tuple[int | None, int | None]:
    """Return (limit, current usage) for this process's cgroup, or (None, None).

    psutil reports host memory even inside a memory-limited container, so the
    cgroup limit must be checked separately for Docker to be zero-config.
    """
    v2_max = cgroup_root / "memory.max"
    if v2_max.exists():
        raw = v2_max.read_text().strip()
        if raw == "max":
            return None, None
        return int(raw), int((cgroup_root / "memory.current").read_text())
    v1_limit = cgroup_root / "memory" / "memory.limit_in_bytes"
    if v1_limit.exists():
        limit = int(v1_limit.read_text())
        if limit >= _CGROUP_V1_UNLIMITED:
            return None, None
        return limit, int((cgroup_root / "memory" / "memory.usage_in_bytes").read_text())
    return None, None


def probe_memory(cgroup_root: Path = Path("/sys/fs/cgroup")) -> MemoryInfo:
    vm = psutil.virtual_memory()
    total, available = vm.total, vm.available
    limit, usage = read_cgroup_memory(cgroup_root)
    if limit is not None:
        total = min(total, limit)
        available = min(available, max(0, limit - usage))
    return MemoryInfo(total_bytes=total, available_bytes=available, cgroup_limit_bytes=limit)


def classify_block_device(major: int, minor: int, sysfs_root: Path = Path("/sys")) -> tuple[str | None, str]:
    """Map a device number to (whole-disk name, kind), following partitions and
    device-mapper (LVM/LUKS) down to the physical disk."""
    dev = sysfs_root / "dev" / "block" / f"{major}:{minor}"
    if not dev.exists():
        return None, "unknown"
    return _classify_sysfs_block(dev.resolve(), depth=0)


def _classify_sysfs_block(dev: Path, depth: int) -> tuple[str | None, str]:
    if (dev / "partition").exists():
        dev = dev.parent
    slaves = sorted((dev / "slaves").iterdir()) if (dev / "slaves").is_dir() else []
    if slaves and depth < 8:
        return _classify_sysfs_block(slaves[0].resolve(), depth + 1)
    name = dev.name
    if name.startswith("nvme"):
        return name, "nvme"
    rotational = dev / "queue" / "rotational"
    if rotational.exists():
        return name, "hdd" if rotational.read_text().strip() == "1" else "ssd"
    return name, "unknown"


def _mount_source(path: Path, mountinfo: Path = Path("/proc/self/mountinfo")) -> str | None:
    """Source device of the deepest mount containing `path` (e.g. '/dev/nvme0n1p2')."""
    best: tuple[int, str] | None = None
    for line in mountinfo.read_text().splitlines():
        fields = line.split()
        mount_point = fields[4]
        source = fields[fields.index("-") + 2]
        contains = path == Path(mount_point) or Path(mount_point) in path.parents
        if contains and (best is None or len(mount_point) > best[0]):
            best = (len(mount_point), source)
    return best[1] if best else None


def probe_storage_device(path: Path, sysfs_root: Path = Path("/sys")) -> tuple[str | None, str]:
    st = os.stat(path)
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    if major != 0:
        return classify_block_device(major, minor, sysfs_root)
    # btrfs/zfs/overlay use anonymous device numbers; fall back to the mount source.
    source = _mount_source(path.resolve())
    if source and source.startswith("/dev/"):
        block = sysfs_root / "class" / "block" / Path(source).name
        if block.exists():
            return _classify_sysfs_block(block.resolve(), depth=0)
    return None, "unknown"


def measure_throughput(directory: Path, size_bytes: int = DEFAULT_PROBE_BYTES) -> tuple[float, float]:
    """Sequential (read, write) bytes/s using a temporary file in `directory`."""
    # Random data so compressing/deduplicating filesystems can't inflate the result.
    chunk = os.urandom(_PROBE_CHUNK)
    fd, name = tempfile.mkstemp(dir=directory, prefix=".fast-moe-probe-")
    try:
        start = time.perf_counter()
        written = 0
        while written < size_bytes:
            view = memoryview(chunk)[: min(_PROBE_CHUNK, size_bytes - written)]
            while view:
                n = os.write(fd, view)
                view = view[n:]
                written += n
        os.fsync(fd)
        write_s = time.perf_counter() - start

        # Evict the file from page cache so the read hits the device, not RAM.
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.lseek(fd, 0, os.SEEK_SET)
        start = time.perf_counter()
        while os.read(fd, _PROBE_CHUNK):
            pass
        read_s = time.perf_counter() - start
    finally:
        os.close(fd)
        os.unlink(name)
    return written / read_s, written / write_s


def resolve_cache_dir(configured: str | os.PathLike | None = None) -> Path:
    """Pick the cold-tier directory: explicit config > env var > /mnt/nvme_cache > ~/.cache."""
    if configured:
        path = Path(configured)
    elif os.environ.get(DEFAULT_CACHE_DIR_ENV):
        path = Path(os.environ[DEFAULT_CACHE_DIR_ENV])
    elif CONVENTIONAL_NVME_DIR.is_dir() and os.access(CONVENTIONAL_NVME_DIR, os.W_OK):
        path = CONVENTIONAL_NVME_DIR
    else:
        path = Path.home() / ".cache" / "fast-moe" / "swap"
    path.mkdir(parents=True, exist_ok=True)
    return path


def probe_storage(directory: Path, probe_bytes: int = DEFAULT_PROBE_BYTES) -> StorageInfo:
    device, kind = probe_storage_device(directory)
    free = psutil.disk_usage(str(directory)).free
    read_bps = write_bps = None
    if probe_bytes > 0:
        if free > 2 * probe_bytes:
            read_bps, write_bps = measure_throughput(directory, probe_bytes)
        else:
            log.warning("Skipping throughput probe: only %d MiB free in %s", free // MiB, directory)
    return StorageInfo(directory, device, kind, free, read_bps, write_bps)


def profile_hardware(
    cache_dir: str | os.PathLike | None = None, probe_bytes: int = DEFAULT_PROBE_BYTES
) -> HardwareProfile:
    return HardwareProfile(
        gpus=probe_gpus(),
        memory=probe_memory(),
        storage=probe_storage(resolve_cache_dir(cache_dir), probe_bytes),
        cpu=probe_cpu(),
    )
