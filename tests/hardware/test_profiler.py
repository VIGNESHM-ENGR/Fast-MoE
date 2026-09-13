import os
from pathlib import Path
from types import SimpleNamespace

from engine.hardware import profiler
from engine.hardware.profiler import GiB, GPUInfo, MiB


def gpus():
    return (
        GPUInfo(0, "a", "GPU-aaaa-1111", 8 * GiB, 8 * GiB),
        GPUInfo(1, "b", "GPU-bbbb-2222", 24 * GiB, 24 * GiB),
    )


def test_visible_devices_unset_keeps_all():
    assert profiler.filter_visible_gpus(gpus(), None) == gpus()


def test_visible_devices_by_index_and_uuid_prefix():
    assert [g.index for g in profiler.filter_visible_gpus(gpus(), "1")] == [1]
    assert [g.index for g in profiler.filter_visible_gpus(gpus(), "GPU-bbbb,0")] == [1, 0]


def test_visible_devices_stops_at_first_invalid_entry():
    assert [g.index for g in profiler.filter_visible_gpus(gpus(), "0,7,1")] == [0]
    assert profiler.filter_visible_gpus(gpus(), "") == ()


def test_cgroup_v2_limit(tmp_path):
    (tmp_path / "memory.max").write_text(f"{16 * GiB}\n")
    (tmp_path / "memory.current").write_text(f"{4 * GiB}\n")
    assert profiler.read_cgroup_memory(tmp_path) == (16 * GiB, 4 * GiB)


def test_cgroup_v2_unlimited(tmp_path):
    (tmp_path / "memory.max").write_text("max\n")
    assert profiler.read_cgroup_memory(tmp_path) == (None, None)


def test_cgroup_v1_unlimited_sentinel(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert profiler.read_cgroup_memory(tmp_path) == (None, None)


def test_no_cgroup_files(tmp_path):
    assert profiler.read_cgroup_memory(tmp_path) == (None, None)


def test_memory_clamped_to_cgroup_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(profiler.psutil, "virtual_memory",
                        lambda: SimpleNamespace(total=128 * GiB, available=100 * GiB))
    (tmp_path / "memory.max").write_text(str(16 * GiB))
    (tmp_path / "memory.current").write_text(str(6 * GiB))
    mem = profiler.probe_memory(tmp_path)
    assert (mem.total_bytes, mem.available_bytes, mem.cgroup_limit_bytes) == (16 * GiB, 10 * GiB, 16 * GiB)


def _fake_disk(sysfs: Path, devices_path: str, rotational: str | None = None) -> Path:
    disk = sysfs / "devices" / devices_path
    disk.mkdir(parents=True)
    if rotational is not None:
        (disk / "queue").mkdir()
        (disk / "queue" / "rotational").write_text(rotational)
    return disk


def _link(sysfs: Path, devnum: str, target: Path) -> None:
    (sysfs / "dev" / "block").mkdir(parents=True, exist_ok=True)
    (sysfs / "dev" / "block" / devnum).symlink_to(target)


def test_nvme_partition_resolves_to_disk(tmp_path):
    disk = _fake_disk(tmp_path, "pci0/nvme/nvme0/nvme0n1", rotational="0")
    part = disk / "nvme0n1p4"
    part.mkdir()
    (part / "partition").write_text("4")
    _link(tmp_path, "259:10", part)
    assert profiler.classify_block_device(259, 10, tmp_path) == ("nvme0n1", "nvme")


def test_sata_ssd_and_hdd_by_rotational_flag(tmp_path):
    _link(tmp_path, "8:0", _fake_disk(tmp_path, "pci0/ata1/sda", rotational="0"))
    _link(tmp_path, "8:16", _fake_disk(tmp_path, "pci0/ata2/sdb", rotational="1"))
    assert profiler.classify_block_device(8, 0, tmp_path) == ("sda", "ssd")
    assert profiler.classify_block_device(8, 16, tmp_path) == ("sdb", "hdd")


def test_device_mapper_follows_slaves_to_physical_disk(tmp_path):
    disk = _fake_disk(tmp_path, "pci0/nvme/nvme1/nvme1n1", rotational="0")
    part = disk / "nvme1n1p2"
    part.mkdir()
    (part / "partition").write_text("2")
    dm = _fake_disk(tmp_path, "virtual/block/dm-0")
    (dm / "slaves").mkdir()
    (dm / "slaves" / "nvme1n1p2").symlink_to(part)
    _link(tmp_path, "252:0", dm)
    assert profiler.classify_block_device(252, 0, tmp_path) == ("nvme1n1", "nvme")


def test_unknown_device_number(tmp_path):
    assert profiler.classify_block_device(1, 2, tmp_path) == (None, "unknown")


def test_throughput_probe_measures_and_cleans_up(tmp_path):
    read_bps, write_bps = profiler.measure_throughput(tmp_path, size_bytes=8 * MiB + 123)
    assert read_bps > 0 and write_bps > 0
    assert list(tmp_path.iterdir()) == []


def test_cache_dir_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(profiler, "CONVENTIONAL_NVME_DIR", tmp_path / "absent")
    monkeypatch.setattr(profiler.Path, "home", lambda: tmp_path / "home")
    monkeypatch.delenv(profiler.DEFAULT_CACHE_DIR_ENV, raising=False)
    assert profiler.resolve_cache_dir() == tmp_path / "home" / ".cache" / "fast-moe" / "swap"

    monkeypatch.setenv(profiler.DEFAULT_CACHE_DIR_ENV, str(tmp_path / "env"))
    assert profiler.resolve_cache_dir() == tmp_path / "env"
    assert profiler.resolve_cache_dir(tmp_path / "cfg") == tmp_path / "cfg"
    assert (tmp_path / "cfg").is_dir()


def test_probe_storage_on_real_tmp_dir(tmp_path):
    info = profiler.probe_storage(tmp_path, probe_bytes=0)
    assert info.free_bytes > 0 and info.read_bytes_per_s is None
    assert info.kind in {"nvme", "ssd", "hdd", "unknown"}
    assert os.path.samefile(info.path, tmp_path)
