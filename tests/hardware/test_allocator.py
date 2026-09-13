from dataclasses import replace

import pytest

from engine.hardware.allocator import AllocatorSettings, plan_budget
from engine.hardware.profiler import GiB
from tests.factories import gpu, make_profile


def test_laptop_6gb_uses_minimum_reserve():
    # 15% of 6GiB is 0.9GiB, below the 1GiB floor.
    budget = plan_budget(make_profile([gpu(6, 5.656)], ram_total=38, ram_avail=31))
    assert budget.vram_reserve_bytes == 1 * GiB
    assert budget.vram_bytes == int(5.656 * GiB) - 1 * GiB
    assert budget.ram_bytes == int(31 * GiB) - int(3.8 * GiB)
    assert budget.pinned_ram_bytes == int(budget.ram_bytes * 0.10)
    assert budget.notes == ()


def test_workstation_24gb_uses_percentage_reserve_and_pinned_cap():
    budget = plan_budget(make_profile([gpu(24, 23.5)], ram_total=128, ram_avail=120, disk_free=2000))
    reserve = int(24 * GiB * 0.15)
    assert budget.vram_reserve_bytes == reserve
    assert budget.vram_bytes == int(23.5 * GiB) - reserve
    assert budget.pinned_ram_bytes == 4 * GiB
    assert budget.disk_bytes == 2000 * GiB - 200 * GiB


def test_small_disk_uses_minimum_headroom():
    budget = plan_budget(make_profile([gpu(8, 7.5)], disk_free=20))
    assert budget.disk_bytes == 15 * GiB


def test_no_gpu_is_cpu_only():
    budget = plan_budget(make_profile())
    assert budget.gpu is None
    assert budget.vram_bytes == 0
    assert budget.pinned_ram_bytes == 0
    assert "No GPU detected" in budget.notes[0]


def test_cpu_only_setting_ignores_gpu():
    budget = plan_budget(make_profile([gpu(24, 23)]), AllocatorSettings(cpu_only=True))
    assert budget.gpu is None and budget.vram_bytes == 0


def test_gpu_mostly_used_by_other_processes_falls_back_to_cpu():
    budget = plan_budget(make_profile([gpu(8, 1.5)]))
    assert budget.gpu is None
    assert "running CPU-only" in budget.notes[0]


def test_picks_gpu_with_most_free_memory():
    budget = plan_budget(make_profile([gpu(24, 2, index=0), gpu(12, 11, index=1)]))
    assert budget.gpu.index == 1


def test_user_caps_apply_to_every_tier():
    settings = AllocatorSettings(max_vram_bytes=2 * GiB, max_ram_bytes=8 * GiB, max_disk_bytes=50 * GiB)
    budget = plan_budget(make_profile([gpu(24, 23)], ram_total=128, ram_avail=120), settings)
    assert (budget.vram_bytes, budget.ram_bytes, budget.disk_bytes) == (2 * GiB, 8 * GiB, 50 * GiB)


def test_ram_never_negative_when_nearly_exhausted():
    budget = plan_budget(make_profile(ram_total=16, ram_avail=1))
    assert budget.ram_bytes == 0


@pytest.mark.parametrize("kind,read_mib,expect_note", [
    ("hdd", 150, True),
    ("ssd", 300, True),
    ("nvme", 3000, False),
    ("unknown", 1200, False),  # e.g. Docker overlay root that measured fast
    ("unknown", None, True),
    ("hdd", None, True),
    ("nvme", None, False),
])
def test_cold_tier_speed_notes(kind, read_mib, expect_note):
    profile = make_profile([gpu(8, 7.5)], kind=kind, read_mib=read_mib or 0)
    if read_mib is None:
        profile = replace(profile, storage=replace(profile.storage, read_bytes_per_s=None))
    assert bool(plan_budget(profile).notes) is expect_note
