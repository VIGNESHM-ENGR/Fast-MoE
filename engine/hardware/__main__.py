"""`python -m engine.hardware`: print the detected hardware profile and tier budget."""

from __future__ import annotations

import argparse
import logging

from engine.hardware.allocator import AllocatorSettings, plan_budget
from engine.hardware.profiler import DEFAULT_PROBE_BYTES, GiB, MiB, profile_hardware


def _gib(n: int) -> str:
    return f"{n / GiB:.2f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", help="cold-tier directory (default: auto)")
    parser.add_argument("--probe-mib", type=int, default=DEFAULT_PROBE_BYTES // MiB,
                        help="throughput probe size in MiB; 0 disables the probe")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    profile = profile_hardware(args.cache_dir, args.probe_mib * MiB)
    budget = plan_budget(profile, AllocatorSettings(cpu_only=args.cpu_only))

    print("== Hardware profile ==")
    for g in profile.gpus:
        print(f"GPU {g.index}: {g.name}  total {_gib(g.total_bytes)}  free {_gib(g.free_bytes)}")
    if not profile.gpus:
        print("GPU: none")
    m = profile.memory
    limit = _gib(m.cgroup_limit_bytes) if m.cgroup_limit_bytes else "none"
    print(f"RAM: total {_gib(m.total_bytes)}  available {_gib(m.available_bytes)}  cgroup limit {limit}")
    s = profile.storage
    speed = (f"read {s.read_bytes_per_s / MiB:.0f} MiB/s  write {s.write_bytes_per_s / MiB:.0f} MiB/s"
             if s.read_bytes_per_s else "not probed")
    print(f"Cold tier: {s.path} on {s.device or '?'} ({s.kind})  free {_gib(s.free_bytes)}  {speed}")
    print(f"CPU threads: {profile.cpu_threads}")

    print("\n== Tier budget ==")
    print(f"VRAM (hot):  {_gib(budget.vram_bytes)}"
          + (f" on {budget.gpu.name}, {_gib(budget.vram_reserve_bytes)} reserved" if budget.gpu else ""))
    print(f"RAM (warm):  {_gib(budget.ram_bytes)}  (pinned {_gib(budget.pinned_ram_bytes)})")
    print(f"Disk (cold): {_gib(budget.disk_bytes)} at {budget.disk_path}")
    for note in budget.notes:
        print(f"note: {note}")


if __name__ == "__main__":
    main()
