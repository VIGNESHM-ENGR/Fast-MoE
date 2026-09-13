"""Run llama.cpp's `llama-fit-params` and parse the CLI arguments it prints.

This is llama.cpp's documented way to see what `--fit` decides: it prints
`-c <ctx> -ngl <layers> -ot "<regex>=CPU,..."` on the last stdout line. Passing
those arguments to `llama-server` with `--fit off` makes the placement shown in
the UI exactly the placement that runs.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path


class FitError(RuntimeError):
    pass


@dataclass(frozen=True)
class FitResult:
    args: tuple[str, ...]
    context: int | None
    gpu_layers: int | None
    cpu_patterns: tuple[str, ...]


def parse_fit_args(stdout: str) -> FitResult:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise FitError("llama-fit-params printed no arguments")
    args = shlex.split(lines[-1])
    context = gpu_layers = None
    cpu_patterns: list[str] = []
    for flag, value in pairwise(args):
        if flag in ("-c", "--ctx-size"):
            context = int(value)
        elif flag in ("-ngl", "--n-gpu-layers", "--gpu-layers"):
            gpu_layers = int(value)
        elif flag in ("-ot", "--override-tensor"):
            for entry in value.split(","):
                pattern, _, buffer_type = entry.rpartition("=")
                if buffer_type.upper() == "CPU":
                    cpu_patterns.append(pattern)
    return FitResult(tuple(args), context, gpu_layers, tuple(cpu_patterns))


def run_fit_params(binary: Path, model: Path, extra_args: list[str], timeout_s: int = 600) -> FitResult:
    proc = subprocess.run([str(binary), "-m", str(model), *extra_args],
                          capture_output=True, text=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise FitError(f"llama-fit-params failed (exit {proc.returncode}):\n{tail}")
    return parse_fit_args(proc.stdout)
