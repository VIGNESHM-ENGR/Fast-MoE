"""One-command launcher: `python -m engine.serve [--model REPO_OR_GGUF]`.

1. Resolve the model (a local .gguf, or a Hugging Face GGUF repo downloaded on demand).
2. Ask llama.cpp's `llama-fit-params` where everything fits on this machine.
3. Show where every layer's experts will live (GPU vs RAM), with sizes.
4. Start `llama-server` with exactly those arguments and supervise it.
"""

from __future__ import annotations

import argparse
import os
import sys

from engine.llama.fit import FitError
from engine.llama.layout import describe_ranges
from engine.llama.runner import Runner, RunPlan, RunSettings
from engine.llama.server import ServerError
from engine.models.model_downloader import DEFAULT_QUANT, DEFAULT_REPO

GiB = 1024**3


def print_plan(plan: RunPlan) -> None:
    layout, fit, placement = plan.layout, plan.fit, plan.placement
    on_gpu = [p.layer for p in placement if p.gpu_bytes and not p.cpu_bytes]
    split = [p.layer for p in placement if p.gpu_bytes and p.cpu_bytes]
    on_cpu = [p.layer for p in placement if p.cpu_bytes and not p.gpu_bytes]
    gpu_experts = sum(p.gpu_bytes for p in placement)
    cpu_experts = sum(p.cpu_bytes for p in placement)

    print(f"\nModel   {plan.model_path.name}  ({layout.architecture}, {layout.block_count} layers, "
          f"{layout.expert_count} experts / {layout.experts_per_token} active, {layout.total_bytes / GiB:.1f} GiB)")
    print(f"Context {fit.context or layout.context_length} tokens (model max {layout.context_length})")
    print(f"Experts on GPU  {gpu_experts / GiB:5.1f} GiB  layers {describe_ranges(on_gpu)}")
    if split:
        print(f"Experts split   {'':9} layers {describe_ranges(split)}")
    print(f"Experts in RAM  {cpu_experts / GiB:5.1f} GiB  layers {describe_ranges(on_cpu)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_REPO, help=f"GGUF file or Hugging Face repo (default {DEFAULT_REPO})")
    parser.add_argument("--quant", default=DEFAULT_QUANT, help=f"quantization when --model is a repo (default {DEFAULT_QUANT})")
    parser.add_argument("--host", default=os.environ.get("FAST_MOE_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx", type=int, help="minimum context to keep; more experts move to RAM to make room")
    parser.add_argument("--vram-margin-mib", type=int, help="VRAM to leave free (llama.cpp default 1024)")
    parser.add_argument("--no-mmap", action="store_true",
                        help="load weights into RAM instead of memory-mapping (faster prompts, slower start)")
    parser.add_argument("llama_args", nargs=argparse.REMAINDER, help="extra llama-server arguments after `--`")
    args = parser.parse_args(argv)
    settings = RunSettings(args.model, args.quant, args.host, args.port, args.ctx,
                           args.vram_margin_mib, args.no_mmap, tuple(args.llama_args))

    runner = Runner()
    try:
        print("Working out where the model fits on this machine...")
        took = runner.start(settings, on_plan=print_plan)
        server = runner.server
        print(f"\nReady in {took:.0f}s  OpenAI API: {server.url}/v1   log: {server.log_path}")
        print("Press Ctrl+C to stop.")
        server.proc.wait()
        print(f"llama-server exited with code {server.proc.returncode}; see {server.log_path}", file=sys.stderr)
        return server.proc.returncode or 1
    except (ServerError, FitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopping llama-server...")
        return 0
    finally:
        runner.stop()


if __name__ == "__main__":
    sys.exit(main())
