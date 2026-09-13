"""One-command launcher: `python -m engine.serve [--model REPO_OR_GGUF]`.

1. Resolve the model (a local .gguf, or a Hugging Face GGUF repo downloaded on demand).
2. Ask llama.cpp's `llama-fit-params` where everything fits on this machine.
3. Show where every layer's experts will live (GPU vs RAM), with sizes.
4. Start `llama-server` with exactly those arguments and supervise it.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from engine.llama.fit import FitError, FitResult, run_fit_params
from engine.llama.layout import ModelLayout, describe_ranges, expert_placement, read_layout
from engine.llama.server import LlamaServer, ServerError, find_binary, port_in_use
from engine.models.model_downloader import DEFAULT_QUANT, DEFAULT_REPO, download, local_model

GiB = 1024**3
LOG_DIR = Path.home() / ".cache" / "fast-moe" / "logs"


def resolve_model(model: str, quant: str) -> Path:
    path = Path(model).expanduser()
    if path.suffix == ".gguf":
        if not path.is_file():
            raise SystemExit(f"Model file not found: {path}")
        return path
    existing = local_model(model, quant)
    if existing:
        return existing
    print(f"{model} ({quant}) is not downloaded yet; downloading...")
    return download(model, quant)[0]


def fit_args(args: argparse.Namespace) -> list[str]:
    extra = []
    if args.ctx:
        extra += ["--fit-ctx", str(args.ctx)]
    if args.vram_margin_mib is not None:
        extra += ["--fit-target", str(args.vram_margin_mib)]
    return extra


def print_plan(model: Path, layout: ModelLayout, fit: FitResult) -> None:
    gpu_layers = fit.gpu_layers if fit.gpu_layers is not None else 0
    placement = expert_placement(layout, gpu_layers, fit.cpu_patterns)
    on_gpu = [p.layer for p in placement if p.gpu_bytes and not p.cpu_bytes]
    split = [p.layer for p in placement if p.gpu_bytes and p.cpu_bytes]
    on_cpu = [p.layer for p in placement if p.cpu_bytes and not p.gpu_bytes]
    gpu_experts = sum(p.gpu_bytes for p in placement)
    cpu_experts = sum(p.cpu_bytes for p in placement)

    print(f"\nModel   {model.name}  ({layout.architecture}, {layout.block_count} layers, "
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx", type=int, help="minimum context to keep; more experts move to RAM to make room")
    parser.add_argument("--vram-margin-mib", type=int, help="VRAM to leave free (llama.cpp default 1024)")
    parser.add_argument("--no-mmap", action="store_true",
                        help="load weights into RAM instead of memory-mapping (faster prompts, slower start)")
    parser.add_argument("llama_args", nargs=argparse.REMAINDER,
                        help="extra llama-server arguments after `--`")
    args = parser.parse_args(argv)

    try:
        server_bin, fit_bin = find_binary("llama-server"), find_binary("llama-fit-params")
        if port_in_use(args.host, args.port):
            raise ServerError(f"Port {args.port} is already in use; pick another with --port.")
        model = resolve_model(args.model, args.quant)

        print("Working out where the model fits on this machine...")
        with ThreadPoolExecutor(max_workers=2) as pool:
            fit_future = pool.submit(run_fit_params, fit_bin, model, fit_args(args))
            layout_future = pool.submit(read_layout, model)
            fit, layout = fit_future.result(), layout_future.result()
        print_plan(model, layout, fit)

        server_args = ["-m", str(model), "--jinja", "--metrics", "--fit", "off", *fit.args]
        if args.no_mmap:
            server_args += ["--load-mode", "none"]
        server_args += [a for a in args.llama_args if a != "--"]
        log_path = LOG_DIR / f"llama-server-{time.strftime('%Y%m%d-%H%M%S')}.log"
        server = LlamaServer(server_bin, server_args, args.host, args.port, log_path)
        server.start()
    except (ServerError, FitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        took = server.wait_healthy()
        print(f"\nReady in {took:.0f}s  OpenAI API: {server.url}/v1   log: {log_path}")
        print("Press Ctrl+C to stop.")
        server.proc.wait()
        print(f"llama-server exited with code {server.proc.returncode}; see {log_path}", file=sys.stderr)
        return server.proc.returncode or 1
    except ServerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopping llama-server...")
        return 0
    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
