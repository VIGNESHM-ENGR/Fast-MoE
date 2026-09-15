"""Decode speed at different context sizes: `python benchmarks/bench_context.py MODEL.gguf [--ctx 4096 16384]`.

A bigger context needs a bigger KV cache in VRAM, so llama.cpp's fit moves expert weights and,
past a point, whole layers (attention included) onto the CPU. For each context this fits the
model the same way the dashboard does, starts llama-server, generates a fixed number of tokens
and reports llama.cpp's own timings with NVML GPU utilisation sampled during generation.
Stop anything else using the GPU first: the fit measures free VRAM.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

import psutil
import pynvml

from engine.llama.layout import describe_ranges
from engine.llama.runner import Runner, RunSettings, cpu_attention_layers

GiB = 1024**3
PROMPT = "Explain how a hash map works, then write a short Python example."


def chat(url: str, max_tokens: int) -> dict:
    body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.load(resp)["timings"]


def sample(handle, pid: int, stop: threading.Event, gpu: list[int], cpu: list[float]) -> None:
    proc = psutil.Process(pid)
    proc.cpu_percent()
    while not stop.wait(0.5):
        gpu.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        cpu.append(proc.cpu_percent())


def bench(model: str, ctx: int, tokens: int, handle) -> dict:
    runner = Runner()
    try:
        load = runner.start(RunSettings(model=model, ctx=ctx))
        plan, server = runner.plan, runner.server
        chat(server.url, 8)  # warm-up: page experts in and compile kernels
        stop, gpu, cpu = threading.Event(), [], []
        sampler = threading.Thread(target=sample, args=(handle, server.proc.pid, stop, gpu, cpu))
        sampler.start()
        timings = chat(server.url, tokens)
        stop.set()
        sampler.join()
        placement = plan.placement
        return {
            "ctx": plan.fit.context,
            "ngl": plan.fit.gpu_layers,
            "cpu_layers": describe_ranges(cpu_attention_layers(plan)) if cpu_attention_layers(plan) else "none",
            "experts_gpu_gib": round(sum(p.gpu_bytes for p in placement) / GiB, 2),
            "experts_ram_gib": round(sum(p.cpu_bytes for p in placement) / GiB, 2),
            "vram_used_mib": pynvml.nvmlDeviceGetMemoryInfo(handle).used // 2**20,
            "load_s": round(load, 1),
            "prompt_tps": round(timings["prompt_per_second"], 1),
            "gen_tps": round(timings["predicted_per_second"], 2),
            "gpu_util_pct": round(statistics.mean(gpu)) if gpu else None,
            "server_cpu_pct": round(statistics.mean(cpu)) if cpu else None,
        }
    finally:
        runner.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="GGUF file or Hugging Face repo")
    parser.add_argument("--ctx", type=int, nargs="+", default=[4096, 16384, 32768, 65536])
    parser.add_argument("--tokens", type=int, default=256, help="tokens to generate per run")
    args = parser.parse_args()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    results = []
    for ctx in args.ctx:
        result = bench(args.model, ctx, args.tokens, handle)
        results.append(result)
        print(json.dumps(result), flush=True)
        time.sleep(2)  # let the driver release VRAM before the next fit
    print("\n| Context | Whole layers on CPU | Experts GPU / RAM | VRAM | GPU busy | Prompt | Writing |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['ctx']:,} | {r['cpu_layers']} | {r['experts_gpu_gib']} / {r['experts_ram_gib']} GiB "
              f"| {r['vram_used_mib']:,} MiB | {r['gpu_util_pct']}% | {r['prompt_tps']} tok/s | {r['gen_tps']} tok/s |")


if __name__ == "__main__":
    main()
