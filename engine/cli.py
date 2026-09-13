"""Main CLI entrypoint: `python -m engine.serve --model <hf_path>` / `fast-moe`.

Wires together the hardware profiler (M1), model loader (M2/M3), memory
manager (M4/M5), prefetcher (M6), and API server (M7) with zero manual
offload/layer flags. See TASKS.md milestone M7.
"""
