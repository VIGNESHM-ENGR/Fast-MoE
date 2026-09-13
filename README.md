# Fast-MoE

**Run large Mixture-of-Experts LLMs on a single consumer GPU (6 GB and up),
with zero manual tuning — and see exactly where every expert lives.**

> Status: early development. Hardware profiling and model downloads work
> today; first serving run is in progress. See [TASKS.md](TASKS.md).

## Why

Sparse MoE models like Qwen3-30B-A3B activate only 8 of 128 experts per
token, so most of their weights can sit in system RAM (or on NVMe) while the
GPU holds the small, always-used parts. The engines that do this well already
exist — Fast-MoE doesn't reinvent them. It packages them into one command,
picks sensible defaults for your machine, and adds a UI that shows what went
where: which experts are on the GPU, which are in RAM, and what each tier costs.

## Built on

| | |
|---|---|
| Inference | [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` — `--fit` places MoE experts across GPU/CPU and sizes the context automatically; weights are memory-mapped, so RAM ↔ NVMe paging is handled by the OS |
| API | llama-server's OpenAI-compatible `/v1/chat/completions` and `/metrics` |
| Models | Quantized GGUF from Hugging Face via `huggingface_hub` |
| UI | Gradio |
| Containers | Official `ghcr.io/ggml-org/llama.cpp` CUDA and CPU images |

```
 models/*.gguf ──▶ llama-server (--fit on) ──▶ OpenAI API :8080 ──▶ your apps
       ▲                 │  GPU: attention, embeddings, KV cache, some experts
       │                 │  RAM: remaining experts (mmap, paged from NVMe)
 model_downloader        │
                         ▼ logs + /metrics
                 Fast-MoE Gradio UI :7860 — config · expert map · resources · chat
```

## Try it now

Profile your hardware:

```bash
pip install -e .
python -m engine.hardware
```

Download the default model (Qwen3-30B-A3B, Q4_K_M, 17.3 GiB) into `models/`:

```bash
python -m engine.models.model_downloader
```

Build llama.cpp (pinned release, CUDA):

```bash
scripts/build_llama_cpp.sh
third_party/llama.cpp/build/bin/llama-server \
  -m models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf --jinja --port 8080
```

## Coming next

`python -m engine.serve` (one-command launcher), the Gradio UI with the expert
map, and `docker compose up`. Track progress in [TASKS.md](TASKS.md).

## Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — inference engine.
- [KTransformers](https://github.com/kvcache-ai/ktransformers) — CPU/GPU hybrid
  MoE research (SOSP'25); candidate future backend.
- [MoE-Infinity](https://arxiv.org/abs/2401.14361) and
  [PreScope](https://arxiv.org/abs/2509.23638) — expert offloading and
  prefetching research informing the NVMe milestone.

## License

Apache-2.0 — see [LICENSE](LICENSE).
