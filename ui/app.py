"""Fast-MoE control panel: `python -m ui.app`, then open http://127.0.0.1:7860."""

from __future__ import annotations

import os
import signal
import sys
import time
from collections.abc import Iterator
from html import escape
from pathlib import Path

import gradio as gr

from engine.hardware.allocator import plan_budget
from engine.hardware.profiler import (
    probe_gpus,
    probe_memory,
    probe_storage_device,
    profile_hardware,
)
from engine.llama.runner import Runner, RunPlan, RunSettings, plan_run
from engine.models import catalog
from engine.models.catalog import Sampling
from engine.models.model_downloader import DEFAULT_QUANT, DEFAULT_REPO, local_model, models_dir
from ui.board import BoardView, render_board, render_titleblock
from ui.chat_hud import render_chat_telemetry
from ui.metrics import read_usage
from ui.studio import (
    MODEL_CARD,
    SYSTEM_PRESETS,
    TUNING_PRESETS,
    build_request,
    model_card_html,
    preset_sampling,
)

ASSETS = Path(__file__).resolve().parent / "assets"
GiB = 1024**3
MiB = 1024**2
CONTEXT_CHOICES = [("4K", 4096), ("8K", 8192), ("16K", 16384), ("32K", 32768), ("64K", 65536), ("128K", 131072)]

runner = Runner()
# Single local user: the panel's state lives in the process, not per browser session.
panel = {"state": "unpowered", "plan": None, "previous": None, "message": None, "detail": ""}

_gpus = probe_gpus()
GPU = max(_gpus, key=lambda g: g.total_bytes) if _gpus else None
RAM_TOTAL = probe_memory().total_bytes
_models_disk = probe_storage_device(models_dir()) if models_dir().is_dir() else (None, "unknown")
DISK_LABEL = f"{_models_disk[1].upper()} disk" if _models_disk[1] != "unknown" else None
_HW_PROFILE = None


def get_hardware_profile():
    global _HW_PROFILE
    if _HW_PROFILE is None:
        try:
            # probe_bytes=0: no disk write benchmark just from opening the panel.
            _HW_PROFILE = profile_hardware(probe_bytes=0)
        except OSError:
            _HW_PROFILE = None
    return _HW_PROFILE


def model_choices() -> list[tuple[str, str]]:
    root = models_dir()
    files = sorted(p for p in root.rglob("*.gguf")
                   if ".cache" not in p.parts and not p.name.startswith(("mmproj", "mtp", "dflash")))
    choices = []
    for f in files:
        entry = catalog.find(f)
        choices.append((f"{entry.name} · {entry.quant} · on disk" if entry else f.name, str(f)))
    for entry in catalog.CATALOG:
        if not local_model(entry.repo, entry.quant):
            choices.append((f"{entry.name} · {entry.download_gib:.1f} GiB download on Apply", entry.repo))
    return choices


def default_model() -> str:
    local = local_model(DEFAULT_REPO, DEFAULT_QUANT)
    if local:
        return str(local)
    choices = model_choices()
    return choices[0][1] if choices else DEFAULT_REPO


def settings_from(model: str, ctx: int, margin: int, mmap: bool) -> RunSettings:
    return RunSettings(model=model, ctx=int(ctx), vram_margin_mib=int(margin), no_mmap=not mmap)


def server_pid() -> int | None:
    return runner.server.proc.pid if runner.running else None


def board_html(message: str | None = None) -> str:
    usage = read_usage(GPU.index if GPU else None, None, server_pid())
    return render_board(BoardView(
        state=panel["state"], gpu_name=GPU.name if GPU else None, gpu_total=GPU.total_bytes if GPU else 0,
        ram_total=RAM_TOTAL, plan=panel["plan"], previous=panel["previous"],
        vram_used=usage.vram_used, ram_used=usage.ram_used, model_cached=usage.model_cached,
        message=message or panel["message"]))


def title_html() -> str:
    return render_titleblock(panel["state"], GPU.name if GPU else None, GPU.total_bytes if GPU else 0,
                             RAM_TOTAL, panel["detail"], DISK_LABEL)


def command_text(plan: RunPlan | None) -> str:
    if plan is None:
        return "# Press 'Preview placement' or 'Apply & start' to see the exact llama.cpp command."
    return "llama-server \\\n  " + " \\\n  ".join(_pair_args(plan.server_args))


def _pair_args(args: tuple[str, ...]) -> list[str]:
    out, i = [], 0
    while i < len(args):
        if args[i].startswith("-") and i + 1 < len(args) and not args[i + 1].startswith("-"):
            value = args[i + 1]
            out.append(f"{args[i]} '{value}'" if any(c in value for c in " |()*\\") else f"{args[i]} {value}")
            i += 2
        else:
            out.append(args[i])
            i += 1
    return out


def _set(state: str, message: str | None = None, detail: str = "") -> None:
    panel["state"], panel["message"], panel["detail"] = state, message, detail


def preview(model: str, ctx: int, margin: int, mmap: bool) -> Iterator[tuple[str, str, str]]:
    if runner.running:
        msg = ("A model is actively running and occupying VRAM. "
               "Stop it first to preview, or click 'Apply & start' to seamlessly relaunch with these settings.")
        yield title_html(), board_html(msg), command_text(panel["plan"])
        return
    settings = settings_from(model, ctx, margin, mmap)
    if Path(model).suffix != ".gguf" and not local_model(model, settings.quant):
        yield title_html(), board_html(f"{model} is not downloaded yet. 'Apply & start' will download it automatically."), command_text(None)
        return
    _set("fitting", detail="calculating MoE layer distribution")
    yield title_html(), board_html(), command_text(panel["plan"])
    try:
        plan = plan_run(settings)
    except Exception as exc:  # noqa: BLE001 - UI boundary: show the failure instead of a stack trace
        _set("fault", f"Placement calculation failed: {exc}")
        yield title_html(), board_html(), command_text(None)
        return
    if panel["plan"] is not None:
        panel["previous"] = panel["plan"]
    panel["plan"] = plan
    _set("unpowered", None, "preview")
    yield title_html(), board_html(), command_text(plan)


def apply(model: str, ctx: int, margin: int, mmap: bool) -> Iterator[tuple[str, str, str]]:
    settings = settings_from(model, ctx, margin, mmap)
    _set("fitting", detail="reallocating VRAM & fitting experts")
    yield title_html(), board_html(), command_text(panel["plan"])
    try:
        runner.stop()
        plan = plan_run(settings)
        if panel["plan"] is not None:
            panel["previous"] = panel["plan"]
        panel["plan"] = plan
        _set("loading", detail="loading weights via mmap")
        yield title_html(), board_html(), command_text(plan)
        took = runner.launch(plan)
    except Exception as exc:  # noqa: BLE001 - UI boundary: show the failure instead of a stack trace
        _set("fault", f"Engine launch failed: {exc}")
        yield title_html(), board_html(), command_text(panel["plan"])
        return
    _set("live", detail=f"ready in {took:.0f}s · {runner.server.url}/v1")
    yield title_html(), board_html(), command_text(plan)


def stop() -> tuple[str, str]:
    runner.stop()
    _set("unpowered", detail="stopped")
    return title_html(), board_html()


def meters_html() -> str:
    usage = read_usage(GPU.index if GPU else None, runner.server.url if runner.running else None, server_pid())

    def meter_card(title: str, val: str, sub: str, frac: float | None, fill_cls: str) -> str:
        bar = (f'<div class="fm-meter-bar"><div class="fm-meter-fill {fill_cls}" '
               f'style="width: {min(max(frac, 0), 1) * 100:.0f}%;"></div></div>') if frac is not None else ""
        return f"""
        <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 0.75rem 1rem;">
          <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.35rem;">
            <span>{escape(title)}</span>
            <span style="font-family: var(--font-mono); color: var(--text-primary); font-weight: 700;">{escape(val)}</span>
          </div>
          {bar}
          <div style="font-size: 0.65rem; color: var(--text-muted); margin-top: 0.3rem;">{escape(sub)}</div>
        </div>
        """

    vram = (meter_card("GPU VRAM", f"{usage.vram_used / GiB:.1f} / {usage.vram_total / GiB:.1f} GB",
                       f"{usage.vram_used / usage.vram_total * 100:.0f}% allocated",
                       usage.vram_used / usage.vram_total, "fill-gpu")
            if usage.vram_total else meter_card("GPU VRAM", "No GPU", "CPU mode", None, ""))
    ram = meter_card("System RAM", f"{usage.ram_used / GiB:.1f} / {usage.ram_total / GiB:.1f} GB",
                     f"{usage.ram_used / usage.ram_total * 100:.0f}% in use"
                     + (f" + {usage.model_cached / GiB:.1f} GB model file cached" if usage.model_cached else ""),
                     usage.ram_used / usage.ram_total, "fill-ram")
    gen_val = f"{usage.gen_tps:.1f} tok/s" if usage.gen_tps else "Idle"
    prompt_val = f"{usage.prompt_tps:.0f} tok/s" if usage.prompt_tps else "Idle"

    speed = f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
      <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 0.75rem; text-align: center;">
        <div style="font-size: 0.65rem; text-transform: uppercase; color: var(--text-secondary);">Generation</div>
        <div style="font-size: 1.15rem; font-weight: 800; color: var(--emerald-bright); font-family: var(--font-mono); margin-top: 0.2rem;">{gen_val}</div>
      </div>
      <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 0.75rem; text-align: center;">
        <div style="font-size: 0.65rem; text-transform: uppercase; color: var(--text-secondary);">Prompt Speed</div>
        <div style="font-size: 1.15rem; font-weight: 800; color: var(--cyan-bright); font-family: var(--font-mono); margin-top: 0.2rem;">{prompt_val}</div>
      </div>
    </div>
    """

    return f'<div style="display: flex; flex-direction: column; gap: 0.65rem; margin-top: 0.75rem;">{vram}{ram}{speed}</div>'


def hardware_html() -> str:
    hp = get_hardware_profile()
    if not hp:
        return "<p style='padding: 1rem; color: var(--text-secondary);'>Hardware telemetry loading or unavailable.</p>"

    budgets = plan_budget(hp)

    # GPU Tile
    gpu = hp.gpus[0] if hp.gpus else None
    gpu_html = f"""
    <div class="fm-hw-card">
      <h3><span>⚡</span> GPU Acceleration</h3>
      <div class="fm-hw-rows">
        <div class="fm-hw-row"><span class="k">Device:</span><span class="v">{escape(gpu.name if gpu else 'No GPU Detected')}</span></div>
        <div class="fm-hw-row"><span class="k">VRAM Capacity:</span><span class="v">{gpu.total_bytes / GiB:.2f} GiB</span></div>
        <div class="fm-hw-row"><span class="k">Free VRAM:</span><span class="v">{gpu.free_bytes / GiB:.2f} GiB</span></div>
        <div class="fm-hw-row"><span class="k">Compute Capability:</span><span class="v">{gpu.compute_capability[0]}.{gpu.compute_capability[1]}</span></div>
      </div>
    </div>
    """ if gpu else """
    <div class="fm-hw-card">
      <h3><span>⚡</span> GPU Acceleration</h3>
      <p style="color: var(--text-muted); font-size: 0.8rem;">No dedicated CUDA GPU detected. System running in CPU-only fallback.</p>
    </div>
    """

    # CPU Tile
    simd_flags = [f for f in ["avx512f", "avx512_vnni", "avx2", "fma", "sse4_2"] if f in hp.cpu.flags]
    simd_badges = "".join(f'<span class="fm-flag-badge">{escape(f)}</span>' for f in simd_flags)
    cpu_html = f"""
    <div class="fm-hw-card">
      <h3><span>🧠</span> Host CPU Architecture</h3>
      <div class="fm-hw-rows">
        <div class="fm-hw-row"><span class="k">Cores / Threads:</span><span class="v">{hp.cpu.physical_cores} Cores / {hp.cpu.logical_threads} Threads</span></div>
        <div class="fm-hw-row"><span class="k">NUMA Nodes:</span><span class="v">{hp.cpu.numa_nodes} Node(s)</span></div>
        <div class="fm-hw-row"><span class="k">Vector SIMD:</span><div class="fm-badge-row">{simd_badges}</div></div>
        <div class="fm-hw-row"><span class="k">Host Total RAM:</span><span class="v">{hp.memory.total_bytes / GiB:.2f} GiB</span></div>
        <div class="fm-hw-row"><span class="k">cgroup RAM Limit:</span><span class="v">{'Unlimited' if not hp.memory.cgroup_limit_bytes else f'{hp.memory.cgroup_limit_bytes / GiB:.1f} GiB'}</span></div>
      </div>
    </div>
    """

    # Storage Benchmark Tile
    storage = hp.storage
    not_measured = "not measured (run python -m engine.hardware)"
    read_speed = f"{storage.read_bytes_per_s / MiB:.0f} MiB/s" if storage.read_bytes_per_s else not_measured
    write_speed = f"{storage.write_bytes_per_s / MiB:.0f} MiB/s" if storage.write_bytes_per_s else not_measured
    storage_html = f"""
    <div class="fm-hw-card">
      <h3><span>💾</span> Cold Tier Storage</h3>
      <div class="fm-hw-rows">
        <div class="fm-hw-row"><span class="k">Device Path:</span><span class="v">{escape(str(storage.path))}</span></div>
        <div class="fm-hw-row"><span class="k">Drive Type:</span><span class="v">{escape(storage.kind.upper())} ({escape(storage.device or 'unknown device')})</span></div>
        <div class="fm-hw-row"><span class="k">Free Capacity:</span><span class="v">{storage.free_bytes / GiB:.1f} GiB</span></div>
        <div class="fm-hw-row"><span class="k">Sequential Read:</span><span class="v" style="color: var(--cyan-bright);">{read_speed}</span></div>
        <div class="fm-hw-row"><span class="k">Sequential Write:</span><span class="v">{write_speed}</span></div>
      </div>
    </div>
    """

    # Budget Allocation Tile
    budget_html = f"""
    <div class="fm-hw-card">
      <h3><span>⚖️</span> Autonomous Budget Allocation</h3>
      <div class="fm-hw-rows">
        <div class="fm-hw-row"><span class="k">VRAM Hot Budget:</span><span class="v" style="color: var(--cyan-bright);">{budgets.vram_bytes / GiB:.2f} GiB ({budgets.vram_reserve_bytes / GiB:.1f} GiB reserve)</span></div>
        <div class="fm-hw-row"><span class="k">RAM Warm Budget:</span><span class="v" style="color: var(--emerald-bright);">{budgets.ram_bytes / GiB:.2f} GiB</span></div>
        <div class="fm-hw-row"><span class="k">Pinned RAM Pool:</span><span class="v">{budgets.pinned_ram_bytes / GiB:.2f} GiB</span></div>
        <div class="fm-hw-row"><span class="k">Cold Swap Budget:</span><span class="v" style="color: var(--amber-bright);">{budgets.disk_bytes / GiB:.1f} GiB</span></div>
      </div>
    </div>
    """

    return f"""
    <div class="fm-hw-grid">
      {gpu_html}
      {cpu_html}
      {storage_html}
      {budget_html}
    </div>
    """


def log_html() -> str:
    lines = list(runner.server.tail)[-40:] if runner.server else []
    if not lines:
        return '<pre class="fm-terminal"><span class="info">● Fast-MoE Log Streamer Initialized.</span>\nWaiting for model server start...</pre>'
    out = []
    for line in lines:
        cls = "err" if " E " in line[:25] or "error" in line.lower() else "warn" if " W " in line[:25] or "warning" in line.lower() else "info" if " I " in line[:25] else ""
        safe = escape(line)
        out.append(f'<span class="{cls}">{safe}</span>' if cls else safe)
    return '<pre class="fm-terminal">' + "\n".join(out) + "</pre>"


def tick():
    board = gr.skip()
    if panel["state"] == "live" and not runner.running:
        _set("fault", "llama-server stopped unexpectedly. See Server Logs tab.")
        board = board_html()
    return title_html(), board, meters_html(), log_html()


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def chat(message: str, history: list, system_prompt: str, temperature: float, top_p: float, top_k: float,
         min_p: float, presence_penalty: float, repeat_penalty: float, max_tokens: float,
         thinking: bool) -> Iterator[tuple[str, list, str]]:
    if not message.strip():
        yield "", history, render_chat_telemetry()
        return
    history = [*history, {"role": "user", "content": message}]
    if not runner.running:
        history.append({"role": "assistant", "content": "⚠️ **No model is running yet.** Go to the "
                        "**⚡ Engine & Memory Topology** tab and click **'Apply & Start'** first."})
        yield "", history, render_chat_telemetry(state="error")
        return
    from openai import OpenAI

    client = OpenAI(base_url=f"{runner.server.url}/v1", api_key="local")
    sent = [{"role": m["role"], "content": _text(m["content"])} for m in history
            if not (m.get("metadata") or {}).get("title")]
    thinking = {"role": "assistant", "content": "", "metadata": {"title": "Thinking Process"}}
    answer = {"role": "assistant", "content": ""}
    plan = panel["plan"]
    ctx_max = (plan.fit.context or plan.layout.context_length) if plan else 4096

    start_time = time.time()
    first_token_time = reasoning_end_time = None
    ttft_ms = reasoning_s = gen_s = 0.0
    chunks = 0
    state = "starting"
    yield "", history, render_chat_telemetry(state=state, ctx_max=ctx_max)

    try:
        sampling = Sampling(temperature=temperature, top_p=top_p, top_k=int(top_k), min_p=min_p,
                            presence_penalty=presence_penalty, repeat_penalty=repeat_penalty)
        request = build_request(system_prompt or "", sent, sampling, max_tokens, thinking)
        stream = client.chat.completions.create(model="local", stream=True, stream_options={"include_usage": True},
                                                **request)
        usage = timings = None
        for chunk in stream:
            usage = chunk.usage or usage
            timings = (chunk.model_extra or {}).get("timings") or timings  # llama-server extension
            if not chunk.choices:
                continue
            now = time.time()
            if first_token_time is None:
                first_token_time = now
                ttft_ms = (now - start_time) * 1000.0
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                state = "reasoning"
                reasoning_s = now - first_token_time
                if thinking not in history:
                    history.append(thinking)
                thinking["content"] += reasoning
            if delta.content:
                reasoning_end_time = reasoning_end_time or now
                state = "generating"
                gen_s = now - reasoning_end_time
                if answer not in history:
                    history.append(answer)
                answer["content"] += delta.content
            chunks += 1
            elapsed = now - first_token_time
            yield "", history, render_chat_telemetry(
                state=state, tps=chunks / elapsed if elapsed > 0 else 0.0, ttft_ms=ttft_ms,
                reasoning_s=reasoning_s, gen_s=gen_s, ctx_max=ctx_max, total_tokens=chunks,
                speed_note="Live estimate (stream chunks/s)")

        tps = timings.get("predicted_per_second", 0.0) if timings else 0.0
        yield "", history, render_chat_telemetry(
            state="completed", tps=tps, ttft_ms=ttft_ms, reasoning_s=reasoning_s, gen_s=gen_s,
            ctx_used=usage.total_tokens if usage else None, ctx_max=ctx_max,
            total_tokens=usage.completion_tokens if usage else chunks,
            speed_note="Measured by llama.cpp" if timings else "Not reported by the server")
    except Exception as exc:  # noqa: BLE001 - UI boundary: report the failure inside the conversation
        history.append({"role": "assistant", "content": f"❌ Request error: {exc}"})
        yield "", history, render_chat_telemetry(state="error")


def build() -> gr.Blocks:
    with gr.Blocks(title="Fast-MoE · Sparse Mixture-of-Experts Orchestrator") as demo:
        title = gr.HTML(title_html())

        with gr.Tabs():
            # TAB 1: Visualizer & Engine
            with gr.Tab("⚡ Engine & Memory Topology", id="tab-engine"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=8, min_width=580):
                        board = gr.HTML(board_html())
                    with gr.Column(scale=4, min_width=320, elem_id="config"):
                        gr.HTML("""
                        <div style="margin-bottom: 0.75rem;">
                          <h2 style="font-size: 1.05rem; font-weight: 800; margin: 0; color: var(--text-primary);">Engine Controls</h2>
                          <p style="font-size: 0.75rem; color: var(--text-secondary); margin: 0.2rem 0 0 0;">Zero-config placement with custom tuning</p>
                        </div>
                        """)
                        model = gr.Dropdown(choices=model_choices(), value=default_model(), allow_custom_value=True,
                                            label="Model", info="Tested models, local GGUF files, or any Hugging Face GGUF repo.")
                        model_card = gr.HTML(model_card_html(default_model()))
                        ctx = gr.Radio(choices=CONTEXT_CHOICES, value=catalog.default_ctx(default_model()),
                                       label="Context Length",
                                       info="More context needs more KV cache on the GPU, so more experts move to RAM and, past a point, whole layers run on the CPU.")
                        margin = gr.Slider(256, 3072, value=1024, step=256, label="GPU Safety Margin (MiB)",
                                           info="Reserved VRAM for desktop display and OS buffers.")
                        mmap = gr.Checkbox(value=True, label="Zero-Copy Memory-Mapped (mmap)",
                                           info="Starts faster (about 10 s vs 29 s measured on a 6 GB laptop); the OS pages weights in from disk as needed.")
                        with gr.Row():
                            preview_btn = gr.Button("Preview Placement", variant="secondary")
                            apply_btn = gr.Button("Apply & Start", variant="primary")
                            stop_btn = gr.Button("Stop", variant="stop")
                        meters = gr.HTML(meters_html())
                        command = gr.Code(command_text(None), language="shell", label="Exact llama-server Command",
                                          interactive=False)

            # TAB 2: Chat Studio
            with gr.Tab("💬 Chat Studio", id="tab-chat"), gr.Row(equal_height=False):
                with gr.Column(scale=7, min_width=520, elem_classes=["fm-chat-container"]):
                    gr.HTML("""
                    <div class="fm-studio-title">
                      <h2>Local AI Chat Playground</h2>
                      <p>Streaming from llama-server's OpenAI-compatible API (:8080/v1). Speed and context are measured by the server.</p>
                    </div>
                    """)
                    chat_hud = gr.HTML(render_chat_telemetry())
                    with gr.Row():
                        p1 = gr.Button("💡 How does Fast-MoE fit a 35B model on 6 GB?", elem_classes=["fm-chip-btn"], size="sm")
                        p2 = gr.Button("⚡ Explain Mixture-of-Experts", elem_classes=["fm-chip-btn"], size="sm")
                        p3 = gr.Button("🐍 Python script to benchmark memory bandwidth", elem_classes=["fm-chip-btn"], size="sm")
                    chatbot = gr.Chatbot(height=520, label="Conversation", show_label=False,
                                         placeholder="Start a model on the Engine tab, then say hello.")
                    with gr.Row():
                        msg = gr.Textbox(placeholder="Type a message to the running model...", show_label=False, scale=1)
                        send_btn = gr.Button("Send", variant="primary", scale=0, min_width=110)
                        clear_btn = gr.Button("Clear", variant="secondary", scale=0, min_width=90)

                with gr.Column(scale=4, min_width=320, elem_classes=["fm-studio-sidebar"]):
                    with gr.Group(elem_classes=["fm-studio-card", "fm-studio-system"]):
                        gr.HTML("""
                        <div class="fm-studio-head">
                          <span class="fm-studio-icon">🧭</span>
                          <div><h3>System Prompt</h3><p>Sets the model's role and tone for the whole conversation.</p></div>
                        </div>
                        """)
                        system_preset = gr.Radio(choices=list(SYSTEM_PRESETS), value="Helpful", show_label=False,
                                                 elem_classes=["fm-segmented"])
                        system_prompt = gr.Textbox(value=SYSTEM_PRESETS["Helpful"], lines=5, max_lines=12,
                                                   show_label=False, placeholder="Describe how the assistant should behave...",
                                                   elem_classes=["fm-system-box"])

                    with gr.Group(elem_classes=["fm-studio-card", "fm-studio-tuning"]):
                        gr.HTML("""
                        <div class="fm-studio-head">
                          <span class="fm-studio-icon">🎛️</span>
                          <div><h3>Generation Tuning</h3><p>How the model picks each next word. Applies from your next message.</p></div>
                        </div>
                        """)
                        start_sampling, start_note = preset_sampling(MODEL_CARD, default_model())
                        tuning_preset = gr.Radio(choices=list(TUNING_PRESETS), value=MODEL_CARD, show_label=False,
                                                 elem_classes=["fm-segmented"])
                        tuning_note = gr.HTML(f'<p class="fm-tuning-note">{start_note}</p>')
                        thinking = gr.Checkbox(value=True, label="Reasoning",
                                               info="Let the model think before answering. Off is faster.",
                                               elem_classes=["fm-think-toggle"])
                        temperature = gr.Slider(0.0, 2.0, value=start_sampling.temperature, step=0.05, label="Temperature",
                                                info="Lower is predictable, higher is inventive.")
                        top_p = gr.Slider(0.05, 1.0, value=start_sampling.top_p, step=0.01, label="Top-p",
                                          info="Only consider the most likely words adding up to this share.")
                        top_k = gr.Slider(0, 200, value=start_sampling.top_k, step=1, label="Top-k",
                                          info="Only consider this many candidate words (0 = all).")
                        with gr.Accordion("Advanced", open=False, elem_classes=["fm-advanced"]):
                            min_p = gr.Slider(0.0, 0.5, value=start_sampling.min_p, step=0.01, label="Min-p",
                                              info="Drop words far less likely than the top choice.")
                            presence_penalty = gr.Slider(0.0, 2.0, value=start_sampling.presence_penalty, step=0.05,
                                                         label="Presence penalty", info="Push toward new topics and words.")
                            repeat_penalty = gr.Slider(1.0, 2.0, value=start_sampling.repeat_penalty, step=0.01,
                                                       label="Repeat penalty", info="Discourage repeating recent tokens.")
                            max_tokens = gr.Slider(256, 32768, value=8192, step=256, label="Max output tokens",
                                                   info="Reasoning counts toward this limit.")

            # TAB 3: Hardware & Benchmarks
            with gr.Tab("📊 Hardware & Benchmarks", id="tab-hardware"):
                gr.HTML("""
                <div style="margin-bottom: 1.25rem;">
                  <h2 style="font-size: 1.15rem; font-weight: 800; margin: 0; color: var(--text-primary);">Hardware Profiling & Tier Benchmarks</h2>
                  <p style="font-size: 0.75rem; color: var(--text-secondary); margin: 0.2rem 0 0 0;">Probed directly from host NVML, sysfs, and sequential storage throughput benchmarks</p>
                </div>
                """)
                gr.HTML(hardware_html())

            # TAB 4: Engine Logs
            with gr.Tab("📜 Engine Logs & Diagnostics", id="tab-logs"):
                gr.HTML("""
                <div style="margin-bottom: 0.75rem;">
                  <h2 style="font-size: 1.15rem; font-weight: 800; margin: 0; color: var(--text-primary);">Server Log Console</h2>
                  <p style="font-size: 0.75rem; color: var(--text-secondary); margin: 0.2rem 0 0 0;">Live stdout/stderr stream from llama-server process</p>
                </div>
                """)
                log = gr.HTML(log_html())

        # Event Handlers
        inputs = [model, ctx, margin, mmap]
        preview_btn.click(preview, inputs, [title, board, command], concurrency_limit=1)
        apply_btn.click(apply, inputs, [title, board, command], concurrency_limit=1)
        stop_btn.click(stop, None, [title, board])

        chat_inputs = [msg, chatbot, system_prompt, temperature, top_p, top_k, min_p, presence_penalty,
                       repeat_penalty, max_tokens, thinking]
        msg.submit(chat, chat_inputs, [msg, chatbot, chat_hud])
        send_btn.click(chat, chat_inputs, [msg, chatbot, chat_hud])
        model.change(model_card_html, model, model_card)
        model.change(catalog.default_ctx, model, ctx)
        system_preset.change(lambda name: SYSTEM_PRESETS[name], system_preset, system_prompt)

        def apply_tuning_preset(preset: str, selected_model: str):
            running = str(runner.plan.model_path) if runner.running and runner.plan else selected_model
            s, note = preset_sampling(preset, running)
            return (s.temperature, s.top_p, s.top_k, s.min_p, s.presence_penalty, s.repeat_penalty,
                    f'<p class="fm-tuning-note">{note}</p>')

        tuning_outputs = [temperature, top_p, top_k, min_p, presence_penalty, repeat_penalty, tuning_note]
        tuning_preset.change(apply_tuning_preset, [tuning_preset, model], tuning_outputs)
        model.change(apply_tuning_preset, [tuning_preset, model], tuning_outputs)
        clear_btn.click(lambda: ([], render_chat_telemetry()), None, [chatbot, chat_hud])

        # Quick prompt buttons
        p1.click(lambda: "How does Fast-MoE run a 35-Billion parameter model on a 6GB GPU without crashing?", None, msg)
        p2.click(lambda: "Explain what Sparse Mixture-of-Experts (MoE) is and why only 8 of 256 experts are active per token.", None, msg)
        p3.click(lambda: "Write a high-performance Python script to measure sequential memory bandwidth between host RAM and GPU VRAM.", None, msg)

        gr.Timer(2.0).tick(tick, None, [title, board, meters, log], show_progress="hidden")

        # Refresh state on tab load
        demo.load(lambda: (title_html(), board_html(), meters_html(), log_html(), command_text(panel["plan"])),
                  None, [title, board, meters, log, command])

    return demo


THEME = gr.themes.Base(
    font=[gr.themes.Font(n) for n in ("Atkinson Hyperlegible Next", "Segoe UI", "sans-serif")],
    font_mono=[gr.themes.Font(n) for n in ("B612 Mono", "monospace")],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#070a12",
    body_text_color="#f8fafc",
    body_text_color_subdued="#94a3b8",
    background_fill_primary="#070a12",
    background_fill_secondary="#0e1424",
    block_background_fill="#0e1424",
    block_border_color="rgba(255, 255, 255, 0.08)",
    block_border_width="1px",
    block_label_background_fill="#0e1424",
    block_label_text_color="#f8fafc",
    block_title_text_color="#f8fafc",
    block_info_text_color="#94a3b8",
    border_color_primary="rgba(255, 255, 255, 0.08)",
    input_background_fill="#070a12",
    input_border_color="rgba(255, 255, 255, 0.12)",
    input_border_color_focus="#06b6d4",
    color_accent="#06b6d4",
    color_accent_soft="#083344",
    slider_color="#06b6d4",
    checkbox_background_color="#070a12",
    checkbox_background_color_selected="#06b6d4",
    checkbox_border_color="rgba(255, 255, 255, 0.15)",
    button_primary_background_fill="#06b6d4",
    button_primary_background_fill_hover="#0891b2",
    button_primary_text_color="#000000",
    button_secondary_background_fill="rgba(255, 255, 255, 0.04)",
    button_secondary_background_fill_hover="rgba(255, 255, 255, 0.08)",
    button_secondary_border_color="rgba(255, 255, 255, 0.12)",
    button_secondary_text_color="#f8fafc",
    code_background_fill="#050811",
)


def main() -> None:
    css = (ASSETS / "board.css").read_text().replace("/gradio_api/file=ui/assets/", f"/gradio_api/file={ASSETS}/")
    # `docker compose down` sends SIGTERM; exit through `finally` so llama-server is stopped cleanly.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        build().queue().launch(server_name=os.environ.get("FAST_MOE_UI_HOST", "127.0.0.1"),
                               server_port=int(os.environ.get("FAST_MOE_UI_PORT", "7860")), theme=THEME, css=css,
                               allowed_paths=[str(ASSETS)])
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
