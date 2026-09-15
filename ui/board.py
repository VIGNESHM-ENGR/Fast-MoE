"""Fast-MoE memory topology view: where every layer's experts live, from a real RunPlan.

Every number shown comes from the plan (llama-fit-params mapped onto GGUF tensors) or
from live readings passed in; anything unknown renders as "—" rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from engine.llama.layout import LayerPlacement, describe_ranges
from engine.llama.runner import RunPlan, context_warning, cpu_attention_layers

GiB = 1024**3
MiB = 1024**2
UNKNOWN = "—"

STATES = {
    "unpowered": "Unpowered",
    "fitting": "Fitting",
    "loading": "Loading",
    "live": "Live",
    "fault": "Fault",
}


@dataclass(frozen=True)
class BoardView:
    state: str
    gpu_name: str | None
    gpu_total: int
    ram_total: int
    plan: RunPlan | None = None
    previous: RunPlan | None = None  # last applied plan, to show what moved
    vram_used: int | None = None
    ram_used: int | None = None
    model_cached: int | None = None  # model file pages llama-server holds in RAM (page cache)
    message: str | None = None


def _gb(n: float) -> str:
    return f"{n / GiB:.1f} GB"


def _size(n: float) -> str:
    return f"{n / GiB:.2f} GiB" if n >= GiB else f"{n / MiB:.0f} MiB"


def summary_sentence(plan: RunPlan) -> str:
    """One plain sentence a first-time user can read immediately."""
    p = plan.placement
    cpu = cpu_attention_layers(plan)
    gpu = [x.layer for x in p if x.gpu_bytes and not x.cpu_bytes]
    split = [x.layer for x in p if x.gpu_bytes and x.cpu_bytes]
    ram = [x.layer for x in p if x.cpu_bytes and not x.gpu_bytes and x.layer not in cpu]
    total = len(p)
    ctx = plan.fit.context or plan.layout.context_length
    parts = []
    if not ram and not split and not cpu:
        parts.append(f"All {total} layers fit on the GPU (fastest execution).")
    elif not gpu and not split and not cpu:
        if plan.fit.gpu_layers:
            parts.append(f"The experts of all {total} layers run from System RAM; GPU holds attention and KV cache.")
        else:
            parts.append(f"All {total} layers, including attention and the KV cache, run from System RAM (no GPU in use).")
    else:
        if gpu:
            parts.append(f"Layers {describe_ranges(gpu)} run in GPU VRAM.")
        if split:
            parts.append(f"Layer {describe_ranges(split)} is split between GPU and RAM.")
        if ram:
            parts.append(f"Layers {describe_ranges(ram)} run in System RAM.")
        if cpu:
            parts.append(f"Layers {describe_ranges(cpu)} run entirely on the CPU.")
    parts.append(f"Active context: {ctx:,} tokens.")
    return " ".join(parts)


def moved_off_gpu(plan: RunPlan, previous: RunPlan | None) -> set[int]:
    if previous is None or previous.model_path != plan.model_path:
        return set()
    before = {x.layer: x for x in previous.placement}
    return {x.layer for x in plan.placement if x.layer in before and x.gpu_bytes < before[x.layer].gpu_bytes}


def render_titleblock(state: str, gpu_name: str | None, gpu_total: int, ram_total: int, detail: str = "",
                      disk_label: str | None = None) -> str:
    """Top navbar with hardware pills and the server state."""
    gpu_label = f"🔥 {gpu_name} ({_gb(gpu_total)} VRAM)" if gpu_name else "🔥 No GPU (CPU only)"
    ram_label = f"🧠 {_gb(ram_total)} RAM"
    disk_pill = f'<div class="fm-pill disk"><strong>❄️ {escape(disk_label)}</strong></div>' if disk_label else ""
    status_detail = f" · {escape(detail)}" if detail else ""

    return f"""
    <header class="fm-navbar">
      <div class="fm-brand">
        <div class="fm-logo-icon">⚡</div>
        <div class="fm-title-group">
          <h1>FAST-MOE</h1>
          <div class="fm-tagline">Sparse Mixture-of-Experts Hybrid Orchestrator</div>
        </div>
      </div>
      <div class="fm-telemetry-pills">
        <div class="fm-pill gpu"><strong>{escape(gpu_label)}</strong></div>
        <div class="fm-pill ram"><strong>{escape(ram_label)}</strong></div>
        {disk_pill}
        <div class="fm-status-pill state-{escape(state)}">
          <span class="fm-pulse-dot"></span>
          <span>{escape(STATES.get(state, state))}{status_detail}</span>
        </div>
      </div>
    </header>
    """


def _meter(used: int | None, total: int, fill_cls: str, label: str, cached: int | None = None) -> str:
    frac = min(used / total, 1.0) if used is not None and total else 0.0
    value = f"{_gb(used)} / {_gb(total)}" if used is not None and total else UNKNOWN
    cache_fill = ""
    if cached and used is not None and total:
        value = f"{_gb(used)} + {_gb(cached)} model / {_gb(total)}"
        cache_frac = min(cached / total, 1.0 - frac)
        cache_fill = f'<div class="fm-meter-fill fill-cache" style="width: {cache_frac * 100:.1f}%;"></div>'
    return f"""
        <div class="fm-tier-meter">
          <div class="fm-meter-bar">
            <div class="fm-meter-fill {fill_cls}" style="width: {frac * 100:.1f}%;"></div>{cache_fill}
          </div>
          <div class="fm-meter-labels">
            <span>{escape(label)}</span>
            <span>{value}</span>
          </div>
        </div>"""


def _row(label: str, value: str, title: str = "") -> str:
    attr = f' title="{escape(title)}"' if title else ""
    return (f'<div class="fm-content-row"><span class="label">{escape(label)}</span>'
            f'<span class="val"{attr}>{escape(value)}</span></div>')


def _explainer(plan: RunPlan | None, gpu_n: int, split_n: int, ram_n: int, gpu_bytes: int, ram_bytes: int) -> str:
    if plan:
        k, n = plan.layout.experts_per_token, plan.layout.expert_count
        routing_badge, routing_value = f"{k} / {n} active", f"{k / n * 100:.1f}% of experts per token"
        routing_desc = (f"Each token is routed to {k} of the {n} experts in every MoE layer, "
                        "so only a small slice of the expert weights does work for any one token.")
        if ram_n or split_n:
            place_badge = "VRAM + RAM" if gpu_n or split_n else "RAM experts"
        else:
            place_badge = "VRAM only"
        place_value = f"{gpu_n} GPU · {split_n} split · {ram_n} RAM"
        place_desc = (f"{_size(gpu_bytes)} of expert weights sit in VRAM and {_size(ram_bytes)} in system RAM, "
                      "as decided by llama.cpp's fit for this machine.")
        mmap = not plan.settings.no_mmap
        file_badge, file_value = ("NVMe mmap" if mmap else "Loaded to RAM"), _size(plan.layout.total_bytes)
        file_desc = ("The model file is memory-mapped: the OS pages weights from disk into RAM as they are used."
                     if mmap else "The whole model file is read into RAM when the server starts.")
    else:
        routing_badge = place_badge = file_badge = "No plan yet"
        routing_value = place_value = file_value = UNKNOWN
        routing_desc = "How many experts each token uses appears once a model is placed."
        place_desc = "Press Preview Placement to see which layers go to VRAM and which to RAM."
        file_desc = "The model file's size and loading mode appear here."

    def card(tier: str, title: str, badge: str, badge_cls: str, value: str, desc: str) -> str:
        return f"""
      <div class="fm-card fm-card-hero {tier}">
        <div class="fm-card-title">
          <span>{escape(title)}</span>
          <span class="badge {badge_cls}">{escape(badge)}</span>
        </div>
        <div class="fm-card-value">{escape(value)}</div>
        <div class="fm-card-desc">{escape(desc)}</div>
      </div>"""

    return ('<div class="fm-explainer-banner">'
            + card("gpu-tier", "Sparse Routing", routing_badge, "badge-amber", routing_value, routing_desc)
            + card("ram-tier", "Hybrid Placement", place_badge, "badge-emerald", place_value, place_desc)
            + card("ssd-tier", "Model File", file_badge, "badge-cyan", file_value, file_desc)
            + "</div>")


def _tiers(view: BoardView, gpu_parts: list[LayerPlacement], ram_parts: list[LayerPlacement],
           gpu_bytes: int, ram_bytes: int) -> str:
    plan = view.plan
    layout = plan.layout if plan else None
    shared = _size(layout.total_bytes - layout.embedding_bytes - gpu_bytes - ram_bytes) if layout else UNKNOWN
    ctx = f"{plan.fit.context or layout.context_length:,} tokens" if plan else UNKNOWN
    if plan and not plan.fit.gpu_layers:
        # CPU-only: nothing is offloaded, so these live in RAM, not in this GPU tier.
        shared, ctx = f"{shared} (in RAM)", f"{ctx} (in RAM)"
    gpu_layers = f"{describe_ranges([p.layer for p in gpu_parts])} ({_size(gpu_bytes)})" if gpu_parts else "none"
    ram_layers = f"{describe_ranges([p.layer for p in ram_parts])} ({_size(ram_bytes)})" if ram_parts else "none"
    model_name = plan.model_path.name if plan else UNKNOWN
    access = UNKNOWN if not plan else ("Memory-mapped (mmap)" if not plan.settings.no_mmap else "Loaded into RAM")

    return f"""
    <div class="fm-tier-container">
      <div class="fm-tier-card fm-tier-gpu">
        <div class="fm-tier-header">
          <div class="fm-tier-icon">🔥</div>
          <div class="fm-tier-info">
            <h3>Tier 1: GPU VRAM (Hot)</h3>
            <p>{escape(view.gpu_name or "No GPU detected")}</p>
          </div>
        </div>
        {_meter(view.vram_used, view.gpu_total, "fill-gpu", "In use")}
        <div class="fm-tier-contents">
          {_row("Attention & shared:", shared)}
          {_row("KV cache:", ctx)}
          {_row("Expert layers:", gpu_layers if plan else UNKNOWN)}
        </div>
      </div>

      <div class="fm-tier-card fm-tier-ram">
        <div class="fm-tier-header">
          <div class="fm-tier-icon">🧠</div>
          <div class="fm-tier-info">
            <h3>Tier 2: System RAM (Warm)</h3>
            <p>System memory</p>
          </div>
        </div>
        {_meter(view.ram_used, view.ram_total, "fill-ram", "In use", view.model_cached)}
        <div class="fm-tier-contents">
          {_row("Expert layers:", ram_layers if plan else UNKNOWN)}
          {_row("Model file in RAM:", _size(view.model_cached) if view.model_cached else UNKNOWN,
                "Memory-mapped model pages. Linux counts them as cache, not used, and can free them.")}
          {_row("Token embeddings:", _size(layout.embedding_bytes) if layout else UNKNOWN,
                "llama.cpp always keeps input embeddings on the CPU")}
        </div>
      </div>

      <div class="fm-tier-card fm-tier-ssd">
        <div class="fm-tier-header">
          <div class="fm-tier-icon">❄️</div>
          <div class="fm-tier-info">
            <h3>Tier 3: Disk (Cold)</h3>
            <p>Model file</p>
          </div>
        </div>
        <div class="fm-tier-meter">
          <div class="fm-meter-bar">
            <div class="fm-meter-fill fill-ssd" style="width: {100 if plan else 0}%;"></div>
          </div>
          <div class="fm-meter-labels">
            <span>File size</span>
            <span>{_size(layout.total_bytes) if layout else UNKNOWN}</span>
          </div>
        </div>
        <div class="fm-tier-contents">
          {_row("GGUF file:", model_name, model_name)}
          {_row("Access mode:", access)}
        </div>
      </div>
    </div>
    """


def _gpu_layers_section(plan: RunPlan, gpu_parts: list[LayerPlacement], split_parts: list[LayerPlacement]) -> str:
    if not gpu_parts and not split_parts:
        return ""
    k, n = plan.layout.experts_per_token, plan.layout.expert_count
    cards = [f"""
        <div class="fm-gpu-core-card">
          <div class="fm-core-header">
            <div class="fm-core-name"><span class="dot-flame"></span><strong>Layer {p.layer:02d}</strong></div>
            <div class="fm-core-status">IN VRAM</div>
            <div class="fm-core-size">{_size(p.gpu_bytes)}</div>
          </div>
          <div class="fm-core-body">
            <div class="fm-core-meta-row"><span>Experts:</span><strong>{n}, top-{k} per token</strong></div>
            <div class="fm-core-meta-row"><span>Expert weights:</span><strong>all on GPU</strong></div>
          </div>
        </div>""" for p in gpu_parts]
    cards += [f"""
        <div class="fm-gpu-core-card fm-core-split">
          <div class="fm-core-header">
            <div class="fm-core-name"><span class="dot-split"></span><strong>Layer {p.layer:02d} (Split)</strong></div>
            <div class="fm-core-status">SPLIT</div>
            <div class="fm-core-size">{_size(p.gpu_bytes)} GPU + {_size(p.cpu_bytes)} RAM</div>
          </div>
          <div class="fm-core-body">
            <div class="fm-core-meta-row"><span>Experts:</span><strong>{n}, top-{k} per token</strong></div>
            <div class="fm-core-meta-row"><span>Why split:</span>
              <strong>part of this layer's expert weights moved to RAM so the rest fit in VRAM</strong></div>
          </div>
        </div>""" for p in split_parts]
    count = len(gpu_parts) + len(split_parts)
    return f"""
    <div class="fm-gpu-cores-section">
      <div class="fm-cores-header">
        <div>
          <h3>🔥 Expert layers in GPU VRAM ({count})</h3>
          <p>The experts of these layers are stored on the GPU. Each token still uses only {k} of their {n} experts.</p>
        </div>
        <div class="fm-cores-badge"><span>{count} of {len(plan.placement)} layers on GPU</span></div>
      </div>
      <div class="fm-gpu-cores-grid">{"".join(cards)}</div>
    </div>
    """


def _matrix(plan: RunPlan | None, moving: set[int]) -> str:
    if not plan:
        return """
    <div class="fm-matrix-section">
      <div class="fm-matrix-header">
        <div class="fm-matrix-title">
          <h3>Layer Topology</h3>
          <p>No model placed yet. Press Preview Placement to map every layer.</p>
        </div>
      </div>
    </div>
    """
    chips = []
    cpu_attention = set(cpu_attention_layers(plan))
    for p in plan.placement:
        if p.layer in cpu_attention:
            cls, label = "layer-cpu", "CPU"
            desc = f"L{p.layer:02d} entirely on the CPU, attention included ({_size(p.cpu_bytes)} experts)"
        elif p.gpu_bytes and not p.cpu_bytes:
            cls, label, desc = "layer-gpu", "GPU", f"L{p.layer:02d} in VRAM ({_size(p.gpu_bytes)})"
        elif p.gpu_bytes and p.cpu_bytes:
            cls, label = "layer-split", "SPLIT"
            desc = f"L{p.layer:02d} split (GPU: {_size(p.gpu_bytes)}, RAM: {_size(p.cpu_bytes)})"
        else:
            cls, label, desc = "layer-ram", "RAM", f"L{p.layer:02d} in RAM ({_size(p.cpu_bytes)})"
        if p.layer in moving:
            desc += " · moved off the GPU compared with the previous plan"
        chips.append(f'<div class="fm-layer-item {cls}" title="{escape(desc)}">'
                     f'<span class="idx">L{p.layer:02d}</span><span class="tier">{label}</span></div>')
    return f"""
    <div class="fm-matrix-section">
      <div class="fm-matrix-header">
        <div class="fm-matrix-title">
          <h3>Layer Topology ({len(plan.placement)} MoE Layers)</h3>
          <p>Hover over any layer to inspect its expert weight sizes</p>
        </div>
        <div class="fm-matrix-legend">
          <div class="fm-legend-item"><span class="fm-legend-chip chip-gpu"></span><span>GPU VRAM (Hot)</span></div>
          <div class="fm-legend-item"><span class="fm-legend-chip chip-split"></span><span>Split Layer</span></div>
          <div class="fm-legend-item"><span class="fm-legend-chip chip-ram"></span><span>System RAM (Warm)</span></div>
          {'<div class="fm-legend-item"><span class="fm-legend-chip chip-cpu"></span><span>CPU only (attention too)</span></div>'
           if cpu_attention else ""}
        </div>
      </div>
      <div class="fm-layer-grid">{"".join(chips)}</div>
    </div>
    """


def render_board(view: BoardView) -> str:
    """Explainer cards, three memory tiers, GPU layer cards and the full layer matrix."""
    plan = view.plan
    placement: tuple[LayerPlacement, ...] = plan.placement if plan else ()
    moving = moved_off_gpu(plan, view.previous) if plan else set()

    gpu_parts = [p for p in placement if p.gpu_bytes and not p.cpu_bytes]
    split_parts = [p for p in placement if p.gpu_bytes and p.cpu_bytes]
    ram_parts = [p for p in placement if p.cpu_bytes and not p.gpu_bytes]
    gpu_bytes = sum(p.gpu_bytes for p in placement)
    ram_bytes = sum(p.cpu_bytes for p in placement)

    caption = view.message or (summary_sentence(plan) if plan
                               else "Click 'Preview Placement' to see where the model fits, or 'Apply & Start' to run it.")
    warning = context_warning(plan) if plan else None
    warning_card = (f'<div class="fm-summary-card fm-warning-card"><div class="icon">⚠️</div>'
                    f'<div><strong>Slow layers:</strong> {escape(warning)}</div></div>') if warning else ""
    return f"""
    <div class="fm-visualizer-wrapper">
      {warning_card}
      {_explainer(plan, len(gpu_parts), len(split_parts), len(ram_parts), gpu_bytes, ram_bytes)}
      {_tiers(view, gpu_parts, ram_parts, gpu_bytes, ram_bytes)}
      {_gpu_layers_section(plan, gpu_parts, split_parts) if plan else ""}
      {_matrix(plan, moving)}
      <div class="fm-summary-card">
        <div class="icon">💡</div>
        <div><strong>Current Allocation:</strong> {escape(caption)}</div>
      </div>
    </div>
    """
