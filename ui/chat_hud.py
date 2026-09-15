"""Cockpit Telemetry Cluster Gauge HUD for Fast-MoE Chat Studio.

Renders high-tech speedometer, context usage meter, TTFT chronometer,
and reasoning vs generation phase timers.
"""

from __future__ import annotations

import math
from html import escape


def _arc_path(cx: float, cy: float, r: float, start_angle: float, end_angle: float) -> str:
    """Generates an SVG arc path between two angles (in degrees)."""
    start_rad = math.radians(start_angle)
    end_rad = math.radians(end_angle)
    x1 = cx + r * math.cos(start_rad)
    y1 = cy + r * math.sin(start_rad)
    x2 = cx + r * math.cos(end_rad)
    y2 = cy + r * math.sin(end_rad)
    large_arc = 1 if (end_angle - start_angle) > 180 else 0
    return f"M {x1:.2f} {y1:.2f} A {r:.2f} {r:.2f} 0 {large_arc} 1 {x2:.2f} {y2:.2f}"


def render_speedometer_svg(tps: float, max_tps: float = 35.0) -> str:
    """Renders a 180-degree curved speedometer dial for generation tok/s."""
    cx, cy, r = 70.0, 65.0, 48.0
    start_deg = 180.0
    end_deg = 360.0
    total_deg = end_deg - start_deg

    # Base track
    bg_arc = _arc_path(cx, cy, r, start_deg, end_deg)

    # Active fill arc
    frac = min(max(tps / max_tps, 0.0), 1.0)
    current_deg = start_deg + (total_deg * frac)
    fill_arc = _arc_path(cx, cy, r, start_deg, current_deg) if frac > 0.01 else ""

    # Needle calculation
    needle_rad = math.radians(current_deg)
    nx = cx + (r - 6) * math.cos(needle_rad)
    ny = cy + (r - 6) * math.sin(needle_rad)

    fill_element = f'<path d="{fill_arc}" fill="none" stroke="url(#tps-grad)" stroke-width="8" stroke-linecap="round" filter="url(#glow-tps)" />' if fill_arc else ""

    return f"""
    <svg class="fm-gauge-svg" viewBox="0 0 140 85" width="140" height="85">
      <defs>
        <linearGradient id="tps-grad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#06b6d4" />
          <stop offset="60%" stop-color="#10b981" />
          <stop offset="100%" stop-color="#8b5cf6" />
        </linearGradient>
        <filter id="glow-tps" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>
      <!-- Background Arc -->
      <path d="{bg_arc}" fill="none" stroke="rgba(255, 255, 255, 0.08)" stroke-width="8" stroke-linecap="round" />
      <!-- Active Value Arc -->
      {fill_element}
      <!-- Needle & Pivot -->
      <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" />
      <circle cx="{cx}" cy="{cy}" r="4.5" fill="#06b6d4" stroke="#ffffff" stroke-width="1.5" />
      <!-- Dial readout text -->
      <text x="{cx}" y="{cy - 12}" text-anchor="middle" class="fm-gauge-num">{tps:.1f}</text>
      <text x="{cx}" y="{cy + 14}" text-anchor="middle" class="fm-gauge-unit">TOK/S</text>
    </svg>
    """


def render_context_meter_svg(used: int | None, total: int) -> str:
    """Renders a fuel-style circular ring gauge for Context/KV cache occupancy.

    `used` is None until llama-server reports token usage for the request."""
    cx, cy, r = 70.0, 48.0, 36.0
    total = max(total, 1)
    frac = min(max((used or 0) / total, 0.0), 1.0)
    percent = frac * 100.0
    percent_text = f"{percent:.0f}%" if used is not None else "—"
    used_text = f"{used:,} / {total:,}" if used is not None else f"— / {total:,}"

    # Perimeter of 360 circle
    perimeter = 2 * math.pi * r
    offset = perimeter * (1.0 - frac)

    color = "#06b6d4" if percent < 60 else "#f59e0b" if percent < 85 else "#f43f5e"

    return f"""
    <svg class="fm-gauge-svg" viewBox="0 0 140 85" width="140" height="85">
      <defs>
        <filter id="glow-ctx" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>
      <!-- Background track -->
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(255, 255, 255, 0.08)" stroke-width="7" />
      <!-- Progress ring -->
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="7"
              stroke-dasharray="{perimeter:.2f}" stroke-dashoffset="{offset:.2f}"
              stroke-linecap="round" transform="rotate(-90 {cx} {cy})" filter="url(#glow-ctx)" />
      <!-- Center readouts -->
      <text x="{cx}" y="{cy + 4}" text-anchor="middle" class="fm-gauge-num" fill="{color}">{percent_text}</text>
      <text x="{cx}" y="{cy + 28}" text-anchor="middle" class="fm-gauge-unit">{used_text}</text>
    </svg>
    """


def render_chat_telemetry(
    state: str = "idle",
    tps: float = 0.0,
    ttft_ms: float = 0.0,
    reasoning_s: float = 0.0,
    gen_s: float = 0.0,
    ctx_used: int | None = None,
    ctx_max: int = 4096,
    total_tokens: int = 0,
    speed_note: str = "Awaiting a request",
    sent_note: str = "",
) -> str:
    """Renders the entire cockpit cluster HUD banner for the Chat Studio."""

    # State configs
    state_map = {
        "idle": ("⚪ STANDBY", "state-idle", "Awaiting user prompt input"),
        "starting": ("⚡ INGESTING", "state-starting", "Ingesting prompt tokens..."),
        "reasoning": ("🧠 THINKING", "state-reasoning", "Streaming reasoning chain..."),
        "generating": ("🚀 ANSWERING", "state-generating", "Streaming final answer tokens..."),
        "completed": ("✓ COMPLETE", "state-completed", f"Finished {total_tokens} tokens"),
        "error": ("❌ ERROR", "state-error", "Generation error occurred"),
        "stopped": ("⏹ STOPPED", "state-stopped", f"Stopped after {total_tokens} tokens"),
    }
    badge_text, badge_cls, state_desc = state_map.get(state, ("⚪ READY", "state-idle", ""))

    speedometer_svg = render_speedometer_svg(tps)
    context_svg = render_context_meter_svg(ctx_used, ctx_max)

    # Chronometer formatting
    if ttft_ms >= 1000:
        ttft_str = f"{ttft_ms / 1000:.2f} s"
    elif ttft_ms > 0:
        ttft_str = f"{ttft_ms:.0f} ms"
    else:
        ttft_str = "--"

    reasoning_str = f"{reasoning_s:.2f} s" if reasoning_s > 0 else "--"
    gen_str = f"{gen_s:.2f} s" if gen_s > 0 else "--"
    total_time = reasoning_s + gen_s
    total_str = f"{total_time:.2f} s" if total_time > 0 else "--"

    return f"""
    <div class="fm-hud-cluster">
      <!-- Instrument 1: Cockpit Status & Mode -->
      <div class="fm-hud-cell fm-hud-mode">
        <div class="fm-hud-cell-title">Engine Mode</div>
        <div class="fm-hud-state-pill {badge_cls}">
          <span class="fm-hud-dot"></span>
          <span>{badge_text}</span>
        </div>
        <div class="fm-hud-subtext">{escape(state_desc)}</div>
        <div class="fm-hud-meta">Tokens: <strong>{total_tokens}</strong></div>
      </div>

      <!-- Instrument 2: Speedometer Cluster -->
      <div class="fm-hud-cell fm-hud-gauge-cell">
        <div class="fm-hud-cell-title">Generation Speed</div>
        {speedometer_svg}
        <div class="fm-hud-subtext">{escape(speed_note)}</div>
      </div>

      <!-- Instrument 3: Context Fuel Meter -->
      <div class="fm-hud-cell fm-hud-gauge-cell">
        <div class="fm-hud-cell-title">Context &amp; KV Cache</div>
        {context_svg}
        <div class="fm-hud-subtext">Window occupancy</div>
      </div>

      <!-- Instrument 4: TTFT Chronometer -->
      <div class="fm-hud-cell fm-hud-timer-cell">
        <div class="fm-hud-cell-title">Time to First Token</div>
        <div class="fm-hud-big-timer">{ttft_str}</div>
        <div class="fm-hud-subtext">Prompt processing latency</div>
        <div class="fm-hud-timeline">
          <div class="fm-tl-item"><span class="k">Prompt:</span> <span class="v">{ttft_str}</span></div>
          <div class="fm-tl-item"><span class="k">Total:</span> <span class="v">{total_str}</span></div>
        </div>
      </div>

      <!-- Instrument 5: Reasoning vs Answering Split Timers -->
      <div class="fm-hud-cell fm-hud-timer-cell">
        <div class="fm-hud-cell-title">Phase Split Breakdown</div>
        <div class="fm-phase-meters">
          <div class="fm-phase-row">
            <span class="fm-phase-label"><span class="dot amber"></span> Reasoning:</span>
            <span class="fm-phase-val">{reasoning_str}</span>
          </div>
          <div class="fm-phase-bar-track">
            <div class="fm-phase-bar-fill fill-amber" style="width: {min((reasoning_s / max(total_time, 0.001)) * 100, 100):.0f}%;"></div>
          </div>
          <div class="fm-phase-row">
            <span class="fm-phase-label"><span class="dot cyan"></span> Answering:</span>
            <span class="fm-phase-val">{gen_str}</span>
          </div>
          <div class="fm-phase-bar-track">
            <div class="fm-phase-bar-fill fill-cyan" style="width: {min((gen_s / max(total_time, 0.001)) * 100, 100):.0f}%;"></div>
          </div>
        </div>
        <div class="fm-hud-subtext">Two-phase execution telemetry</div>
      </div>
      {f'<div class="fm-hud-sent">Sent to llama-server: {escape(sent_note)}</div>' if sent_note else ""}
    </div>
    """
