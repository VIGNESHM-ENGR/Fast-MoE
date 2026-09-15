"""Chat Studio settings: system prompt presets, sampling presets and the request they produce."""

from __future__ import annotations

from html import escape
from pathlib import Path

from engine.models import catalog
from engine.models.catalog import Sampling

SYSTEM_PRESETS: dict[str, str] = {
    "None": "",
    "Helpful": "You are a helpful, accurate assistant. Answer clearly and say so when you are unsure.",
    "Concise": "You are an expert. Answer in as few words as possible; use short bullet points when they help.",
    "Coder": ("You are a senior software engineer. Give working code first, then a short explanation. "
              "Point out bugs, security issues and edge cases."),
    "Teacher": "You are a patient teacher. Explain step by step in plain language and end with a short example.",
}

MODEL_CARD = "Model card"
TUNING_PRESETS = (MODEL_CARD, "Focused", "Balanced", "Creative")

# llama.cpp's built-in defaults (common/common.h), used when a model has no catalog entry.
LLAMA_CPP_DEFAULTS = Sampling(temperature=0.8, top_p=0.95, top_k=40, min_p=0.05)

FAST_MOE_PRESETS: dict[str, Sampling] = {
    "Focused": Sampling(temperature=0.3, top_p=0.9, top_k=20, min_p=0.05),
    "Balanced": Sampling(temperature=0.7, top_p=0.9, top_k=40, min_p=0.05),
    "Creative": Sampling(temperature=1.2, top_p=0.98, top_k=100, min_p=0.02),
}


def preset_sampling(preset: str, model: str | Path | None) -> tuple[Sampling, str]:
    """Sampling values for a preset and a one-line note saying where they come from."""
    if preset == MODEL_CARD:
        entry = catalog.find(model)
        if entry:
            return entry.recommended, f"Official settings from the {entry.name} model card."
        return LLAMA_CPP_DEFAULTS, "No model card on file for this model; using llama.cpp's defaults."
    return FAST_MOE_PRESETS[preset], f"Fast-MoE's “{preset}” preset. Tweak any slider to taste."


CUSTOM_NOTE = "Custom settings. Pick a preset to reset the sliders."


def settings_summary(sampling: Sampling, max_tokens: int, thinking: bool) -> str:
    """The sampling settings a reply was requested with, in one line."""
    return (f"temp {sampling.temperature:g} · top-p {sampling.top_p:g} · top-k {sampling.top_k} · "
            f"min-p {sampling.min_p:g} · presence {sampling.presence_penalty:g} · "
            f"repeat {sampling.repeat_penalty:g} · max {int(max_tokens)} · reasoning {'on' if thinking else 'off'}")


def build_request(system_prompt: str, messages: list[dict], sampling: Sampling, max_tokens: int,
                  thinking: bool) -> dict:
    """Keyword arguments for openai `chat.completions.create` against llama-server."""
    system = [{"role": "system", "content": system_prompt.strip()}] if system_prompt.strip() else []
    return {
        "messages": system + messages,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "presence_penalty": sampling.presence_penalty,
        "max_tokens": int(max_tokens),
        # llama-server extensions to the OpenAI schema.
        "extra_body": {
            "top_k": int(sampling.top_k),
            "min_p": sampling.min_p,
            "repeat_penalty": sampling.repeat_penalty,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        },
    }


def model_card_html(model: str | None) -> str:
    entry = catalog.find(model)
    if entry is None:
        name = Path(model).name if model else "No model selected"
        return f"""
        <div class="fm-model-card">
          <div class="fm-model-card-head"><span class="fm-model-name">{escape(name)}</span>
            <span class="fm-model-tag">custom</span></div>
          <p class="fm-model-note">Not in the tested catalog. Any Mixture-of-Experts GGUF that llama.cpp supports
          should work; placement and sizes are still read from the file.</p>
        </div>"""
    s = entry.recommended
    on_disk = Path(str(model)).suffix == ".gguf"
    status = "on disk" if on_disk else f"{entry.download_gib:.1f} GiB download"
    return f"""
        <div class="fm-model-card">
          <div class="fm-model-card-head">
            <span class="fm-model-name">{escape(entry.name)}</span>
            <span class="fm-model-tag {"ready" if on_disk else ""}">{escape(status)}</span>
          </div>
          <div class="fm-model-facts">
            <span>{escape(entry.params)}</span><span>{escape(entry.experts)}</span><span>{escape(entry.quant)}</span>
          </div>
          <div class="fm-model-sampling">
            <span class="k">Model card settings</span>
            <code>temp {s.temperature:g}</code><code>top-p {s.top_p:g}</code><code>top-k {s.top_k}</code>
          </div>
          <a class="fm-model-link" href="{escape(entry.card_url)}" target="_blank" rel="noopener">Model card ↗</a>
        </div>"""
