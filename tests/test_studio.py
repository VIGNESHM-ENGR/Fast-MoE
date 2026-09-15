from engine.models.catalog import Sampling
from ui.studio import (
    LLAMA_CPP_DEFAULTS,
    MODEL_CARD,
    SYSTEM_PRESETS,
    build_request,
    model_card_html,
    preset_sampling,
)

GEMMA = "/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"


def test_model_card_preset_uses_official_values_and_says_so():
    sampling, note = preset_sampling(MODEL_CARD, GEMMA)
    assert (sampling.temperature, sampling.top_p, sampling.top_k) == (1.0, 0.95, 64)
    assert "Gemma 4 26B-A4B model card" in note


def test_unknown_model_falls_back_to_llama_cpp_defaults():
    sampling, note = preset_sampling(MODEL_CARD, "/models/x/unknown.gguf")
    assert sampling == LLAMA_CPP_DEFAULTS and "llama.cpp's defaults" in note


def test_named_presets_are_labelled_as_fast_moe_presets():
    sampling, note = preset_sampling("Creative", GEMMA)
    assert sampling.temperature > 1.0 and "Fast-MoE" in note


def test_build_request_adds_system_prompt_and_llama_server_extensions():
    s = Sampling(temperature=0.6, top_p=0.9, top_k=20, min_p=0.05, presence_penalty=1.5, repeat_penalty=1.1)
    req = build_request("  Be brief.  ", [{"role": "user", "content": "hi"}], s, 512.0, thinking=False)
    assert req["messages"][0] == {"role": "system", "content": "Be brief."}
    assert (req["temperature"], req["top_p"], req["presence_penalty"], req["max_tokens"]) == (0.6, 0.9, 1.5, 512)
    assert req["extra_body"] == {"top_k": 20, "min_p": 0.05, "repeat_penalty": 1.1,
                                 "chat_template_kwargs": {"enable_thinking": False}}


def test_empty_system_prompt_sends_no_system_message():
    req = build_request(SYSTEM_PRESETS["None"], [{"role": "user", "content": "hi"}], LLAMA_CPP_DEFAULTS, 256, True)
    assert [m["role"] for m in req["messages"]] == ["user"]


def test_model_card_html_for_catalog_and_custom_models():
    html = model_card_html(GEMMA)
    assert "Gemma 4 26B-A4B" in html and "on disk" in html and "top-k 64" in html
    assert "15.8 GiB download" in model_card_html("unsloth/gemma-4-26B-A4B-it-GGUF")
    custom = model_card_html("/models/x/<b>odd</b>.gguf")
    assert "custom" in custom and "<b>odd" not in custom
