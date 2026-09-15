from engine.models.catalog import CATALOG, default_ctx, find


def test_find_by_repo_file_and_folder():
    assert find("unsloth/gemma-4-26B-A4B-it-GGUF").name == "Gemma 4 26B-A4B"
    assert find("/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf").name == "Gemma 4 26B-A4B"
    assert find("models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf").name == "Qwen3.6-35B-A3B"
    assert find("models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf").name == "Qwen3-30B-A3B"


def test_similar_names_do_not_collide():
    # "Qwen3-30B-A3B" must not match a Qwen3.6 file and vice versa.
    assert find("Qwen3.6-35B-A3B-Q4_K_M.gguf").name == "Qwen3.6-35B-A3B"
    assert find("some/other-model.gguf") is None
    assert find(None) is None


def test_recommended_sampling_matches_model_cards():
    by_name = {m.name: m.recommended for m in CATALOG}
    gemma = by_name["Gemma 4 26B-A4B"]
    assert (gemma.temperature, gemma.top_p, gemma.top_k) == (1.0, 0.95, 64)
    qwen36 = by_name["Qwen3.6-35B-A3B"]
    assert (qwen36.temperature, qwen36.top_p, qwen36.top_k, qwen36.presence_penalty) == (1.0, 0.95, 20, 1.5)


def test_default_context_follows_measurements():
    assert default_ctx("ggml-org/Qwen3.6-35B-A3B-GGUF") == 65536
    assert default_ctx("/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf") == 32768
    assert default_ctx("some/other-model.gguf") == 4096
