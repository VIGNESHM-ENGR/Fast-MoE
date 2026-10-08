from pathlib import Path

import pytest

from engine.llama import runner as r
from engine.llama.fit import parse_fit_args
from engine.llama.layout import ExpertTensor, ModelLayout
from engine.llama.server import ServerError

FIT = parse_fit_args('-c 8192 -ngl 5 -ot "blk\\.3\\.ffn_(up|down|gate)_(ch|)exps=CPU"')
LAYOUT = ModelLayout("qwen35moe", 4, 256, 8, 262144, 0, tuple(
    ExpertTensor(f"blk.{i}.ffn_{k}_exps.weight", i, 100) for i in range(4) for k in ("gate", "up", "down")))


def test_server_args_run_exactly_the_fitted_placement():
    args = r.server_args(r.RunSettings(no_mmap=True, extra_args=("--", "-np", "1")), Path("/m.gguf"), FIT)
    assert args[:6] == ("-m", "/m.gguf", "--jinja", "--metrics", "--fit", "off")
    assert args[6:12] == FIT.args
    assert args[-4:] == ("--load-mode", "none", "-np", "1")


def test_fit_params_args_map_user_settings():
    assert r.fit_params_args(r.RunSettings()) == []
    assert r.fit_params_args(r.RunSettings(ctx=32768, vram_margin_mib=512)) == [
        "--fit-ctx", "32768", "--fit-target", "512"]


def test_cpu_moe_keeps_experts_in_ram_at_a_fixed_context():
    # --fit-ctx is a minimum that fit grows to fill the GPU; on unified memory that would
    # take the RAM the memory-mapped experts need, so the context is fixed with -c.
    assert r.fit_params_args(r.RunSettings(ctx=8192, cpu_moe=True)) == ["--cpu-moe", "-c", "8192"]
    assert r.fit_params_args(r.RunSettings(cpu_moe=True)) == ["--cpu-moe"]


def test_plan_run_combines_fit_and_layout(monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    calls = {}
    monkeypatch.setattr(r, "find_binary", lambda name: Path(f"/bin/{name}"))
    monkeypatch.setattr(r, "run_fit_params", lambda b, m, a: calls.setdefault("fit", (b, m, a)) and FIT)
    monkeypatch.setattr(r, "layout_for", lambda p: LAYOUT)
    plan = r.plan_run(r.RunSettings(model=str(model), ctx=8192))
    assert calls["fit"] == (Path("/bin/llama-fit-params"), model, ["--fit-ctx", "8192"])
    assert [p.layer for p in plan.placement if p.cpu_bytes] == [3]
    assert plan.server_args[0:2] == ("-m", str(model))


def test_missing_model_file_is_a_clear_error():
    with pytest.raises(ServerError, match="Model file not found"):
        r.resolve_model("/nope/model.gguf", "Q4_K_M")


def test_repo_model_uses_local_copy_before_downloading(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "local_model", lambda repo, quant: tmp_path / "x.gguf")
    monkeypatch.setattr(r, "download", lambda *a: pytest.fail("should not download"))
    assert r.resolve_model("org/Repo-GGUF", "Q4_K_M") == tmp_path / "x.gguf"


def test_api_host_defaults_to_loopback_and_follows_env(monkeypatch):
    monkeypatch.delenv("FAST_MOE_API_HOST", raising=False)
    assert r.RunSettings().host == "127.0.0.1"
    monkeypatch.setenv("FAST_MOE_API_HOST", "0.0.0.0")
    assert r.RunSettings().host == "0.0.0.0"


def test_server_url_uses_loopback_when_listening_on_all_interfaces(tmp_path):
    from engine.llama.server import LlamaServer

    assert LlamaServer(Path("/x"), [], "0.0.0.0", 8080, tmp_path / "l.log").url == "http://127.0.0.1:8080"
    assert LlamaServer(Path("/x"), [], "192.168.1.5", 9000, tmp_path / "l.log").url == "http://192.168.1.5:9000"
