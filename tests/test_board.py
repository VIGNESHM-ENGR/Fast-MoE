import re
from dataclasses import replace
from pathlib import Path

from engine.llama.fit import parse_fit_args
from engine.llama.layout import ExpertTensor, LayerPlacement, ModelLayout
from engine.llama.runner import RunPlan, RunSettings
from ui.board import BoardView, moved_off_gpu, render_board, render_titleblock, summary_sentence
from ui.chat_hud import render_chat_telemetry

GiB = 1024**3
MiB = 1024**2


def plan(gpu_layers: list[int], split: list[int], ram: list[int], ctx=4096, path="/m/a.gguf", no_mmap=False):
    placement = []
    for layer in sorted(gpu_layers + split + ram):
        g = 300 * MiB if layer in gpu_layers else 100 * MiB if layer in split else 0
        c = 200 * MiB if layer in split else 300 * MiB if layer in ram else 0
        placement.append(LayerPlacement(layer, g, c))
    n = len(placement)
    layout = ModelLayout("qwen35moe", n, 256, 8, 262144, 20 * GiB,
                         tuple(ExpertTensor(f"blk.{i}.ffn_up_exps.weight", i, 300 * MiB) for i in range(n)),
                         embedding_bytes=600 * MiB)
    fit = parse_fit_args(f"-c {ctx} -ngl {n + 1}")
    return RunPlan(RunSettings(no_mmap=no_mmap), Path(path), layout, fit, tuple(placement), ())


def view(p, **kw):
    return BoardView(state=kw.pop("state", "unpowered"), gpu_name="RTX 3060 Laptop GPU", gpu_total=6 * GiB,
                     ram_total=38 * GiB, plan=p, **kw)


def test_summary_sentence_names_every_zone():
    s = summary_sentence(plan([0, 1, 2], [3], [4, 5, 6]))
    assert "Layers 0-2 run in GPU VRAM" in s
    assert "Layer 3 is split between GPU and RAM" in s
    assert "Layers 4-6 run in System RAM" in s
    assert "4,096 tokens" in s


def test_summary_sentence_all_gpu_and_all_ram():
    assert "All 3 layers fit on the GPU" in summary_sentence(plan([0, 1, 2], [], []))
    assert "all 3 layers run from System RAM" in summary_sentence(plan([], [], [0, 1, 2]))


def test_board_without_plan_renders_and_invents_nothing():
    # Regression: the empty board used to crash building placeholder layers.
    html = render_board(view(None, vram_used=None))
    assert "Preview Placement" in html
    assert not re.search(r"Exp #|GB/s|MB/s|Pruned|Instant Start|0ms PCIe|Optimal", html)
    assert "fm-layer-item" not in html  # no fake layer chips
    assert "—" in html


def test_board_counts_and_sizes_come_from_the_plan():
    html = render_board(view(plan([0, 1], [2], [3, 4]), vram_used=5 * GiB, ram_used=10 * GiB))
    assert "2 GPU · 1 split · 2 RAM" in html
    assert "8 / 256 active" in html and "3.1% of experts per token" in html
    assert html.count('fm-layer-item layer-gpu') == 2
    assert html.count('fm-layer-item layer-split') == 1
    assert html.count('fm-layer-item layer-ram') == 2
    assert "Layer Topology (5 MoE Layers)" in html
    assert "Expert layers in GPU VRAM (3)" in html
    assert "600 MiB" in html  # token embeddings
    assert "5.0 GB / 6.0 GB" in html and "10.0 GB / 38.0 GB" in html
    assert not re.search(r"Exp #|GB/s|MB/s|Pruned|0ms PCIe", html)


def test_unknown_vram_is_shown_as_unknown_not_estimated():
    html = render_board(view(plan([0], [], [1]), vram_used=None, ram_used=4 * GiB))
    assert "In use</span>\n            <span>—" in html


def test_access_mode_follows_settings():
    assert "Memory-mapped (mmap)" in render_board(view(plan([0], [], [1])))
    assert "Loaded into RAM" in render_board(view(plan([0], [], [1], no_mmap=True)))


def test_moved_layers_are_marked_against_the_previous_plan():
    before = plan([0, 1, 2], [], [3])
    after = plan([0], [1], [2, 3])
    assert moved_off_gpu(after, before) == {1, 2}
    assert render_board(view(after, previous=before)).count("moved off the GPU compared with the previous plan") == 2
    assert moved_off_gpu(after, replace(before, model_path=Path("/other.gguf"))) == set()


def test_messages_are_escaped():
    html = render_board(view(None, message="<b>not downloaded</b>"))
    assert "&lt;b&gt;not downloaded&lt;/b&gt;" in html and "<b>not" not in html


def test_titleblock_state_cpu_only_and_optional_disk():
    live = render_titleblock("live", "RTX", 6 * GiB, 38 * GiB, "ready", disk_label="NVME disk")
    assert "state-live" in live and "NVME disk" in live
    cpu = render_titleblock("unpowered", None, 0, 38 * GiB)
    assert "No GPU (CPU only)" in cpu and "fm-pill disk" not in cpu


def test_chat_hud_shows_unknown_context_until_usage_arrives():
    waiting = render_chat_telemetry(state="generating", ctx_used=None, ctx_max=8192)
    assert "— / 8,192" in waiting
    done = render_chat_telemetry(state="completed", ctx_used=812, ctx_max=8192, speed_note="Measured by llama.cpp")
    assert "812 / 8,192" in done and "Measured by llama.cpp" in done
