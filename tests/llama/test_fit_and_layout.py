import os
import socket

import pytest

from engine.llama.fit import FitError, parse_fit_args
from engine.llama.layout import ExpertTensor, ModelLayout, describe_ranges, expert_placement
from engine.llama.server import ServerError, find_binary, port_in_use

MiB = 1024**2

# Captured from llama-fit-params (llama.cpp b10937) for Qwen3-30B-A3B Q4_K_M on a 6 GB RTX 3060 Laptop.
REAL_OUTPUT = (
    "some log line\n"
    '-c 4096 -ngl 49 -ot "blk\\.8\\.ffn_down.*=CPU,'
    'blk\\.9\\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,'
    'blk\\.10\\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"\n'
)


def layout(blocks=12):
    tensors = []
    for layer in range(blocks):
        for kind, size in (("gate", 108), ("up", 108), ("down", 157)):
            tensors.append(ExpertTensor(f"blk.{layer}.ffn_{kind}_exps.weight", layer, size * MiB))
    return ModelLayout("qwen3moe", blocks, 128, 8, 40960, 0, tuple(tensors))


def test_parse_real_fit_output():
    fit = parse_fit_args(REAL_OUTPUT)
    assert fit.context == 4096 and fit.gpu_layers == 49
    assert fit.cpu_patterns[0] == "blk\\.8\\.ffn_down.*"
    assert len(fit.cpu_patterns) == 3
    assert fit.args[:4] == ("-c", "4096", "-ngl", "49")


def test_parse_empty_output_fails():
    with pytest.raises(FitError):
        parse_fit_args("\n\n")


def test_placement_full_split_and_cpu_layers():
    placement = expert_placement(layout(), gpu_layers=13, cpu_patterns=parse_fit_args(REAL_OUTPUT).cpu_patterns)
    by_layer = {p.layer: p for p in placement}
    assert by_layer[7].cpu_bytes == 0 and by_layer[7].gpu_bytes == 373 * MiB
    assert by_layer[8].cpu_bytes == 157 * MiB and by_layer[8].gpu_bytes == 216 * MiB  # only ffn_down on CPU
    assert by_layer[9].gpu_bytes == 0 and by_layer[10].gpu_bytes == 0


def test_layers_not_offloaded_keep_experts_on_cpu():
    # -ngl 5 on 12 blocks offloads the last layers only (i_gpu_start = 12 + 1 - 5 = 8)
    placement = expert_placement(layout(), gpu_layers=5, cpu_patterns=())
    assert [p.layer for p in placement if p.gpu_bytes] == [8, 9, 10, 11]


def test_describe_ranges():
    assert describe_ranges([5, 0, 1, 2, 6, 9]) == "0-2, 5-6, 9"
    assert describe_ranges([]) == "none"


def test_find_binary_prefers_env_dir(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("LLAMA_CPP_BIN_DIR", str(tmp_path))
    assert find_binary("llama-server") == exe
    with pytest.raises(ServerError, match="build_llama_cpp.sh"):
        find_binary("definitely-not-a-llama-binary")


def test_port_in_use():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        assert port_in_use("127.0.0.1", port)


@pytest.mark.skipif(not os.environ.get("FAST_MOE_REAL_GGUF"), reason="set FAST_MOE_REAL_GGUF to a model path")
def test_read_real_gguf_layout():
    from pathlib import Path

    from engine.llama.layout import read_layout
    real = read_layout(Path(os.environ["FAST_MOE_REAL_GGUF"]))
    assert real.block_count > 0 and real.expert_count > 0 and real.expert_tensors
