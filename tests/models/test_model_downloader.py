from types import SimpleNamespace

import pytest

from engine.models import model_downloader as dl

GiB = 1024**3

FILES = {
    "Qwen3-30B-A3B-Q4_K_M.gguf": 18 * GiB,
    "Qwen3-30B-A3B-Q4_K_S.gguf": 17 * GiB,
    "Qwen3-30B-A3B-Q8_0.gguf": 32 * GiB,
    "Q4_K/Big-Q4_K-00001-of-00002.gguf": 40 * GiB,
    "Q4_K/Big-Q4_K-00002-of-00002.gguf": 30 * GiB,
}


class FakeApi:
    def __init__(self, files):
        self.files = files

    def model_info(self, repo_id, files_metadata):
        siblings = [SimpleNamespace(rfilename=n, size=s) for n, s in self.files.items()]
        siblings.append(SimpleNamespace(rfilename="README.md", size=10))
        return SimpleNamespace(siblings=siblings)


def test_lists_only_gguf_files():
    assert dl.list_gguf_files("x/y", FakeApi(FILES)) == FILES


def test_quant_match_is_exact_token():
    assert list(dl.select_quant(FILES, "Q4_K_M")) == ["Qwen3-30B-A3B-Q4_K_M.gguf"]
    assert list(dl.select_quant(FILES, "q8_0")) == ["Qwen3-30B-A3B-Q8_0.gguf"]
    # Q4_K picks the multi-part set but not Q4_K_M / Q4_K_S
    assert sorted(dl.select_quant(FILES, "Q4_K")) == [
        "Q4_K/Big-Q4_K-00001-of-00002.gguf", "Q4_K/Big-Q4_K-00002-of-00002.gguf"]


def test_quant_names_for_error_message():
    assert dl.quant_names(FILES) == ["Q4_K", "Q4_K_M", "Q4_K_S", "Q8_0"]


def test_complete_files_are_not_counted_again(tmp_path):
    files = {"a.gguf": 100, "b.gguf": 200}
    (tmp_path / "a.gguf").write_bytes(b"x" * 100)
    (tmp_path / "b.gguf").write_bytes(b"x" * 50)  # partial
    assert dl.bytes_still_needed(tmp_path, files) == 200


def test_missing_quant_lists_alternatives(tmp_path):
    with pytest.raises(SystemExit, match="Available: .*Q8_0"):
        dl.download("x/y", "IQ2_XXS", tmp_path, FakeApi(FILES))


def test_refuses_when_disk_too_small(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * GiB))
    with pytest.raises(dl.NotEnoughDisk, match="Need 18.0 GiB"):
        dl.download("Qwen/Qwen3-30B-A3B-GGUF", "Q4_K_M", tmp_path, FakeApi(FILES))


def test_downloads_into_repo_named_folder(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _: SimpleNamespace(free=500 * GiB))
    monkeypatch.setattr(dl, "hf_hub_download",
                        lambda repo, name, local_dir: calls.append((repo, name, local_dir)) or f"{local_dir}/{name}")
    paths = dl.download("Qwen/Qwen3-30B-A3B-GGUF", "Q4_K_M", tmp_path, FakeApi(FILES))
    assert calls == [("Qwen/Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q4_K_M.gguf", tmp_path / "Qwen3-30B-A3B-GGUF")]
    assert paths[0].name == "Qwen3-30B-A3B-Q4_K_M.gguf"
