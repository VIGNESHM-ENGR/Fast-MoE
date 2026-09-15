from types import SimpleNamespace

from ui import metrics

GiB = 1024**3


def test_process_file_rss_reads_proc_status(tmp_path):
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "status").write_text("Name:\tllama-server\nRssAnon:\t  900000 kB\nRssFile:\t17482752 kB\n")
    assert metrics.process_file_rss(42, tmp_path) == 17482752 * 1024
    assert metrics.process_file_rss(7, tmp_path) is None


def fake_vm(monkeypatch, total, available):
    monkeypatch.setattr(metrics.psutil, "virtual_memory", lambda: SimpleNamespace(total=total, available=available))


def test_ram_usage_on_the_host_excludes_page_cache(tmp_path, monkeypatch):
    fake_vm(monkeypatch, 38 * GiB, 30 * GiB)
    assert metrics.ram_usage(tmp_path) == (8 * GiB, 38 * GiB)


def test_ram_usage_in_a_limited_container_uses_the_limit_and_anon_memory(tmp_path, monkeypatch):
    fake_vm(monkeypatch, 38 * GiB, 30 * GiB)
    (tmp_path / "memory.max").write_text(f"{20 * GiB}\n")
    (tmp_path / "memory.current").write_text(f"{19 * GiB}\n")
    (tmp_path / "memory.stat").write_text(f"anon {3 * GiB}\nfile {16 * GiB}\n")
    assert metrics.ram_usage(tmp_path) == (3 * GiB, 20 * GiB)
