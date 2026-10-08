#!/usr/bin/env python3
"""Pick catalog models with the space bar and download their Q4 GGUF files.

    python3 models/downloader.py

Installs `huggingface_hub` first if it is missing, then shows the Fast-MoE
catalog: up/down (or j/k) to move, space to select or unselect, `a` to select
all or none, enter to download, `q` to quit. Downloads go through
`engine.models.model_downloader`, so files land in `models/<repo-name>/`,
finished files are skipped, and a model that doesn't fit on disk is reported
and skipped.
"""

from __future__ import annotations

import curses
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.models.catalog import CATALOG, CatalogModel  # stdlib-only, safe before install

REQUIREMENTS = ["huggingface_hub>=1.0"]  # same pin as pyproject.toml


def install_requirements() -> None:
    if importlib.util.find_spec("huggingface_hub"):
        return
    if not importlib.util.find_spec("pip"):
        sys.exit("pip is missing. Install it (Ubuntu/JetPack: sudo apt install python3-pip) and rerun.")
    print(f"Installing {', '.join(REQUIREMENTS)} into {sys.executable} ...")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *REQUIREMENTS]
    if sys.prefix == sys.base_prefix:  # not in a venv: don't touch system site-packages
        cmd.insert(4, "--user")
    if subprocess.call(cmd) != 0:
        sys.exit("pip install failed. Create a venv and rerun from it:\n"
                 "  python3 -m venv .venv && .venv/bin/python models/downloader.py")


def pick(stdscr: curses.window, models: tuple[CatalogModel, ...], present: set[str]) -> list[CatalogModel]:
    curses.curs_set(0)
    selected = [False] * len(models)
    row = 0
    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, "Select models: up/down move, space toggle, a all/none, enter download, q quit")
        for i, m in enumerate(models):
            mark = "[x]" if selected[i] else "[ ]"
            note = "  (downloaded)" if m.repo in present else ""
            line = f" {mark} {m.name:<20} {m.quant:<7} {m.download_gib:5.1f} GiB  {m.repo}{note}"
            stdscr.addnstr(i + 2, 0, line, curses.COLS - 1, curses.A_REVERSE if i == row else 0)
        total = sum(m.download_gib for m, s in zip(models, selected) if s)
        stdscr.addstr(len(models) + 3, 0, f"Selected: {sum(selected)} model(s), {total:.1f} GiB")
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            row = (row - 1) % len(models)
        elif key in (curses.KEY_DOWN, ord("j")):
            row = (row + 1) % len(models)
        elif key == ord(" "):
            selected[row] = not selected[row]
        elif key == ord("a"):
            selected = [not all(selected)] * len(models)
        elif key in (curses.KEY_ENTER, 10, 13):
            return [m for m, s in zip(models, selected) if s]
        elif key in (ord("q"), 27):
            return []


def main() -> None:
    if not sys.stdin.isatty():
        sys.exit("downloader.py is interactive; for scripts use: python -m engine.models.model_downloader")
    install_requirements()
    from engine.models.model_downloader import NotEnoughDisk, download, local_model

    present = {m.repo for m in CATALOG if local_model(m.repo, m.quant)}
    chosen = curses.wrapper(pick, CATALOG, present)
    if not chosen:
        print("Nothing selected.")
        return

    failed = []
    for m in chosen:
        print(f"\n== {m.name} ==")
        try:
            for path in download(m.repo, m.quant):
                print(path)
        except (NotEnoughDisk, SystemExit) as exc:
            print(f"Skipped: {exc}")
            failed.append(m.name)
    if failed:
        sys.exit(f"\nNot downloaded: {', '.join(failed)}")
    print("\nAll selected models downloaded.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Rerun to resume; finished files are kept.")
