"""Download quantized GGUF model files from Hugging Face into `models/`.

`python -m engine.models.model_downloader [--repo REPO] [--quant Q4_K_M] [--list]`

Transfers go through `huggingface_hub`, which resumes interrupted downloads
and skips files that are already complete. Multi-part GGUFs
(`*-00001-of-00003.gguf`) are fetched as a set; llama.cpp loads them from the
first part.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REPO = "Qwen/Qwen3-30B-A3B-GGUF"
DEFAULT_QUANT = "Q4_K_M"
MODELS_DIR_ENV = "FAST_MOE_MODELS_DIR"
DISK_HEADROOM_BYTES = 5 * 1024**3
GiB = 1024**3


class NotEnoughDisk(RuntimeError):
    pass


def models_dir() -> Path:
    default = Path(__file__).resolve().parents[2] / "models"
    return Path(os.environ.get(MODELS_DIR_ENV, default))


def list_gguf_files(repo_id: str, api: HfApi | None = None) -> dict[str, int]:
    """Map of GGUF filename -> size in bytes."""
    info = (api or HfApi()).model_info(repo_id, files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings if s.rfilename.endswith(".gguf")}


def select_quant(files: dict[str, int], quant: str) -> dict[str, int]:
    # Match the quant as a whole token so Q4_K doesn't also pick Q4_K_M or Q4_K_S.
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(quant)}(?![A-Za-z0-9_])", re.IGNORECASE)
    return {name: size for name, size in files.items() if pattern.search(name)}


def quant_names(files: dict[str, int]) -> list[str]:
    found = {m.group(0).upper() for name in files for m in re.finditer(r"(?:UD-)?I?Q\d[A-Z0-9_]*|F16|BF16", name)}
    return sorted(found)


def bytes_still_needed(dest: Path, files: dict[str, int]) -> int:
    return sum(size for name, size in files.items()
               if not ((dest / name).exists() and (dest / name).stat().st_size == size))


def download(repo_id: str = DEFAULT_REPO, quant: str = DEFAULT_QUANT, root: Path | None = None,
             api: HfApi | None = None) -> list[Path]:
    available = list_gguf_files(repo_id, api)
    files = select_quant(available, quant)
    if not files:
        raise SystemExit(f"No {quant} GGUF in {repo_id}. Available: {', '.join(quant_names(available))}")

    dest = (root or models_dir()) / repo_id.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    needed = bytes_still_needed(dest, files)
    free = shutil.disk_usage(dest).free
    if needed + DISK_HEADROOM_BYTES > free:
        raise NotEnoughDisk(
            f"Need {needed / GiB:.1f} GiB (+{DISK_HEADROOM_BYTES / GiB:.0f} GiB headroom) in {dest}, "
            f"only {free / GiB:.1f} GiB free. Set {MODELS_DIR_ENV} to a bigger drive."
        )

    print(f"{repo_id} {quant}: {len(files)} file(s), {sum(files.values()) / GiB:.1f} GiB "
          f"({needed / GiB:.1f} GiB left to download) -> {dest}")
    return [Path(hf_hub_download(repo_id, name, local_dir=dest)) for name in sorted(files)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a quantized GGUF model into models/.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face repo (default {DEFAULT_REPO})")
    parser.add_argument("--quant", default=DEFAULT_QUANT, help=f"quantization (default {DEFAULT_QUANT})")
    parser.add_argument("--list", action="store_true", help="list GGUF files in the repo and exit")
    args = parser.parse_args()

    if args.list:
        for name, size in sorted(list_gguf_files(args.repo).items()):
            print(f"{size / GiB:7.2f} GiB  {name}")
        return
    for path in download(args.repo, args.quant):
        print(path)


if __name__ == "__main__":
    main()
