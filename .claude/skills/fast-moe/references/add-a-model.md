# Add a model

A model counts as supported once it ran on real hardware with its placement and speed
recorded. Any MoE GGUF that llama.cpp supports can already be loaded by path; this adds it to
the catalog with official settings and verified numbers.

1. **Confirm the model exists and is usable.** Query
   `https://huggingface.co/api/models?search=<name>` for the base repo; note `license` and
   `gated`, total/active parameters, expert count and experts per token from the model card.
2. **Confirm llama.cpp supports it.** Find its architecture name (GGUF `general.architecture`,
   or the model card) in `third_party/llama.cpp/src/llama-arch.cpp`. If absent, the pinned
   llama.cpp is too old: bump `LLAMA_CPP_TAG` and the Docker `BASE` tag first.
3. **Pick a GGUF repo.** Prefer `ggml-org/<model>-GGUF` (built by the llama.cpp team); fall
   back to `unsloth/` or `bartowski/` when ggml-org lacks Q4_K_M. List files with
   `python -m engine.models.model_downloader --repo <repo> --list` and choose a Q4 file
   (Q4_K_M, or unsloth's UD-Q4_K_M; both match `--quant Q4_K_M`).
4. **Download it**: `python -m engine.models.model_downloader --repo <repo> --quant Q4_K_M`.
   If it stalls, rerun with `HF_HUB_DISABLE_XET=1` (see lessons.md).
5. **Record official sampling.** Copy temperature, top-p, top-k, min-p and presence penalty
   from the model card's recommended settings (use its general or thinking-mode values).
6. **Add a `CatalogModel`** to `CATALOG` in `engine/models/catalog.py`: name, repo, quant,
   `file_prefix` (the GGUF file-name start), download size in GiB, params, experts,
   `recommended`, `card_url`.
7. **Test the catalog**: in `tests/models/test_catalog.py`, assert `find()` resolves the repo,
   the file path and the folder, that the name does not collide with similar models, and that
   the sampling matches the card. `ruff` and `pytest` pass.
8. **Run it for real** (verify.md): Preview Placement and Apply & Start in the dashboard, one
   chat with the Model card preset, and note placement, context, time to live and decode speed.
9. **Document**: add a row to "Tested with" in `README.md` with the measured numbers, an entry
   under `[Unreleased]` in `CHANGELOG.md`, and the reference row in `verify.md`.

Done when the model answers in the dashboard, its numbers are in the README, and CI is green.
