# models/

Downloaded GGUF weights live here (everything except this README is
git-ignored). Fetch the default model with:

```bash
python -m engine.models.model_downloader                 # ggml-org/Qwen3.6-35B-A3B-GGUF, Q4_K_M (19.0 GiB)
python -m engine.models.model_downloader --list          # show available quantizations
python -m engine.models.model_downloader --repo <hf-repo> --quant Q4_K_M
```

Files land in `models/<repo-name>/`. Interrupted downloads resume where they
stopped. Set `FAST_MOE_MODELS_DIR` to store models elsewhere (e.g. a bigger
NVMe drive).
