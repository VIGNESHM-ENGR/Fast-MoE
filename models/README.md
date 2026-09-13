# models/

Downloaded GGUF weights live here (everything except this README is
git-ignored). Fetch the default model with:

```bash
python -m engine.models.model_downloader                 # Qwen/Qwen3-30B-A3B-GGUF, Q4_K_M (18.6 GB)
python -m engine.models.model_downloader --list          # show available quantizations
python -m engine.models.model_downloader --repo <hf-repo> --quant Q4_K_M
```

Files land in `models/<repo-name>/`. Interrupted downloads resume where they
stopped. Set `FAST_MOE_MODELS_DIR` to store models elsewhere (e.g. a bigger
NVMe drive).
