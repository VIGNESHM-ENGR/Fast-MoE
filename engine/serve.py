"""Zero-flag entrypoint: `python -m engine.serve --model <hf id>`.

Profiles hardware, derives the launch plan, fetches weights, then spawns and
supervises the sglang-kt server. See TASKS.md milestone M4.
"""
