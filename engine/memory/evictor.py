"""LRU / token-decay eviction policy deciding which KV blocks and cold
experts spill from VRAM -> RAM -> NVMe, and when to promote them back.

See TASKS.md milestone M8 (built only where upstream sglang-kt falls short).
"""
