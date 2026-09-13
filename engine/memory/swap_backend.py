"""Async tiered transfer engine: VRAM<->RAM via CUDA-stream non-blocking
copies, RAM<->NVMe via an async disk I/O path (backend chosen by benchmark:
io_uring bindings vs. mmap+madvise vs. thread-pool executor).

See TASKS.md milestone M8 (built only where upstream sglang-kt falls short).
"""
