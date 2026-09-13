"""Adapter layer onto KTransformers: turns a generic MoEArchDescriptor (see
engine/models/descriptor.py) plus a hardware budget plan (see
engine/hardware/allocator.py) into a KTransformers module-injection rule
set, instead of hand-written per-model YAML.

See TASKS.md milestone M2.
"""
