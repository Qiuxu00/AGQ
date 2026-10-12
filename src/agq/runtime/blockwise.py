"""Decode together on block entry, release together even after exceptions."""
from types import MethodType

import torch


class BlockLifetime:
    def __init__(self):
        self.active = None
        self.entries = self.exits = self.max_dense_bytes = self.max_live_matrices = 0
        self.calls = {}
        self.groups = {}
        self.originals = {}

    def report(self):
        if self.active is not None or self.entries != self.exits or any(m._temporary is not None for g in self.groups.values() for m in g):
            raise RuntimeError("dense block weights escaped their lifetime")
        return {"blocks": len(self.groups), "entries": self.entries, "exits": self.exits,
                "calls_by_block": dict(self.calls), "max_live_matrices": self.max_live_matrices,
                "max_temporary_dense_weight_bytes": self.max_dense_bytes,
                "dense_weights_live_outside_block": 0}

    def remove(self):
        self.report()
        for name, (block, original) in self.originals.items():
            block.forward = original
            for module in self.groups[name]:
                module._block_owned = False
        self.originals.clear()


def install_block_lifetime(model, groups):
    """groups maps block paths to full PackedLinear module paths.

Blocks must be disjoint (no nesting) and each target belongs to one block.
This wrapper is for inference, not concurrent forwards or gradient checkpointing.
"""
    from .linear import PackedLinear
    names = list(groups)
    if not names or any(a != b and (a.startswith(b + ".") or b.startswith(a + ".")) for a in names for b in names):
        raise ValueError("nonempty disjoint block groups are required")
    life, seen = BlockLifetime(), set()
    for name, paths in groups.items():
        block = model.get_submodule(name)
        group = []
        if not paths:
            raise ValueError("empty block group")
        for path in paths:
            module = model.get_submodule(path)
            if path in seen or not path.startswith(name + ".") or not isinstance(module, PackedLinear) or module._block_owned:
                raise ValueError(f"invalid or duplicate block target: {path}")
            seen.add(path)
            group.append(module)
        life.groups[name] = group
        life.originals[name] = (block, block.forward)
    for name, (block, original) in life.originals.items():
        group = life.groups[name]
        for module in group:
            module._block_owned = True

        def forward(self, *args, _name=name, _group=group, _original=original, **kwargs):
            if life.active is not None:
                raise RuntimeError("multiple active blocks or reentrant forward")
            life.active = _name
            life.entries += 1
            life.calls[_name] = life.calls.get(_name, 0) + 1
            # eval() alone leaves autograd enabled. Enforce this wrapper's
            # inference-only contract so operations cannot retain recovered
            # dense weights in a backward graph after the block has returned.
            with torch.inference_mode():
                try:
                    for module in _group:
                        if module._temporary is not None:
                            raise RuntimeError("stale dense block weight")
                        module._temporary = module.decode()
                    life.max_live_matrices = max(life.max_live_matrices, len(_group))
                    life.max_dense_bytes = max(life.max_dense_bytes, sum(m._temporary.numel() * m._temporary.element_size() for m in _group))
                    return _original(*args, **kwargs)
                finally:
                    for module in _group:
                        module._temporary = None
                    life.active = None
                    life.exits += 1

        block.forward = MethodType(forward, block)
    return life
