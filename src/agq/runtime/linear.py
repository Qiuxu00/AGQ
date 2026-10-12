"""Compressed Linear modules with exact BF16 reconstruction and optional kernels."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agq.core.codec import decode as decode_blob
from agq.core.packing import unpack_codes


class PackedLinear(nn.Module):
    """Store codes/scales only; dense weight exists for one call or owning block.

backend='reference' uses portable PyTorch extraction; 'triton' uses the original
fused GPU kernel and requires CUDA/Triton. BF16 weights are reconstructed before
matrix multiplication. This is not a native low-bit integer GEMM.
"""
    def __init__(self, payload, bias=None, *, device=None, backend="reference"):
        super().__init__()
        self.bits = payload["bits"]
        self.out_features, self.in_features = payload["shape"]
        packed = payload["packed"].detach().cpu()
        unpack_codes(packed.numpy(), self.bits, self.in_features)
        if packed.shape[0] != self.out_features:
            raise ValueError("packed row count mismatch")
        if backend not in ("reference", "triton"):
            raise ValueError("backend must be reference or triton")
        if device is None:
            device = "cpu" if bias is None else bias.device
        scale = payload["scale"].detach().reshape(-1, 1)
        zero = payload["zero_or_reserved"].detach().reshape(-1, 1)
        if scale.dtype != torch.bfloat16 or zero.dtype != torch.uint8:
            raise ValueError("payload requires BF16 scales and uint8 zero/reserved values")
        if scale.numel() != self.out_features or zero.numel() != self.out_features:
            raise ValueError("one scale and zero/reserved value required per row")
        if not torch.isfinite(scale).all() or not (scale > 0).all() or torch.any(zero >= (1 << self.bits)):
            raise ValueError("invalid quantizer metadata")
        if bias is not None and (bias.ndim != 1 or bias.numel() != self.out_features):
            raise ValueError("bias shape mismatch")
        self.register_buffer("packed", packed.to(device))
        self.register_buffer("scale", scale.to(device))
        self.register_buffer("zero_or_reserved", zero.to(device))
        self.register_buffer("bias", None if bias is None else bias.detach().clone().to(device))
        self.register_buffer("encoded", None)
        self.entropy = None
        self.backend = backend
        self._temporary = None
        self._block_owned = False

    def compress(self, blob):
        """Install a verified raw/GHC1 stream, without retaining packed cache."""
        if self._temporary is not None or self.packed is None:
            raise RuntimeError("compress once before block execution")
        raw = decode_blob(blob, max_output_bytes=self.packed.numel())
        if raw != self.packed.cpu().numpy().tobytes():
            raise ValueError("encoded stream does not restore the current packed codes")
        if blob[0] == 0:
            return self
        if self.scale.is_cuda:
            from .entropy import GPUHuffman
            self.entropy = GPUHuffman(blob[1:], self.scale.device)
        else:
            self.encoded = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy())
        self.packed = None
        return self

    def _packed_codes(self):
        if self.packed is not None:
            return self.packed
        if self.entropy is not None:
            return self.entropy.decode().reshape(self.out_features, -1)
        raw = decode_blob(self.encoded.cpu().numpy().tobytes())
        return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).reshape(self.out_features, -1).to(self.scale.device)

    def decode(self):
        packed = self._packed_codes()
        if self.backend == "triton":
            if not packed.is_cuda:
                raise ValueError("Triton backend requires CUDA tensors")
            from types import SimpleNamespace
            from .fast_decode import decode
            proxy = SimpleNamespace(packed=packed, scale=self.scale, zero_or_reserved=self.zero_or_reserved,
                                    out_features=self.out_features, in_features=self.in_features, bits=self.bits)
            return decode(proxy)
        offsets = torch.arange(self.in_features, device=packed.device) * self.bits
        index, shifts = offsets // 8, offsets % 8
        padded = F.pad(packed.to(torch.int32), (0, 1))
        q = ((padded[:, index] | (padded[:, index + 1] << 8)) >> shifts) & ((1 << self.bits) - 1)
        dense = ((2 * q.float() - 1) * self.scale.float() if self.bits == 1
                 else (q.float() - self.zero_or_reserved.float()) * self.scale.float())
        return dense.to(torch.bfloat16)

    def forward(self, x):
        if self._block_owned and self._temporary is None:
            raise RuntimeError("packed Linear called outside its owning block")
        weight = self._temporary if self._temporary is not None else self.decode()
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, weight.to(x.dtype), bias)


def replace_linears(model, payloads, *, backend="reference", encoded=None):
    """Replace named nn.Linear modules in a user-loaded model; returns modules.

The existing model supplies all non-target weights and biases. Loading it may
temporarily require full-precision memory; this is not the paper's empty-model
standalone loader. Exported weights are user-local and are not bundled here.
"""
    prepared = {}
    for name, payload in payloads.items():
        old = model.get_submodule(name)
        if not isinstance(old, nn.Linear) or list(old.weight.shape) != list(payload["shape"]):
            raise ValueError(f"target shape/type mismatch: {name}")
        module = PackedLinear(payload, old.bias, device=old.weight.device, backend=backend)
        if encoded is not None and name in encoded:
            module.compress(encoded[name])
        prepared[name] = module
    # Validate and prepare every target before changing the user's model.
    for name, module in prepared.items():
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, module)
    return prepared
