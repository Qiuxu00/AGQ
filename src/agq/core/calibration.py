"""Fixed-bit per-channel calibration adapted from EXP018/release/learn_quantizer.

OmniQuant-inspired learnable weight clipping and block reconstruction; this is
not the full OmniQuant implementation (no learnable equivalent transforms).
Original weights and bitwidths stay fixed. W1 learns signed magnitudes.
"""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .packing import pack_codes, unpack_codes


def ste_round(value):
    return value + (torch.round(value) - value).detach()


class LearnedLinear(nn.Module):
    def __init__(self, linear, bits):
        super().__init__()
        if not isinstance(linear, nn.Linear) or type(bits) is not int or bits not in (1, 2, 3, 4):
            raise ValueError("an nn.Linear and W1--W4 are required")
        self.bits = bits
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.register_buffer("original", linear.weight.detach().clone())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())
        w = self.original.float()
        if not torch.isfinite(w).all():
            raise ValueError("weights must be finite")
        if bits == 1:
            maximum = w.abs().amax(1, keepdim=True)
            rho = (w.abs().mean(1, keepdim=True) / maximum.clamp_min(1e-8)).clamp(1e-6, 1 - 1e-6)
            self.register_buffer("maximum", maximum)
            self.a = nn.Parameter(torch.logit(rho))
        else:
            self.register_buffer("row_low", w.amin(1, keepdim=True).clamp_max(0))
            self.register_buffer("row_high", w.amax(1, keepdim=True).clamp_min(0))
            self.a_low = nn.Parameter(torch.full_like(self.row_low, 4.0))
            self.a_high = nn.Parameter(torch.full_like(self.row_high, 4.0))

    def codes(self, start=0, end=None):
        w = self.original[start:end].float()
        if self.bits == 1:
            scale = (self.a[start:end].sigmoid() * self.maximum[start:end]).clamp_min(1e-8).to(torch.bfloat16)
            zero = torch.zeros_like(scale, dtype=torch.uint8)
            q = (w >= 0).float()
        else:
            low = self.row_low[start:end] * self.a_low[start:end].sigmoid()
            high = self.row_high[start:end] * self.a_high[start:end].sigmoid()
            scale = ((high - low) / ((1 << self.bits) - 1)).clamp_min(1e-8).to(torch.bfloat16)
            zero = ste_round(-low / scale.float()).clamp(0, (1 << self.bits) - 1)
            q = (ste_round(w / scale.float()) + zero).clamp(0, (1 << self.bits) - 1)
        return q, scale, zero

    def weight(self, start=0, end=None):
        q, s, z = self.codes(start, end)
        return ((2 * q - 1) * s.float() if self.bits == 1 else s.float() * (q - z)).to(torch.bfloat16)

    def forward(self, x):
        # BF16 inputs reproduce the experiment. FP32 inputs make CPU inspection
        # possible while keeping the actual reconstructed weights rounded to BF16.
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.weight().to(x.dtype), bias)


def export_matrix(module):
    """Export rowwise codes and verify exact codes and reconstructed BF16 values."""
    if not isinstance(module, LearnedLinear):
        raise TypeError("expected LearnedLinear")
    rows, columns = module.original.shape
    pieces, scales, zeros = [], [], []
    with torch.no_grad():
        for start in range(0, rows, 128):
            end = min(start + 128, rows)
            q, s, z = module.codes(start, end)
            codes = q.to(torch.uint8).cpu().numpy()
            packed = pack_codes(codes, module.bits)
            decoded = unpack_codes(packed, module.bits, columns)
            if not np.array_equal(decoded, codes):
                raise RuntimeError("integer packing verification failed")
            restored = torch.from_numpy(decoded).to(s.device).float()
            dense = ((2 * restored - 1) * s.float() if module.bits == 1
                     else s.float() * (restored - z.to(torch.uint8).float())).to(torch.bfloat16)
            if not torch.equal(dense, module.weight(start, end)):
                raise RuntimeError("BF16 reconstruction verification failed")
            pieces.append(torch.from_numpy(packed))
            scales.append(s.cpu())
            zeros.append(z.to(torch.uint8).cpu())
    return {"bits": module.bits, "shape": [rows, columns], "packed": torch.cat(pieces),
            "scale": torch.cat(scales), "zero_or_reserved": torch.cat(zeros)}


def _assign(model, name, module):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, module)


def calibrate_block(block, allocation, inputs, *, teacher_inputs=None, epochs=20,
                    learning_rate=0.01, forward=None, loss_fn=None):
    """Fit a copy of a block against its frozen teacher using AdamW.

allocation maps relative Linear names to bits. inputs is a list of student
block inputs; teacher_inputs optionally carries the unquantized upstream path.
forward(block, sample) can handle attention masks/tuple outputs. loss_fn can
mask padding tokens. Inputs must already reside on the block's device.
Returns (calibrated_student, history); never mutates the supplied block.
"""
    samples = list(inputs)
    teacher_samples = samples if teacher_inputs is None else list(teacher_inputs)
    if not samples or len(samples) != len(teacher_samples) or not allocation:
        raise ValueError("nonempty allocation and matching teacher/student samples required")
    if type(epochs) is not int or epochs < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("nonnegative integer epochs and positive learning rate required")
    call = forward or (lambda model, sample: model(sample))
    criterion = loss_fn or (lambda output, target, sample: F.mse_loss(output.float(), target.float()))
    teacher, student = copy.deepcopy(block).eval(), copy.deepcopy(block).eval()
    teacher.requires_grad_(False)
    student.requires_grad_(False)
    learned = {}
    for name, bits in allocation.items():
        module = LearnedLinear(student.get_submodule(name), bits)
        _assign(student, name, module)
        learned[name] = module
    parameters = [p for module in learned.values() for p in module.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0, betas=(0.9, 0.999), eps=1e-8)
    # CPU cache bounds teacher output residency; each target is moved to the
    # output device at use. This still assumes one teacher and student block fit.
    with torch.no_grad():
        targets = [call(teacher, sample).detach().cpu() for sample in teacher_samples]
    del teacher

    def measure():
        with torch.no_grad():
            values = []
            for sample, target in zip(samples, targets):
                output = call(student, sample)
                values.append(float(criterion(output, target.to(output.device), sample)))
            return math.fsum(values) / len(values)

    history = {"initial_loss": measure(), "epoch_loss": [], "epochs": epochs,
               "samples": len(samples), "trainable_scalars": sum(p.numel() for p in parameters),
               "original_weights_frozen": True, "allocation_fixed": True}
    for _ in range(epochs):
        losses = []
        for sample, target in zip(samples, targets):
            optimizer.zero_grad(set_to_none=True)
            output = call(student, sample)
            loss = criterion(output, target.to(output.device), sample)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite reconstruction loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError("missing or nonfinite quantizer gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        history["epoch_loss"].append(math.fsum(losses) / len(losses))
    history["final_loss"] = measure()
    return student, history
