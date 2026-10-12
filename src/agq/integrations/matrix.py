"""Weight reconstruction comparisons with official HQQ and Quanto APIs.

This experiment operates on a matrix and held-out linear-layer inputs. It is
not an OpenVLA/LIBERO evaluation and does not implement the AGQ allocation or
calibration pipeline. References and tested versions are in docs/BASELINES.md.
"""
from __future__ import annotations

import copy
import importlib.util
import math
from importlib.metadata import version

import numpy as np

METHOD_INFO = {
    "rtn": {"name": "Round-to-nearest row-wise reference", "bits": [1, 2, 3, 4, 8], "extra": None},
    "hqq": {"name": "Official HQQ matrix adapter", "bits": [1, 2, 3, 4, 8], "extra": "hqq"},
    "quanto": {"name": "Official Optimum Quanto CPU int8 adapter", "bits": [8], "extra": "quanto"},
}


def _matrix(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty 2-D matrix")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    array = np.array(array, dtype=np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} cannot be represented as finite float32 values")
    return array


def rtn_reconstruct(weight, bits):
    """Simple per-row min/max RTN; binary weights use mean absolute magnitude.

    Float32 metadata is deliberate: this baseline is not AGQ's BF16 learned
    quantizer. The returned byte count includes row padding and metadata.
    """
    weight = _matrix(weight, "weight")
    if type(bits) is not int or bits not in METHOD_INFO["rtn"]["bits"]:
        raise ValueError("RTN bits must be one of 1, 2, 3, 4, 8")
    if bits == 1:
        scale = np.abs(weight).mean(axis=1, keepdims=True)
        codes = (weight >= 0).astype(np.uint8)
        recovered = (2 * codes.astype(np.float32) - 1) * scale
        metadata = scale.nbytes
    else:
        low = np.minimum(weight.min(axis=1, keepdims=True), 0)
        high = np.maximum(weight.max(axis=1, keepdims=True), 0)
        levels = (1 << bits) - 1
        scale = np.maximum((high - low) / levels, np.finfo(np.float32).tiny)
        zero = np.clip(np.rint(-low / scale), 0, levels).astype(np.uint8)
        codes = np.clip(np.rint(weight / scale) + zero, 0, levels).astype(np.uint8)
        recovered = scale * (codes.astype(np.float32) - zero)
        metadata = scale.nbytes + zero.nbytes
    packed_bytes = weight.shape[0] * math.ceil(weight.shape[1] * bits / 8)
    return recovered, packed_bytes + metadata


def _require(module, extra):
    try:
        found = importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        found = False
    if not found:
        raise RuntimeError(f"Optional adapter missing: install with pip install -e '.[{extra}]'")


def _upstream_reconstruct(method, weight, bits, group_size):
    import torch

    torch_weight = torch.from_numpy(weight.copy())
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    layer.weight = torch.nn.Parameter(torch_weight, requires_grad=False)
    if method == "hqq":
        _require("hqq", "hqq")
        from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

        if group_size <= 0 or weight.size % group_size:
            raise ValueError("HQQ requires a positive group size dividing weight.numel()")
        alignment = {1: 8, 2: 4, 4: 2}.get(bits, 1)
        if (weight.size // group_size) % alignment:
            raise ValueError(f"HQQ W{bits} packing requires the number of groups to be divisible by {alignment}")
        settings = BaseQuantizeConfig(nbits=bits, group_size=group_size, axis=1)
        quantized = HQQLinear(layer, quant_config=settings, compute_dtype=torch.float32,
                              device="cpu", del_orig=False)
        with torch.no_grad():
            recovered = quantized.dequantize().float().cpu().numpy()
        package = {"hqq": version("hqq")}
    elif method == "quanto":
        _require("optimum.quanto", "quanto")
        from optimum.quanto import freeze, qint8, quantize

        model = torch.nn.Sequential(copy.deepcopy(layer))
        quantize(model, weights=qint8)
        freeze(model)
        with torch.no_grad():
            recovered = model[0].weight.dequantize().float().cpu().numpy()
        package = {"optimum-quanto": version("optimum-quanto")}
    else:
        raise ValueError(f"Unknown adapter: {method}")
    # Upstream tensor subclasses have method-specific storage/serialization.
    # Never infer a real model size from average bitwidth or dense tensor numel.
    return np.asarray(recovered), None, package


def compare_matrices(weight, inputs, methods=("rtn",), bits=4, group_size=64):
    """Quantize identical weights, then measure common dense reconstruction error.

    Optional methods call their actual upstream quantizers. Latency is excluded:
    this shared dense projection measures numerical error, not optimized kernels.
    No original or quantized weight arrays are written into the result.
    """
    weight = _matrix(weight, "weight")
    inputs = _matrix(inputs, "inputs")
    if inputs.shape[1] != weight.shape[1]:
        raise ValueError("input features must equal weight.shape[1]")
    methods = tuple(methods)
    if type(bits) is not int or type(group_size) is not int or group_size <= 0:
        raise ValueError("bits and group_size must be positive integers, not booleans")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("specify at least one method, without duplicates")
    for method in methods:
        if method not in METHOD_INFO:
            raise ValueError(f"Unknown method {method!r}; choose {', '.join(METHOD_INFO)}")
        if bits not in METHOD_INFO[method]["bits"]:
            raise ValueError(f"{method} does not support {bits} bits in this adapter")
    reference = inputs.astype(np.float64) @ weight.astype(np.float64).T
    results = []
    for method in methods:
        if method == "rtn":
            recovered, payload_bytes = rtn_reconstruct(weight, bits)
            packages = {"numpy": np.__version__}
        else:
            recovered, payload_bytes, packages = _upstream_reconstruct(method, weight, bits, group_size)
        if recovered.shape != weight.shape or not np.isfinite(recovered).all():
            raise RuntimeError(f"{method} returned an invalid reconstructed matrix")
        prediction = inputs.astype(np.float64) @ recovered.astype(np.float64).T
        delta = prediction - reference
        results.append({
            "method": method,
            "bits": bits,
            "weight_mse": float(np.square(recovered.astype(np.float64) - weight).mean()),
            "projection_mse": float(np.square(delta).mean()),
            "projection_relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30)),
            "packed_payload_bytes": payload_bytes,
            "payload_note": "row-packed codes + FP32 scale + optional uint8 zero" if method == "rtn" else "not measured; use upstream serialization for physical storage",
            "packages": packages,
        })
    return {
        "schema_version": 1,
        "evaluation": "matrix_reconstruction_not_robot_task_success",
        "device": "cpu",
        "weight_shape": list(weight.shape),
        "input_shape": list(inputs.shape),
        "group_size": group_size,
        "reference": "same float32 source weights and inputs; float64 dense error accumulation",
        "weights_included": False,
        "results": results,
    }
