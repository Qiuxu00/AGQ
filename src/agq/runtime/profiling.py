"""Action-space sensitivity and explicit inference timing boundaries."""
from collections import defaultdict
import math
import time

import numpy as np
import torch


def balanced_action_mse(reference, perturbed, task_ids, trajectory_ids):
    """Equal tasks, equal trajectories per task, equal frames per trajectory.

Arrays are [observations, horizon, 7] normalized actions. The first six
components enter the MSE; the gripper does not. No trajectory length weighting.
"""
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(perturbed, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 7 or not a.shape[0] or not a.shape[1]:
        raise ValueError("matching nonempty [observations, horizon, 7] actions required")
    if len(task_ids) != len(a) or len(trajectory_ids) != len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite actions and matching task/trajectory identifiers required")
    frame_mse = np.mean((a[:, :, :6] - b[:, :, :6]) ** 2, axis=(1, 2))
    groups = defaultdict(lambda: defaultdict(list))
    for value, task, trajectory in zip(frame_mse, task_ids, trajectory_ids):
        groups[task][trajectory].append(float(value))
    return float(np.mean([np.mean([np.mean(frames) for frames in tasks.values()]) for tasks in groups.values()]))


def probe_action_sensitivity(model, matrix_names, samples, predict_normalized, task_ids, trajectory_ids, *, quantize_weight=None):
    """Perturb one matrix to W4 and restore it in finally.

predict_normalized(model, sample) must return [H,7] normalized actions.
The default uses this release's unlearned W4 rowwise quantizer; formal paper
probe bytes came from pinned historical quantizers. Supply quantize_weight
to reproduce that exact probe. This helper does not relabel new scans as paper data.
"""
    from agq.core.calibration import LearnedLinear
    if not samples or not matrix_names or len(set(matrix_names)) != len(matrix_names):
        raise ValueError("nonempty samples and unique matrix names required")
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            reference = np.asarray([predict_normalized(model, sample) for sample in samples])
            records = []
            for name in matrix_names:
                module = model.get_submodule(name)
                if not isinstance(module, torch.nn.Linear):
                    raise TypeError(f"not a Linear: {name}")
                original = module.weight.detach().clone()
                try:
                    weight = LearnedLinear(module, 4).weight() if quantize_weight is None else quantize_weight(module)
                    module.weight.copy_(weight)
                    perturbed = np.asarray([predict_normalized(model, sample) for sample in samples])
                    sensitivity = balanced_action_mse(reference, perturbed, task_ids, trajectory_ids)
                finally:
                    module.weight.copy_(original)
                records.append({"name": name, "rows": module.out_features, "columns": module.in_features,
                                "sensitivity": sensitivity, "probe_bits": 4})
            return records
    finally:
        model.train(was_training)


def benchmark_inference(call, *, device="cpu", warmup=3, repeats=10):
    """Synchronized whole-call wall time; CUDA peak allocated after warmup.

Includes any decoding in call; excludes model construction and warmup. GB is
decimal. This is a portable harness, not a claim to duplicate paper kernel timing.
"""
    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats <= 0:
        raise ValueError("nonnegative warmup and positive repeat counts required")
    device = torch.device(device)
    cuda = device.type == "cuda"
    sync = (lambda: torch.cuda.synchronize(device)) if cuda else (lambda: None)
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        elapsed = []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            call()
            sync()
            elapsed.append((time.perf_counter() - start) * 1000)
    peak = torch.cuda.max_memory_allocated(device) if cuda else None
    return {"latency_ms_mean": math.fsum(elapsed) / repeats, "latency_ms_samples": elapsed,
            "warmup": warmup, "repeats": repeats, "device": str(device),
            "peak_allocated_bytes": peak, "peak_allocated_GB": None if peak is None else peak / 1e9,
            "timing_scope": "whole callable, synchronized wall time after warmup",
            "memory_scope": "PyTorch peak allocated after warmup; no reserved/driver allocation"}
