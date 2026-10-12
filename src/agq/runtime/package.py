"""User-local target-weight packages; non-target weights remain in the base model.

Numeric NPZ arrays are loaded with allow_pickle=False. A manifest records SHA256
for each file. This is a release format, not the historical standalone package.
No exported weights are included in or uploaded by this repository.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from agq.core.codec import encode, decode
from .linear import PackedLinear


def save_payloads(directory, payloads, *, codec_backend="python"):
    """Create a new target-only directory; never overwrite an existing export."""
    folder = Path(directory)
    if not payloads:
        raise ValueError("nonempty target payloads required")
    folder.mkdir(parents=True, exist_ok=False)
    records = {}
    for index, (name, payload) in enumerate(sorted(payloads.items())):
        if not isinstance(name, str) or not name:
            raise ValueError("nonempty module names required")
        checked = PackedLinear(payload)
        raw = checked.packed.cpu().numpy().tobytes()
        encoded, codec = encode(raw, checked.bits, backend=codec_backend)
        filename = f"matrix_{index:04d}.npz"
        path = folder / filename
        np.savez(path, encoded=np.frombuffer(encoded, dtype=np.uint8),
                 scale_bf16=checked.scale.cpu().view(torch.uint16).numpy(),
                 zero_or_reserved=checked.zero_or_reserved.cpu().numpy())
        records[name] = {"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "bits": checked.bits, "shape": [checked.out_features, checked.in_features],
                         "codec": codec}
    manifest = {"format": "agq-targets-v1", "target_matrices": len(records), "matrices": records,
                "non_target_weights_included": False, "biases_included": False}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def load_payloads(directory):
    """Verify and restore (payload_mapping, encoded_mapping) for model injection."""
    folder = Path(directory).resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "agq-targets-v1" or manifest.get("target_matrices") != len(manifest.get("matrices", {})):
        raise ValueError("invalid AGQ target manifest")
    payloads, encoded = {}, {}
    for name, record in manifest["matrices"].items():
        path = (folder / record["file"]).resolve()
        if path.parent != folder or path.suffix != ".npz":
            raise ValueError("target file must be an NPZ directly inside the package")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"target file SHA256 mismatch: {name}")
        rows, columns = record["shape"]
        bits = record["bits"]
        if any(type(v) is not int or v <= 0 for v in (rows, columns)) or type(bits) is not int or bits not in (1, 2, 3, 4):
            raise ValueError("invalid target shape or bitwidth")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"encoded", "scale_bf16", "zero_or_reserved"}:
                raise ValueError("unexpected target arrays")
            if data["encoded"].dtype != np.uint8 or data["encoded"].ndim != 1 or data["scale_bf16"].dtype != np.uint16 or data["zero_or_reserved"].dtype != np.uint8:
                raise ValueError("invalid target array dtype")
            blob = data["encoded"].tobytes()
            expected = rows * ((columns * bits + 7) // 8)
            raw = decode(blob, max_output_bytes=expected)
            if len(raw) != expected or hashlib.sha256(raw).hexdigest() != record["codec"]["raw_sha256"]:
                raise ValueError("packed target digest/shape mismatch")
            payload = {"bits": bits, "shape": [rows, columns],
                       "packed": torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).reshape(rows, -1),
                       "scale": torch.from_numpy(data["scale_bf16"].copy()).view(torch.bfloat16),
                       "zero_or_reserved": torch.from_numpy(data["zero_or_reserved"].copy())}
        PackedLinear(payload)  # validate metadata before the caller mutates a model
        payloads[name], encoded[name] = payload, blob
    return payloads, encoded
