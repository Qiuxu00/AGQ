"""CPU calibration -> packed export -> grouped coding -> block-wise inference.

Uses generated tensors only. These errors/timings are not LIBERO results.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from agq.core.calibration import calibrate_block, export_matrix
from agq.core.codec import encode
from agq.runtime.linear import replace_linears
from agq.runtime.blockwise import install_block_lifetime


def run_demo(bits=(1, 2, 3, 4), epochs=4, seed=7):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    block = nn.Sequential(nn.Linear(32, 48), nn.GELU(), nn.Linear(48, 32)).eval()
    samples = [torch.randn(3, 32) for _ in range(8)]
    heldout = torch.randn(5, 32)
    cases = []
    for bit in bits:
        student, history = calibrate_block(block, {"0": bit, "2": bit}, samples, epochs=epochs)
        payloads = {"block." + name: export_matrix(student.get_submodule(name)) for name in ("0", "2")}
        encoded, storage = {}, []
        for name, payload in payloads.items():
            encoded[name], record = encode(payload["packed"].numpy().tobytes(), bit)
            storage.append(record)
        model = nn.Module()
        model.add_module("block", copy.deepcopy(block))
        replace_linears(model, payloads, encoded=encoded)
        lifetime = install_block_lifetime(model, {"block": list(payloads)})
        with torch.inference_mode():
            actual, expected, reference = model.block(heldout), student(heldout), block(heldout)
        if not torch.equal(actual, expected):
            raise RuntimeError("packed inference differs from calibrated BF16 weights")
        cases.append({"bits": bit, "calibration": history,
                      "heldout_output_mse": float((actual - reference).square().mean()),
                      "packed_inference_exact": True,
                      "integer_payload_bytes": sum(r["stored_bytes"] for r in storage),
                      "quantizer_metadata_bytes": sum(3 * p["shape"][0] for p in payloads.values()),
                      "codec_modes": [r["mode"] for r in storage], "lifetime": lifetime.report()})
    return {"kind": "synthetic CPU functional demonstration", "seed": seed,
            "weights_distributed": False, "paper_result": False, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, nargs="+", choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_demo(args.bits, args.epochs, args.seed)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
