"""Small, offline-first entry point for quantization comparisons."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from .integrations.matrix import METHOD_INFO, compare_matrices


def build_parser():
    parser = argparse.ArgumentParser(description="AGQ quantization research tools (no bundled weights)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("methods", help="show matrix adapters and their local availability")
    comparison = sub.add_parser("compare", help="compare reconstructed matrices; this is not a robot evaluation")
    comparison.add_argument("--methods", default="rtn", help="comma-separated: rtn,hqq,quanto")
    comparison.add_argument("--bits", type=int, default=4)
    comparison.add_argument("--group-size", type=int, default=64)
    comparison.add_argument("--weights", type=Path, help="optional local 2-D .npy array (never uploaded)")
    comparison.add_argument("--inputs", type=Path, help="optional matching local .npy input array")
    comparison.add_argument("--rows", type=int, default=32)
    comparison.add_argument("--columns", type=int, default=128)
    comparison.add_argument("--samples", type=int, default=64)
    comparison.add_argument("--seed", type=int, default=7)
    comparison.add_argument("--output", type=Path, help="save only scalar metrics as JSON")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "methods":
        info = {}
        for key, spec in METHOD_INFO.items():
            module = {"rtn": "numpy", "hqq": "hqq", "quanto": "optimum.quanto"}[key]
            try:
                available = importlib.util.find_spec(module) is not None
            except ModuleNotFoundError:
                available = False
            info[key] = {**spec, "installed": available, "scope": "matrix reconstruction"}
        print(json.dumps(info, indent=2))
        return 0
    try:
        if min(args.rows, args.columns, args.samples) <= 0:
            raise ValueError("rows, columns, and samples must be positive")
        for path in (args.weights, args.inputs):
            if path is not None and path.suffix.lower() != ".npy":
                raise ValueError("matrix inputs must use the .npy array format")
        rng = np.random.default_rng(args.seed)
        weight = np.load(args.weights, allow_pickle=False) if args.weights else rng.standard_normal((args.rows, args.columns), dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError("weights must be a 2-D array")
        inputs = np.load(args.inputs, allow_pickle=False) if args.inputs else rng.standard_normal((args.samples, weight.shape[1]), dtype=np.float32)
        result = compare_matrices(weight, inputs, [m.strip() for m in args.methods.split(",")], args.bits, args.group_size)
        result["data_origin"] = {"weight": "local_user_matrix" if args.weights else "synthetic", "inputs": "local_user_inputs" if args.inputs else "synthetic", "seed": args.seed}
        rendered = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            if any(path is not None and path.resolve() == args.output.resolve() for path in (args.weights, args.inputs)):
                raise ValueError("output must not overwrite a source matrix")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f"agq: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
