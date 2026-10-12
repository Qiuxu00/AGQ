"""Encode/decode user-local packed bytes using grouped GHC1 and raw fallback."""
import argparse
import json
from pathlib import Path

from agq.core.codec import encode, decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["encode", "decode"])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--backend", choices=["python", "native"], default="python")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if args.mode == "encode":
        data, report = encode(raw, args.bits, backend=args.backend)
    else:
        data, report = decode(raw), {"decoded_bytes": None}
        report["decoded_bytes"] = len(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
