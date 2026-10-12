"""Allocate one backbone from explicit matrix profiles and candidate byte costs."""
import argparse
import json
from pathlib import Path

from agq.core.allocation import allocate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True, help="JSON list of matrix records")
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=300)
    args = parser.parse_args()
    records = json.loads(args.profiles.read_text(encoding="utf-8"))
    result = allocate(records, args.budget_bytes, args.lambda_value, time_limit=args.time_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
