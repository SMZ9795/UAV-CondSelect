"""Select conditions from a saved matrix using the Python standard library."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.selection import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True,
                        help="JSON containing order and a unit-diagonal similarity matrix")
    parser.add_argument("--lambda", dest="weight", type=float, required=True,
                        help="nonnegative coverage-overlap trade-off weight")
    parser.add_argument("--output", type=Path,
                        help="write the full selection trace as JSON")
    args = parser.parse_args()
    try:
        data = json.loads(args.matrix.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not {"matrix", "order"} <= data.keys():
            raise ValueError("input must contain matrix and order")
        result = select(data["matrix"], args.weight, data["order"])
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Selected conditions: " + ", ".join(result["retained"]))
    scores = result["final"]
    print(f"F={scores['F']:.6f}  R={scores['R']:.6f}  J={scores['J']:.6f}")
    if args.output:
        print(f"Selection trace: {args.output}")


if __name__ == "__main__":
    main()
