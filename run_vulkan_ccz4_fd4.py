"""Qualify the complete resident fourth-order CCZ4/full-stack path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import (
    qualify_vulkan_multidimensional_full_stack,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="8,8")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_ccz4_fd4_results.json"),
    )
    args = parser.parse_args()
    shape = tuple(int(value) for value in args.shape.split(","))
    if len(shape) != 2:
        raise ValueError("shape must contain two comma-separated values")
    report = qualify_vulkan_multidimensional_full_stack(
        shape=shape,
        steps=args.steps,
        derivative_order=4,
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["qualification"], indent=2))
    print(f"qualified={report['qualified']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
