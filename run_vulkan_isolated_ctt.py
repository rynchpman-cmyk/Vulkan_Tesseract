"""Qualify the resident fp64 isolated CTT constraint solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import qualify_vulkan_isolated_ctt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="10,10,10")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_isolated_ctt_results.json"),
    )
    args = parser.parse_args()
    shape = tuple(int(value) for value in args.shape.split(","))
    if len(shape) != 3:
        raise ValueError("shape must contain three comma-separated values")
    report = qualify_vulkan_isolated_ctt(shape=shape)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["qualification"], indent=2))
    print(f"qualified={report['qualified']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
