"""Qualify resident Vulkan physical and characteristic boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import qualify_vulkan_boundaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=48)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_boundary_results.json"),
    )
    args = parser.parse_args()
    report = qualify_vulkan_boundaries(
        points=args.points,
        seed=args.seed,
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
