"""Qualify resident constrained active-neural spacetime evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import (
    run_vulkan_constrained_convergence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", default="8,16,32")
    parser.add_argument("--final-time", type=float, default=1.0e-6)
    parser.add_argument("--picard-iterations", type=int, default=3)
    parser.add_argument("--elliptic-tolerance", type=float, default=2.0e-8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_constrained_results.json"),
    )
    args = parser.parse_args()
    report = run_vulkan_constrained_convergence(
        tuple(int(value) for value in args.resolutions.split(",")),
        final_time=args.final_time,
        picard_iterations=args.picard_iterations,
        elliptic_tolerance=args.elliptic_tolerance,
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
