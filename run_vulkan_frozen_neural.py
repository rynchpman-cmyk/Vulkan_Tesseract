"""Run the GPU-resident qualified frozen-neural phase frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import (
    run_vulkan_frozen_neural_frontier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", default="16,32,64")
    parser.add_argument("--final-time", type=float, default=0.1)
    parser.add_argument("--cfl", type=float, default=0.12)
    parser.add_argument(
        "--reconstruction",
        choices=(
            "piecewise_constant",
            "weno5_z",
            "carter_characteristic_weno5_z",
            "carter_characteristic_roe",
        ),
        default="piecewise_constant",
    )
    parser.add_argument(
        "--positivity-preserving",
        action="store_true",
    )
    parser.add_argument(
        "--problem",
        choices=(
            "moving_phase_contact",
            "original_neural_phase_riemann",
        ),
        default="moving_phase_contact",
    )
    parser.add_argument(
        "--module",
        type=Path,
        default=Path("tesseract_nr_gpu.slang"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_frozen_neural_results.json"),
    )
    args = parser.parse_args()
    report = run_vulkan_frozen_neural_frontier(
        tuple(int(value) for value in args.resolutions.split(",")),
        final_time=args.final_time,
        cfl=args.cfl,
        reconstruction=args.reconstruction,
        positivity_preserving=args.positivity_preserving,
        problem=args.problem,
        module_path=args.module,
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
