"""Run the original learned-phase Riemann problem on the full Vulkan ABI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import (
    run_vulkan_original_neural_phase_riemann_frontier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", default="16,32,64,128")
    parser.add_argument("--final-time", type=float, default=0.02)
    parser.add_argument("--cfl", type=float, default=0.12)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("tesseract_neural_phase_riemann_results.json"),
    )
    parser.add_argument(
        "--module",
        type=Path,
        default=Path("tesseract_nr_gpu.slang"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "vulkan_original_neural_phase_full_roe_results.json"
        ),
    )
    args = parser.parse_args()
    report = run_vulkan_original_neural_phase_riemann_frontier(
        tuple(int(value) for value in args.resolutions.split(",")),
        final_time=args.final_time,
        cfl=args.cfl,
        reference_path=args.reference,
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
