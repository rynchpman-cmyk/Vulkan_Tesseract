"""Run the complete resident 1D Carter flux/path eigensystem frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import (
    run_vulkan_carter_eigensystem_frontier,
)


def _resolutions(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth-resolutions", default="16,32,64")
    parser.add_argument("--shock-resolutions", default="16,32")
    parser.add_argument("--phase-resolutions", default="16,32")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_carter_eigensystem_results.json"),
    )
    args = parser.parse_args()
    report = run_vulkan_carter_eigensystem_frontier(
        _resolutions(args.smooth_resolutions),
        _resolutions(args.shock_resolutions),
        _resolutions(args.phase_resolutions),
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
