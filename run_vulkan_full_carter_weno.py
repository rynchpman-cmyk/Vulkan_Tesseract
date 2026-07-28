"""Qualify the resident full eleven-field Carter WENO5-Z path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import (
    qualify_vulkan_full_carter_weno,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", default="16,32,64")
    parser.add_argument("--final-time", type=float, default=0.005)
    parser.add_argument("--cfl", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_full_carter_weno_results.json"),
    )
    args = parser.parse_args()
    report = qualify_vulkan_full_carter_weno(
        tuple(int(value) for value in args.resolutions.split(",")),
        final_time=args.final_time,
        cfl=args.cfl,
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
