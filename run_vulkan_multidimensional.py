"""Qualify the resident multidimensional Carter/CCZ4 Vulkan stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import (
    qualify_vulkan_multidimensional_full_stack,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_multidimensional_results.json"),
    )
    args = parser.parse_args()
    report = qualify_vulkan_multidimensional_full_stack(
        shape=(args.nx, args.ny),
        steps=args.steps,
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"qualified={report['qualified']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
