"""Qualify fp64 Vulkan TOV construction and strong-field evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import (
    qualify_vulkan_tov_strong_field,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_tov_results.json"),
    )
    args = parser.parse_args()
    report = qualify_vulkan_tov_strong_field(
        points=args.points,
        steps=args.steps,
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
