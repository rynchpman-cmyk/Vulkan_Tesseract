"""Qualify Vulkan mixed-vector stress and implicit carrier drag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import (
    qualify_vulkan_mixed_sources,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=12)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_mixed_source_results.json"),
    )
    args = parser.parse_args()
    report = qualify_vulkan_mixed_sources(
        points=args.points, seed=args.seed
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
