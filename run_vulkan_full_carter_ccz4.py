"""Qualify full-vector sources, nine-field Carter geometry, and CCZ4 coupling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import (
    qualify_vulkan_ccz4_geometry,
    qualify_vulkan_full_carter_block,
    qualify_vulkan_mixed_sources,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-points", type=int, default=12)
    parser.add_argument("--carter-points", type=int, default=6)
    parser.add_argument("--ccz4-points", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_full_carter_ccz4_results.json"),
    )
    args = parser.parse_args()
    report = {
        "schema": "vulkan-tesseract.full-carter-ccz4.v1",
        "mixed_sources": qualify_vulkan_mixed_sources(
            points=args.source_points
        ),
        "full_carter_block": qualify_vulkan_full_carter_block(
            points=args.carter_points
        ),
        "ccz4_geometry": qualify_vulkan_ccz4_geometry(
            points=args.ccz4_points
        ),
    }
    report["qualified"] = all(
        report[name]["qualified"]
        for name in (
            "mixed_sources",
            "full_carter_block",
            "ccz4_geometry",
        )
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in ("mixed_sources", "full_carter_block", "ccz4_geometry"):
        print(f"{name}: qualified={report[name]['qualified']}")
    print(f"qualified={report['qualified']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
