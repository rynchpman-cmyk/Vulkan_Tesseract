"""Run the complete resident Vulkan NR frontier qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_frontier import run_vulkan_frontier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("theory33_frozen_variants.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_frontier_results.json"),
    )
    args = parser.parse_args()
    report = run_vulkan_frontier(artifact_path=args.artifact)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["qualification"], indent=2))
    print(f"qualified={report['qualified']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
