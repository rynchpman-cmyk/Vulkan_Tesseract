"""Generate the exhaustive Vulkan Tesseract capability-parity manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_parity import audit_vulkan_capability_parity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("../TESSERACT"),
        help="Optional original TESSERACT checkout for source identity audit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_parity_results.json"),
    )
    args = parser.parse_args()
    reference = (
        args.reference_root
        if args.reference_root.is_dir()
        else None
    )
    report = audit_vulkan_capability_parity(
        reference_root=reference
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
