#!/usr/bin/env python3
"""Live regression for the full Carter/CCZ4 Vulkan promotion."""

from __future__ import annotations

from tesseract_nr.vulkan_validation import (
    qualify_vulkan_ccz4_geometry,
    qualify_vulkan_full_carter_block,
    qualify_vulkan_mixed_sources,
)


def main() -> int:
    reports = {
        "mixed_sources": qualify_vulkan_mixed_sources(points=6),
        "full_carter_block": qualify_vulkan_full_carter_block(points=4),
        "ccz4_geometry": qualify_vulkan_ccz4_geometry(points=4),
    }
    failed = [
        name for name, report in reports.items() if not report["qualified"]
    ]
    if failed:
        raise RuntimeError(f"full Carter/CCZ4 Vulkan failures: {failed}")
    print("mixed vector stress and implicit drag: qualified")
    print("curved multidimensional nine-field eigensystem: qualified")
    print("CCZ4 projection and matter coupling: qualified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
