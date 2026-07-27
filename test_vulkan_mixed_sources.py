#!/usr/bin/env python3
"""Live Vulkan regression for mixed stress and implicit drag."""

from __future__ import annotations

from tesseract_nr.vulkan_validation import (
    qualify_vulkan_mixed_sources,
)


def main() -> int:
    report = qualify_vulkan_mixed_sources(points=8)
    if not report["qualified"]:
        failed = [
            name
            for name, value in report["qualification"].items()
            if not value
        ]
        raise RuntimeError(
            f"Vulkan mixed-source qualification failed: {failed}"
        )
    maximum_vector_error = max(
        item["maximum_vector_stress_absolute_error"]
        for item in report["mode_audits"]
    )
    maximum_drag_error = max(
        item["maximum_drag_primitive_absolute_error"]
        for item in report["mode_audits"]
    )
    print("backend: Slang/Vulkan full-vector local sources")
    print(f"maximum mixed-stress error: {maximum_vector_error:.9g}")
    print(f"maximum implicit-drag error: {maximum_drag_error:.9g}")
    print("GPU mixed-source regression passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
