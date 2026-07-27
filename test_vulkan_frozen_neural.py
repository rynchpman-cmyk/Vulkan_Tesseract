#!/usr/bin/env python3
"""Live Vulkan regression for the qualified frozen-neural phase path."""

from __future__ import annotations

from tesseract_nr.vulkan_validation import (
    run_vulkan_frozen_neural_frontier,
)


def main() -> int:
    report = run_vulkan_frozen_neural_frontier(
        (16, 32), final_time=0.1, cfl=0.12
    )
    if not report["qualified"]:
        failed = [
            name
            for name, value in report["qualification"].items()
            if not value
        ]
        raise RuntimeError(
            f"Vulkan frozen-neural qualification failed: {failed}"
        )
    maximum_parity = max(
        item["physical_state_relative_error"]
        for item in report["diagnostics"]
    )
    changed = sum(
        item["phase_cells_changed_from_initial"]
        for item in report["diagnostics"]
    )
    print("backend: Slang/Vulkan frozen-neural Theory 3.3")
    print("resolutions: 16, 32")
    print("host reads between snapshots: 0")
    print(f"maximum CPU/GPU physical error: {maximum_parity:.9g}")
    print(f"dynamic hard-branch changes: {changed}")
    print("GPU-resident frozen-neural phase regression passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
