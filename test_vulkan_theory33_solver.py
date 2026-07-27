#!/usr/bin/env python3
"""Live Vulkan regression for the GPU-resident Theory 3.3 vertical slice."""

from __future__ import annotations

from tesseract_nr.vulkan_validation import (
    run_vulkan_theory33_frontier,
)


def main() -> int:
    report = run_vulkan_theory33_frontier(
        (16, 32), final_time=0.001, cfl=0.1
    )
    if not report["qualified"]:
        failed = [
            name
            for name, value in report["qualification"].items()
            if not value
        ]
        raise RuntimeError(
            f"Vulkan Theory 3.3 qualification failed: {failed}"
        )
    maximum_parity = max(
        item["physical_state_relative_error"]
        for item in report["diagnostics"]
    )
    maximum_recovery = max(
        item["maximum_recovery_residual"]
        for item in report["diagnostics"]
    )
    print("backend: Slang/Vulkan GPU-resident Theory 3.3")
    print("resolutions: 16, 32")
    print("host reads during evolution: 0")
    print(f"maximum CPU/GPU physical error: {maximum_parity:.9g}")
    print(f"maximum recovery residual: {maximum_recovery:.9g}")
    print("GPU-resident Theory 3.3 regression passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
