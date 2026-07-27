#!/usr/bin/env python3
"""Live Vulkan regression for WENO5-Z and positivity preservation."""

from __future__ import annotations

from tesseract_nr.vulkan_validation import run_vulkan_weno_frontier


def main() -> int:
    report = run_vulkan_weno_frontier(
        (16, 32),
        (16,),
        phase_final_time=0.1,
    )
    if not report["qualified"]:
        failed = [
            name
            for name, value in report["qualification"].items()
            if not value
        ]
        raise RuntimeError(f"Vulkan WENO qualification failed: {failed}")
    torture = report["positivity_torture"]
    orders = report["interface_accuracy"]["observed_orders"]
    print("backend: Slang/Vulkan WENO5-Z + positivity")
    print(f"interface orders: {orders[0]:.3f}, {orders[1]:.3f}")
    print(f"torture minimum theta: {torture['gpu_minimum_theta']:.6f}")
    print("GPU WENO/positivity regression passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
