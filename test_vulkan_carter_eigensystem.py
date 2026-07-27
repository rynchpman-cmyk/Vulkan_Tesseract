#!/usr/bin/env python3
"""Live Vulkan regression for the numerical Carter characteristic basis."""

from __future__ import annotations

from tesseract_nr.vulkan_validation import (
    run_vulkan_carter_eigensystem_frontier,
)


def main() -> int:
    report = run_vulkan_carter_eigensystem_frontier(
        (16, 32), (16,), (16,)
    )
    if not report["qualified"]:
        failed = [
            name
            for name, value in report["qualification"].items()
            if not value
        ]
        raise RuntimeError(
            f"Vulkan Carter eigensystem qualification failed: {failed}"
        )
    maximum_error = max(
        item["maximum_eigenvalue_error"]
        for item in report["symbol_audits"]
    )
    maximum_residual = max(
        item["maximum_gpu_eigenpair_residual"]
        for item in report["symbol_audits"]
    )
    print("backend: Slang/Vulkan numerical Carter eigensystem")
    print(f"maximum CPU/GPU eigenvalue error: {maximum_error:.9g}")
    print(f"maximum eigenpair residual: {maximum_residual:.9g}")
    print("GPU Carter characteristic regression passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
