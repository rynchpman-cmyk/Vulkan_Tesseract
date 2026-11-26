#!/usr/bin/env python3
"""Live regression for the full Carter/CCZ4 Vulkan promotion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tesseract_nr.vulkan_backend import VulkanTheory33MultidimensionalBackend
from tesseract_nr.vulkan_validation import (
    qualify_vulkan_ccz4_geometry,
    qualify_vulkan_full_carter_block,
    qualify_vulkan_mixed_sources,
)


def check_transverse_carrier_path(module_path: Path | None = None) -> None:
    """Pure transverse carrier flow has a zero normal momentum RHS."""
    points = 8
    coordinate = 2.0 * np.pi * np.arange(points) / points
    primitive = np.zeros((points, 9), dtype=np.float32)
    primitive[:, 0] = np.log(0.6)
    primitive[:, 1] = -1.5
    primitive[:, 5] = np.log(0.05)
    primitive[:, 7] = 0.31 + 0.15 * np.sin(coordinate)
    primitive[:, 8] = 0.17 + 0.10 * np.cos(coordinate)
    geometry = np.tile(np.r_[np.eye(3).ravel(), 1.0, np.zeros(3)], (points, 1))
    for boundary in ("periodic", "radiative_outflow"):
        backend = VulkanTheory33MultidimensionalBackend(
            (points,), boundary_mode=boundary, module_path=module_path,
        )
        state = backend.initialize(
            primitive, np.zeros((points, 20)), geometry, np.zeros(points),
        )
        backend._rhs_dispatch(state.tensor, state.primitive)
        normal_momentum_rhs = backend._rhs.to_numpy()[:, 6]
        np.testing.assert_allclose(
            normal_momentum_rhs, 0.0, atol=3e-6, rtol=0.0, err_msg=boundary,
        )


def main() -> int:
    check_transverse_carrier_path()
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
