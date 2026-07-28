from __future__ import annotations

import unittest

import numpy as np

from tesseract_nr.ccz4 import CCZ4Solver
from tesseract_nr.grid import CartesianGrid
from tesseract_nr.vulkan_ccz4_vacuum import (
    VulkanCCZ4VacuumEvolutionBackend,
)
from tesseract_nr.vulkan_mots import (
    VulkanCCZ4Layout,
    brill_lindquist_puncture_ccz4,
    pack_ccz4_state,
    unpack_ccz4_state,
)


class VulkanMOTSLiveTests(unittest.TestCase):
    def test_ccz4_pack_roundtrip_and_production_gate(self) -> None:
        layout = VulkanCCZ4Layout((8, 9, 10), (2.0, 2.25, 2.5))
        grid = CartesianGrid(
            layout.shape,
            layout.lengths,
            origin=layout.origins,
            method="fd4",
        )
        state = CCZ4Solver(grid).flat_state()
        x, y, z = grid.coordinates()
        state.trace_K = x - 2.0 * y + 3.0 * z
        state.gamma_hat[1] = x * y
        packed = pack_ccz4_state(state, layout, dtype=np.float64)
        restored = unpack_ccz4_state(packed, layout, time=0.5)
        self.assertTrue(
            np.array_equal(restored.trace_K, state.trace_K)
        )
        self.assertTrue(
            np.array_equal(restored.gamma_hat, state.gamma_hat)
        )
        self.assertEqual(restored.time, 0.5)

        production = VulkanCCZ4Layout(
            (120, 120, 120), (3.0, 3.0, 3.0)
        ).diagnostics()
        self.assertAlmostEqual(
            production["reference_cells_per_radius"], 16.72
        )
        self.assertTrue(production["reference_resolution_gate_passed"])

    def test_resident_vacuum_step_has_no_hidden_readback(self) -> None:
        layout = VulkanCCZ4Layout((8, 8, 8), (2.0, 2.0, 2.0))
        backend = VulkanCCZ4VacuumEvolutionBackend(
            layout.shape,
            lengths=layout.lengths,
            origins=layout.origins,
        )
        initial = brill_lindquist_puncture_ccz4(layout)
        state = backend.initialize(initial)
        self.assertEqual(state.host_reads, 0)
        dt = 1.0e-4
        backend.step(state, dt)
        self.assertEqual(state.host_reads, 0)
        snapshot = backend.snapshot(state)
        self.assertEqual(state.host_reads, 1)
        self.assertEqual(snapshot.steps, 1)
        self.assertTrue(np.all(np.isfinite(snapshot.ccz4_state)))
        self.assertGreater(float(np.min(snapshot.ccz4_state[:, 18])), 0.0)
        self.assertGreater(float(np.min(snapshot.ccz4_state[:, 24])), 0.0)
        self.assertLess(
            float(np.max(snapshot.projection_audit[:, :2])), 1.0e-4
        )
        self.assertEqual(
            int(np.count_nonzero(snapshot.projection_audit[:, 2] < 0.5)),
            0,
        )

        grid = CartesianGrid(
            layout.shape,
            layout.lengths,
            origin=layout.origins,
            method="fd4",
        )
        cpu = CCZ4Solver(grid)
        cpu_after = cpu.step(unpack_ccz4_state(initial, layout), dt)
        gpu_after = unpack_ccz4_state(
            snapshot.ccz4_state, layout, time=snapshot.time
        )
        maximum_error = max(
            float(
                np.max(
                    np.abs(
                        getattr(gpu_after, name)
                        - getattr(cpu_after, name)
                    )
                )
            )
            for name in (
                "conformal_metric",
                "conformal_A",
                "conformal_factor",
                "trace_K",
                "theta",
                "gamma_hat",
                "lapse",
                "shift",
                "shift_driver",
            )
        )
        self.assertLess(maximum_error, 3.0e-4)


if __name__ == "__main__":
    unittest.main()
