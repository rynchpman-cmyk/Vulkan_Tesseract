from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tesseract_nr.ccz4 import CCZ4Solver
from tesseract_nr.grid import CartesianGrid
from tesseract_nr.vulkan_mots import (
    VulkanCCZ4Layout,
    VulkanPunctureMOTSObserver,
    VulkanPunctureMOTSParameters,
    brill_lindquist_puncture_ccz4,
    load_vulkan_ccz4_checkpoint,
    pack_ccz4_state,
    puncture_centers,
    unpack_ccz4_state,
)


class VulkanMOTSBridgeTests(unittest.TestCase):
    def test_x_fastest_ccz4_pack_roundtrip(self) -> None:
        layout = VulkanCCZ4Layout(
            (8, 9, 10), (2.0, 2.25, 2.5), derivative_order=4
        )
        grid = CartesianGrid(
            layout.shape,
            layout.lengths,
            origin=layout.origins,
            method="fd4",
        )
        state = CCZ4Solver(grid).flat_state()
        x, y, z = grid.coordinates()
        state.trace_K = 0.1 * x - 0.2 * y + 0.3 * z
        state.theta = x * y * z
        state.gamma_hat[2] = np.sin(z)
        state.shift[0] = np.cos(x)
        state.shift_driver[1] = y * y
        packed = pack_ccz4_state(state, layout, dtype=np.float64)
        restored = unpack_ccz4_state(packed, layout, time=0.75)
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
        ):
            self.assertTrue(
                np.array_equal(
                    getattr(restored, name), getattr(state, name)
                )
            )
        self.assertEqual(restored.time, 0.75)

    def test_production_layout_crosses_reference_sixteen_cell_gate(self) -> None:
        layout = VulkanCCZ4Layout(
            (120, 120, 120), (3.0, 3.0, 3.0)
        )
        report = layout.diagnostics(reference_radius=0.418)
        self.assertEqual(layout.spacing, (0.025, 0.025, 0.025))
        self.assertAlmostEqual(
            report["reference_cells_per_radius"], 16.72
        )
        self.assertTrue(report["reference_resolution_gate_passed"])

    def test_direct_puncture_initial_data_has_correct_ccz4_and_lapse(self) -> None:
        layout = VulkanCCZ4Layout(
            (16, 16, 16), (2.0, 2.0, 2.0)
        )
        packed = brill_lindquist_puncture_ccz4(layout)
        state = unpack_ccz4_state(packed, layout)
        for axis in range(3):
            self.assertTrue(
                np.all(state.conformal_metric[axis, axis] == 1.0)
            )
        self.assertTrue(
            np.array_equal(state.lapse, state.conformal_factor)
        )
        self.assertGreater(float(np.min(state.conformal_factor)), 0.0)
        self.assertLess(float(np.min(state.conformal_factor)), 1.0)
        grid = CartesianGrid(
            layout.shape,
            layout.lengths,
            origin=layout.origins,
            method="fd4",
        )
        lower, upper = puncture_centers(grid, state)
        self.assertAlmostEqual(lower, -0.39, delta=layout.spacing[2])
        self.assertAlmostEqual(upper, 0.39, delta=layout.spacing[2])

    def test_checkpoint_slice_and_observer_continuation_are_json_safe(self) -> None:
        layout = VulkanCCZ4Layout(
            (8, 8, 8), (2.0, 2.0, 2.0), derivative_order=4
        )
        packed = brill_lindquist_puncture_ccz4(layout)
        parameters = VulkanPunctureMOTSParameters(
            maximum_degree=2,
            individual_maximum_degree=1,
            maximum_iterations=1,
            track_individual=False,
            compute_constraints=False,
        )
        observer = VulkanPunctureMOTSObserver(layout, parameters)
        payload = json.loads(json.dumps(observer.checkpoint()))
        restarted = VulkanPunctureMOTSObserver(layout, parameters)
        restarted.restore(payload)
        self.assertEqual(restarted.checkpoint(), payload)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.npz"
            np.savez(
                path,
                schema=np.array(
                    "vulkan-tesseract.ccz4-observer-snapshot.v1"
                ),
                shape=np.asarray(layout.shape),
                lengths=np.asarray(layout.lengths),
                origins=np.asarray(layout.origins),
                derivative_order=np.array(layout.derivative_order),
                ccz4_state=packed,
                time=np.array(0.25),
                steps=np.array(10),
            )
            snapshot = load_vulkan_ccz4_checkpoint(path)
        self.assertEqual(snapshot.layout, layout)
        self.assertEqual(snapshot.time, 0.25)
        self.assertEqual(snapshot.steps, 10)
        self.assertTrue(np.array_equal(snapshot.ccz4_state, packed))


if __name__ == "__main__":
    unittest.main()
