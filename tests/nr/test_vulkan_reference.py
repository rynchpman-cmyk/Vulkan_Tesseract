from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from tesseract_nr.vulkan_validation import (
    CpuMaterialReference,
    neural_phase_contact_primitives,
    smooth_theory33_primitives,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "vulkan_theory33_results.json"
FROZEN_RESULTS = ROOT / "vulkan_frozen_neural_results.json"
WENO_RESULTS = ROOT / "vulkan_weno_results.json"
CARTER_EIGEN_RESULTS = ROOT / "vulkan_carter_eigensystem_results.json"
FULL_CARTER_CCZ4_RESULTS = (
    ROOT / "vulkan_full_carter_ccz4_results.json"
)
MULTIDIMENSIONAL_RESULTS = (
    ROOT / "vulkan_multidimensional_results.json"
)


class VulkanReferenceContractTests(unittest.TestCase):
    def test_cpu_vertical_slice_preserves_number_balances(self) -> None:
        reference = CpuMaterialReference.initialize(
            smooth_theory33_primitives(16)
        )
        initial_baryon = np.sum(reference.state[:, 0])
        initial_carrier = np.sum(reference.state[:, 1])
        reference.step(1.0e-4)
        self.assertGreater(np.min(reference.state[:, 0]), 0.0)
        self.assertGreater(np.min(reference.state[:, 1]), 0.0)
        self.assertAlmostEqual(
            float(np.sum(reference.state[:, 0])),
            float(initial_baryon),
            places=13,
        )
        self.assertAlmostEqual(
            float(np.sum(reference.state[:, 1])),
            float(initial_carrier),
            places=13,
        )

    def test_tracked_gpu_resident_ladder_is_qualified(self) -> None:
        report = json.loads(RESULTS.read_text(encoding="utf-8"))
        self.assertTrue(report["qualified"])
        self.assertEqual(report["resolutions"], [16, 32, 64])
        for diagnostic in report["diagnostics"]:
            self.assertEqual(
                diagnostic["host_reads_after_evolution"], 0
            )
            self.assertEqual(diagnostic["recovery_failures"], 0)
            self.assertLess(
                diagnostic["physical_state_relative_error"], 2.0e-5
            )
            self.assertTrue(
                all(diagnostic["qualification"].values())
            )

    def test_cpu_frozen_neural_slice_preserves_number_balances(self) -> None:
        reference = CpuMaterialReference.initialize(
            neural_phase_contact_primitives(16),
            constitutive_model="qualified_frozen",
        )
        initial_baryon = np.sum(reference.state[:, 0])
        initial_carrier = np.sum(reference.state[:, 1])
        reference.step(1.0e-4)
        self.assertAlmostEqual(
            float(np.sum(reference.state[:, 0])),
            float(initial_baryon),
            places=13,
        )
        self.assertAlmostEqual(
            float(np.sum(reference.state[:, 1])),
            float(initial_carrier),
            places=13,
        )

    def test_tracked_frozen_neural_ladder_is_qualified(self) -> None:
        report = json.loads(FROZEN_RESULTS.read_text(encoding="utf-8"))
        self.assertTrue(report["qualified"])
        self.assertEqual(report["resolutions"], [16, 32, 64])
        self.assertEqual(
            report["constitutive_model"], "qualified_frozen"
        )
        self.assertTrue(
            report["qualification"]["phase_crossing_exercised"]
        )
        for diagnostic in report["diagnostics"]:
            self.assertEqual(
                diagnostic["host_reads_before_evolution"],
                diagnostic["host_reads_after_evolution"],
            )
            self.assertEqual(diagnostic["recovery_failures"], 0)
            self.assertEqual(
                diagnostic["phase_disagreements_with_cpu"], 0
            )
            self.assertLess(
                diagnostic["physical_state_relative_error"], 3.0e-5
            )

    def test_tracked_vulkan_weno_ladder_is_qualified(self) -> None:
        report = json.loads(WENO_RESULTS.read_text(encoding="utf-8"))
        self.assertTrue(report["qualified"])
        self.assertEqual(report["reconstruction"], "componentwise_weno5_z")
        self.assertGreater(
            min(report["interface_accuracy"]["observed_orders"]), 4.5
        )
        self.assertGreater(
            report["positivity_torture"]["gpu_minimum_theta"], 0.0
        )
        self.assertLess(
            report["positivity_torture"]["gpu_minimum_theta"], 1.0
        )
        self.assertTrue(
            report["qualification"]["phase_crossing_exercised"]
        )

    def test_tracked_vulkan_carter_eigensystem_is_qualified(self) -> None:
        report = json.loads(
            CARTER_EIGEN_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(
            report["reconstruction"],
            "branch-aware_carter_characteristic_weno5_z",
        )
        self.assertTrue(
            report["qualification"]["cpu_gpu_eigenvalues_match"]
        )
        self.assertTrue(
            report["qualification"][
                "branch_aware_face_selection_matches"
            ]
        )
        for item in report["symbol_audits"]:
            self.assertEqual(item["gpu_invalid_faces"], 0)
            self.assertLess(item["maximum_eigenvalue_error"], 1.0e-3)

    def test_tracked_full_carter_ccz4_layer_is_qualified(self) -> None:
        report = json.loads(
            FULL_CARTER_CCZ4_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertTrue(report["mixed_sources"]["qualified"])
        self.assertTrue(report["full_carter_block"]["qualified"])
        self.assertTrue(report["ccz4_geometry"]["qualified"])
        for item in report["full_carter_block"]["directional_audits"]:
            self.assertEqual(item["invalid_eigensystems"], 0)
            self.assertLess(
                item["maximum_eigenvalue_absolute_error"], 1.0e-3
            )

    def test_tracked_multidimensional_stack_is_qualified(self) -> None:
        report = json.loads(
            MULTIDIMENSIONAL_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(report["shape"], [8, 8])
        self.assertEqual(
            report["constitutive_model"], "qualified_frozen"
        )
        self.assertTrue(
            report["qualification"][
                "derivative_ccz4_matches_fd2_cpu"
            ]
        )
        self.assertTrue(
            report["qualification"][
                "curved_nine_variable_recovery_valid"
            ]
        )
        self.assertEqual(
            report["metrics"]["host_reads_during_evolution"], 0
        )
        self.assertEqual(
            report["metrics"]["invalid_final_recoveries"], 0
        )


if __name__ == "__main__":
    unittest.main()
