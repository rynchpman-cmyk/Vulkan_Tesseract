from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from tesseract_nr.vulkan_validation import (
    CpuMaterialReference,
    neural_phase_contact_primitives,
    original_neural_phase_riemann_primitives,
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
VULKAN_CARTER_RIEMANN_RESULTS = (
    ROOT / "vulkan_carter_riemann_results.json"
)
FULL_CARTER_WENO_RESULTS = (
    ROOT / "vulkan_full_carter_weno_results.json"
)
MIXED_PROCA_RESULTS = ROOT / "vulkan_mixed_proca_results.json"
CONSTRAINED_RESULTS = ROOT / "vulkan_constrained_results.json"
ORIGINAL_NEURAL_PHASE_ROE_RESULTS = (
    ROOT / "vulkan_original_neural_phase_roe_results.json"
)
ORIGINAL_NEURAL_PHASE_FULL_ROE_RESULTS = (
    ROOT / "vulkan_original_neural_phase_full_roe_results.json"
)
FRONTIER_RESULTS = ROOT / "vulkan_frontier_results.json"
CCZ4_FD4_RESULTS = ROOT / "vulkan_ccz4_fd4_results.json"
AMR_RESULTS = ROOT / "vulkan_amr_results.json"
BOUNDARY_RESULTS = ROOT / "vulkan_boundary_results.json"
TOV_RESULTS = ROOT / "vulkan_tov_results.json"
ISOLATED_CTT_RESULTS = ROOT / "vulkan_isolated_ctt_results.json"
PARITY_RESULTS = ROOT / "vulkan_parity_results.json"
ORIGINAL_WORKFLOWS_RESULTS = (
    ROOT / "vulkan_original_workflows_results.json"
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

    def test_original_neural_phase_riemann_initial_state(self) -> None:
        primitives = original_neural_phase_riemann_primitives(16)
        self.assertTrue(np.all(primitives[:, 0] == 0.10))
        self.assertTrue(np.all(primitives[:, 1] == 0.12))
        self.assertTrue(np.all(primitives[:, 2] == 0.0))
        self.assertTrue(np.all(primitives[:, 3] == 0.005))
        np.testing.assert_array_equal(
            primitives[:, 4],
            np.r_[np.full(8, 0.112), np.full(8, 0.103)],
        )

    def test_cpu_characteristic_roe_preserves_number_balances(self) -> None:
        reference = CpuMaterialReference.initialize(
            original_neural_phase_riemann_primitives(16),
            constitutive_model="qualified_frozen",
            reconstruction="carter_characteristic_roe",
            positivity_preserving=True,
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

    def test_tracked_original_phase_roe_gpu_ladder_is_qualified(
        self,
    ) -> None:
        report = json.loads(
            ORIGINAL_NEURAL_PHASE_ROE_RESULTS.read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(
            report["problem"], "original_neural_phase_riemann"
        )
        self.assertEqual(
            report["reconstruction"], "carter_characteristic_roe"
        )
        self.assertEqual(report["resolutions"], [16, 32, 64, 128])
        self.assertEqual(
            [
                item["phase_cells_changed_from_initial"]
                for item in report["diagnostics"]
            ],
            [0, 0, 0, 3],
        )
        for diagnostic in report["diagnostics"]:
            self.assertEqual(diagnostic["recovery_failures"], 0)
            self.assertEqual(
                diagnostic["phase_disagreements_with_cpu"], 0
            )
            self.assertLess(
                diagnostic["physical_state_relative_error"], 3.0e-5
            )

    def test_tracked_original_full_carter_roe_ladder_is_qualified(
        self,
    ) -> None:
        report = json.loads(
            ORIGINAL_NEURAL_PHASE_FULL_ROE_RESULTS.read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(
            report["problem"], "original_neural_phase_riemann"
        )
        self.assertEqual(report["reconstruction"], "full_carter_roe")
        self.assertEqual(
            report["physical_block"],
            "nine Carter fields plus entropy and tracer",
        )
        self.assertEqual(report["resolutions"], [16, 32, 64, 128])
        self.assertEqual(
            report["phase_changes_from_initial_count"],
            report["reference_phase_changes"],
        )
        self.assertEqual(
            report["phase_changes_from_initial_count"],
            [0, 0, 0, 4],
        )
        self.assertTrue(all(report["qualification"].values()))
        for diagnostic in report["diagnostics"]:
            self.assertEqual(diagnostic["recovery_failures"], 0)
            self.assertEqual(diagnostic["invalid_eigensystems"], 0)
            self.assertEqual(
                diagnostic["host_reads_before_evolution"],
                diagnostic["host_reads_after_evolution"],
            )

    def test_tracked_analytic_full_carter_shocks_are_qualified(
        self,
    ) -> None:
        report = json.loads(
            VULKAN_CARTER_RIEMANN_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(
            report["reference_schema"],
            "tesseract.full-carter-riemann-frontier.v1",
        )
        self.assertTrue(all(report["qualification"].values()))
        for name in ("counterflow_shock", "colliding_streams"):
            ladder = report[name]
            self.assertEqual(ladder["resolutions"], [16, 32, 64, 128])
            self.assertTrue(ladder["qualified"])
            for diagnostic in ladder["diagnostics"]:
                self.assertEqual(diagnostic["recovery_failures"], 0)
                self.assertEqual(diagnostic["invalid_eigensystems"], 0)

    def test_tracked_full_carter_weno_is_qualified(self) -> None:
        report = json.loads(
            FULL_CARTER_WENO_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(
            report["reconstruction"], "full_carter_weno5_z"
        )
        self.assertEqual(report["physical_fields"], 9)
        self.assertEqual(report["extended_fields"], ["entropy", "tracer"])
        self.assertTrue(all(report["qualification"].values()))

    def test_tracked_mixed_proca_evolution_is_qualified(self) -> None:
        report = json.loads(
            MIXED_PROCA_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertTrue(report["subsystem"]["qualified"])
        self.assertTrue(report["coupled_stack"]["qualified"])
        self.assertTrue(all(report["qualification"].values()))
        self.assertEqual(
            report["subsystem"]["host_reads_during_evolution"], 0
        )
        self.assertEqual(
            report["coupled_stack"]["host_reads_during_evolution"], 0
        )

    def test_tracked_resident_constrained_ladder_is_qualified(
        self,
    ) -> None:
        report = json.loads(
            CONSTRAINED_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(
            report["schema"],
            "vulkan-tesseract.constrained-convergence.v2",
        )
        self.assertEqual(report["resolutions"], [8, 16, 32])
        self.assertTrue(all(report["qualification"].values()))
        for model in ("analytic_control", "qualified_frozen"):
            campaign = report[model]
            self.assertTrue(campaign["qualified"])
            self.assertGreater(
                campaign["convergence"]["initial_constraints"][
                    "hamiltonian"
                ]["medium_fine_order"],
                1.8,
            )
            self.assertGreater(
                campaign["convergence"]["final_constraints"][
                    "momentum"
                ]["medium_fine_order"],
                1.8,
            )
            self.assertTrue(
                campaign["qualification"][
                    "micro_horizon_energy_balance_is_controlled"
                ]
            )
            for level in campaign["levels"]:
                self.assertTrue(
                    level["resident_elliptic_report"]["converged"]
                )
                self.assertEqual(
                    level["host_reads_during_evolution"], 0
                )
                self.assertEqual(level["recovery_failures"], 0)
        self.assertGreater(
            min(
                level["phase_two_fraction"]
                for level in report["qualified_frozen"]["levels"]
            ),
            0.99,
        )

    def test_resident_cmc_entrypoints_are_present(self) -> None:
        main_shader = (ROOT / "tesseract_nr_gpu.slang").read_text(
            encoding="utf-8"
        )
        fp64_shader = (ROOT / "tesseract_cmc_fp64.slang").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "void theory33BalancedCounterflowKernel(", main_shader
        )
        self.assertIn("void theory33CMCUpdateKernel(", main_shader)
        self.assertIn(
            "void theory33CMCTrace64Kernel(", fp64_shader
        )
        self.assertIn(
            "void theory33CMCPack64Kernel(", fp64_shader
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

    def test_tracked_full_frontier_is_qualified(self) -> None:
        report = json.loads(
            FRONTIER_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertTrue(all(report["qualification"].values()))
        self.assertEqual(
            report["long_horizon"]["analytic_control"]["steps"], 128
        )
        self.assertEqual(
            report["long_horizon"]["qualified_frozen"]["steps"], 128
        )
        self.assertTrue(
            report["two_dimensional_convergence"]["qualified"]
        )
        self.assertTrue(report["three_dimensional_smoke"]["qualified"])

    def test_tracked_fd4_ccz4_is_qualified(self) -> None:
        report = json.loads(
            CCZ4_FD4_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(report["ccz4_derivative_order"], 4)
        self.assertTrue(all(report["qualification"].values()))

    def test_tracked_amr_primitives_are_qualified(self) -> None:
        report = json.loads(AMR_RESULTS.read_text(encoding="utf-8"))
        self.assertTrue(report["qualified"])
        self.assertTrue(all(report["qualification"].values()))
        self.assertEqual(
            [item["dimensions"] for item in report["transfer"]],
            [1, 2, 3],
        )

    def test_tracked_physical_boundaries_are_qualified(self) -> None:
        report = json.loads(
            BOUNDARY_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertTrue(all(report["qualification"].values()))
        self.assertEqual(report["host_reads_during_evolution"], 0)
        self.assertEqual(report["invalid_recoveries"], 0)
        self.assertEqual(
            report["fourth_order_full_stack_host_reads"], 0
        )
        self.assertEqual(
            report["fourth_order_full_stack_invalid_recoveries"], 0
        )
        self.assertTrue(
            report["qualification"][
                "fourth_order_ccz4_boundary_matches_cpu"
            ]
        )
        self.assertTrue(
            report["qualification"][
                "fourth_order_proca_boundary_is_gauss_compatible"
            ]
        )
        self.assertEqual(
            [
                (
                    item["dimensions"],
                    item["derivative_order"],
                    item["qualified"],
                )
                for item in report["multidimensional_boundary_audits"]
            ],
            [
                (2, 2, True),
                (2, 4, True),
                (3, 2, True),
                (3, 4, True),
            ],
        )

    def test_tracked_tov_strong_field_path_is_qualified(self) -> None:
        report = json.loads(TOV_RESULTS.read_text(encoding="utf-8"))
        self.assertTrue(report["qualified"])
        self.assertTrue(all(report["qualification"].values()))
        self.assertEqual(report["host_reads_during_evolution"], 0)
        self.assertEqual(report["invalid_recoveries"], 0)
        self.assertEqual(
            report["checkpoint_vector_roundtrip_error"], 0.0
        )

    def test_tracked_isolated_ctt_is_qualified(self) -> None:
        report = json.loads(
            ISOLATED_CTT_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertTrue(all(report["qualification"].values()))
        self.assertLess(
            report["cpu_parity"][
                "conformal_factor_maximum_absolute_error"
            ],
            1.0e-9,
        )

    def test_exhaustive_capability_manifest_is_qualified(self) -> None:
        report = json.loads(
            PARITY_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertTrue(all(report["qualification"].values()))
        from tesseract_nr.vulkan_parity import EXPECTED_REFERENCE_DIVERGENCES

        reference = report["reference_source_audit"]
        self.assertTrue(reference["checked"])
        self.assertFalse(reference["missing_files"])
        self.assertFalse(reference["unexpected_content_mismatches"])
        self.assertEqual(
            reference["expected_divergences"],
            {name: EXPECTED_REFERENCE_DIVERGENCES[name] for name in reference["content_mismatches"]},
        )
        self.assertTrue(
            all(row["qualified"] for row in report["capabilities"])
        )
        self.assertTrue(report["implementation_sha256"])
        for relative, expected in report[
            "implementation_sha256"
        ].items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                expected,
            )

    def test_retained_original_vulkan_workflows_are_live(self) -> None:
        report = json.loads(
            ORIGINAL_WORKFLOWS_RESULTS.read_text(encoding="utf-8")
        )
        self.assertTrue(report["qualified"])
        self.assertTrue(all(report["qualification"].values()))
        self.assertTrue(
            all(
                check["passed"]
                for check in report["live_checks"].values()
            )
        )
        self.assertGreaterEqual(
            report["conditioned_literal_audit"]["steps"], 128
        )


if __name__ == "__main__":
    unittest.main()
