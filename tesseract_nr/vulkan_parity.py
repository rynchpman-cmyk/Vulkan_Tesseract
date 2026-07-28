"""Aggregate, machine-readable original-to-Vulkan capability audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

IMPLEMENTATION_PATHS = (
    "run_vulkan_original_workflows.py",
    "run_vulkan_parity.py",
    "tesseract_nr_gpu.slang",
    "tesseract_cmc_fp64.slang",
    "tesseract_ctt_fp64.slang",
    "tesseract_tov_fp64.slang",
    "theory33_qualified_frozen.slang",
    "theory33_frozen_variants.json",
    "tesseract_nr/vulkan_backend.py",
    "tesseract_nr/vulkan_boundary.py",
    "tesseract_nr/vulkan_amr.py",
    "tesseract_nr/vulkan_ctt.py",
    "tesseract_nr/vulkan_tov.py",
    "tesseract_nr/vulkan_frontier.py",
    "tesseract_nr/vulkan_parity.py",
    "tesseract_nr/vulkan_validation.py",
)


CAPABILITIES = (
    {
        "id": "literal_carter_kkt_recurrence_and_ad",
        "mode": "retained_original_vulkan",
        "paths": (
            "carter_tesseract_kkt.comp",
            "carter_master_ad.slang",
            "carter_tesseract_full.slang",
            "carter_tesseract_conditioned.slang",
            "run_full_slang.ps1",
            "run_slang_ad.ps1",
            "run_conditioned_literal.ps1",
            "run_progressive_full.ps1",
            "hybrid_tesseract.py",
            "progressive_full_unroll.py",
            "migrate_full_slang.py",
            "hybrid_model_conditioned.json",
        ),
        "artifacts": ("vulkan_original_workflows_results.json",),
    },
    {
        "id": "theory33_physical_hybrid_and_native_reverse",
        "mode": "retained_original_vulkan",
        "paths": (
            "theory33_hybrid.py",
            "theory33_hybrid.slang",
            "run_theory33_hybrid.ps1",
            "run_theory33_slang.ps1",
            "run_theory33_frozen_slang.ps1",
        ),
        "artifacts": ("vulkan_original_workflows_results.json",),
    },
    {
        "id": "theory33_training_replay_and_event_bptt",
        "mode": "retained_optional_training",
        "paths": (
            "theory33_advanced.py",
            "theory33_experiments.py",
            "run_theory33_advanced.py",
            "run_theory33_experiments.py",
            "generate_theory33_kinetic_data.py",
            "export_theory33_frozen_slang.py",
            "train_hybrid.py",
            "train_lyapunov_conditioned.py",
            "theory33_frozen_variants.json",
            "data/theory33_kinetic_metadata.json",
            "data/theory33_kinetic_transport.csv",
            "data/theory33_noisy_phase_trajectories.csv",
            "theory33_advanced_results.json",
            "theory33_experiment_results.json",
        ),
        "artifacts": (
            "theory33_advanced_results.json",
            "theory33_experiment_results.json",
        ),
    },
    {
        "id": "resident_theory33_analytic_and_frozen_material",
        "mode": "resident_vulkan",
        "artifacts": (
            "vulkan_theory33_results.json",
            "vulkan_frozen_neural_results.json",
        ),
        "paths": (
            "run_vulkan_theory33.ps1",
            "run_vulkan_frozen_neural.ps1",
        ),
    },
    {
        "id": "resident_positivity_weno_and_complete_eigensystems",
        "mode": "resident_vulkan",
        "artifacts": (
            "vulkan_weno_results.json",
            "vulkan_carter_eigensystem_results.json",
            "vulkan_full_carter_weno_results.json",
        ),
    },
    {
        "id": "resident_full_carter_roe_dlm_and_riemann_ladders",
        "mode": "resident_vulkan",
        "artifacts": (
            "vulkan_original_neural_phase_full_roe_results.json",
            "vulkan_carter_riemann_results.json",
        ),
    },
    {
        "id": "resident_mixed_vector_drag_and_dynamic_proca",
        "mode": "resident_vulkan",
        "artifacts": (
            "vulkan_mixed_source_results.json",
            "vulkan_mixed_proca_results.json",
        ),
    },
    {
        "id": "resident_curved_multidimensional_carter_ccz4",
        "mode": "resident_vulkan",
        "artifacts": (
            "vulkan_full_carter_ccz4_results.json",
            "vulkan_multidimensional_results.json",
            "vulkan_ccz4_fd4_results.json",
        ),
    },
    {
        "id": "resident_constrained_cmc_and_continuum",
        "mode": "resident_vulkan",
        "artifacts": ("vulkan_constrained_results.json",),
    },
    {
        "id": "resident_128_step_2d_and_3d_frontier",
        "mode": "resident_vulkan",
        "artifacts": ("vulkan_frontier_results.json",),
    },
    {
        "id": "resident_physical_boundaries",
        "mode": "resident_vulkan",
        "artifacts": ("vulkan_boundary_results.json",),
        "paths": (
            "tesseract_nr/vulkan_boundary.py",
            "run_vulkan_boundaries.ps1",
        ),
    },
    {
        "id": "resident_amr_transfer_reflux_and_gauss_projection",
        "mode": "resident_vulkan",
        "artifacts": ("vulkan_amr_results.json",),
        "paths": (
            "tesseract_nr/vulkan_amr.py",
            "run_vulkan_amr.ps1",
        ),
    },
    {
        "id": "resident_tov_strong_field_and_complete_restart",
        "mode": "resident_vulkan",
        "artifacts": ("vulkan_tov_results.json",),
        "paths": (
            "tesseract_tov_fp64.slang",
            "tesseract_nr/vulkan_tov.py",
            "run_vulkan_tov.ps1",
        ),
    },
    {
        "id": "resident_isolated_ctt_constraints",
        "mode": "resident_vulkan",
        "artifacts": ("vulkan_isolated_ctt_results.json",),
        "paths": (
            "tesseract_ctt_fp64.slang",
            "tesseract_nr/vulkan_ctt.py",
            "run_vulkan_isolated_ctt.ps1",
        ),
    },
    {
        "id": "python_reference_engine_diagnostics_io_and_visualization",
        "mode": "retained_cpu_oracle_and_postprocessing",
        "paths": (
            "run_tesseract_nr.py",
            "run_tesseract_nr_constrained.py",
            "run_tesseract_nr_convergence.py",
            "run_tesseract_nr_frontier.py",
            "run_tesseract_nr_shocks.py",
            "tesseract_nr/backend.py",
            "tesseract_nr/kkt_horizon.py",
            "tesseract_nr/amr.py",
            "tesseract_nr/boundary.py",
            "tesseract_nr/tov.py",
            "tesseract_nr/io.py",
            "tesseract_nr/nr_diagnostics.py",
            "tesseract_full_stack_visualizer.py",
            "visualize_tesseract.py",
            "run_tesseract.py",
            "run.ps1",
            "run_visualization.ps1",
        ),
        "artifacts": (
            "tesseract_nr_constrained_results.json",
            "tesseract_nr_convergence_results.json",
            "tesseract_nr_frontier_results.json",
            "tesseract_nr_shock_results.json",
        ),
    },
)


def _artifact_passes(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid: {exc}"
    if "qualified" in payload:
        return bool(payload["qualified"]), "qualified"
    if "passed" in payload:
        return bool(payload["passed"]), "passed"
    return False, "no qualification field"


def _reference_files(reference_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in reference_root.glob("*"):
        if path.is_file() and path.suffix in {".py", ".slang", ".comp"}:
            files.append(path)
    for directory in ("tesseract_nr",):
        files.extend(
            path
            for path in (reference_root / directory).rglob("*.py")
            if path.is_file()
        )
    data_root = reference_root / "data"
    if data_root.is_dir():
        files.extend(
            path
            for path in data_root.rglob("*")
            if path.is_file() and path.suffix in {".csv", ".json"}
        )
    return sorted(files)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_vulkan_capability_parity(
    *,
    root: str | Path = ROOT,
    reference_root: str | Path | None = None,
) -> dict[str, object]:
    """Audit every declared capability and optional source-tree identity."""
    project = Path(root).resolve()
    rows: list[dict[str, object]] = []
    for specification in CAPABILITIES:
        paths = tuple(specification.get("paths", ()))
        artifacts = tuple(specification.get("artifacts", ()))
        missing_paths = [
            value for value in paths if not (project / value).is_file()
        ]
        artifact_status = {}
        for value in artifacts:
            passes, status = _artifact_passes(project / value)
            artifact_status[value] = {
                "passes": passes,
                "status_field": status,
            }
        qualified = not missing_paths and all(
            item["passes"] for item in artifact_status.values()
        )
        rows.append(
            {
                "id": specification["id"],
                "execution_mode": specification["mode"],
                "required_paths": list(paths),
                "missing_paths": missing_paths,
                "artifacts": artifact_status,
                "qualified": qualified,
            }
        )

    reference_report: dict[str, object] = {
        "checked": False,
        "reference_root": None,
        "files_compared": 0,
        "missing_files": [],
        "content_mismatches": [],
        "identical": None,
    }
    if reference_root is not None:
        reference = Path(reference_root).resolve()
        if reference.is_dir():
            missing = []
            mismatches = []
            reference_files = _reference_files(reference)
            for source in reference_files:
                relative = source.relative_to(reference)
                target = project / relative
                if not target.is_file():
                    missing.append(relative.as_posix())
                elif _digest(source) != _digest(target):
                    mismatches.append(relative.as_posix())
            reference_report = {
                "checked": True,
                "reference_root": str(reference),
                "files_compared": len(reference_files),
                "missing_files": missing,
                "content_mismatches": mismatches,
                "identical": not missing and not mismatches,
            }

    qualification = {
        "every_declared_capability_has_evidence": all(
            bool(row["qualified"]) for row in rows
        ),
        "all_resident_artifacts_are_qualified": all(
            bool(row["qualified"])
            for row in rows
            if row["execution_mode"] == "resident_vulkan"
        ),
        "retained_original_surfaces_are_present": all(
            not row["missing_paths"]
            for row in rows
            if str(row["execution_mode"]).startswith("retained")
        ),
    }
    if reference_report["checked"]:
        qualification["computational_source_copy_matches_reference"] = bool(
            reference_report["identical"]
        )
    implementation_digests = {
        value: _digest(project / value)
        for value in IMPLEMENTATION_PATHS
        if (project / value).is_file()
    }
    qualification["implementation_manifest_is_complete"] = (
        len(implementation_digests) == len(IMPLEMENTATION_PATHS)
    )
    return {
        "schema": "vulkan-tesseract.capability-parity.v1",
        "scope": (
            "All original computational capabilities, separated into "
            "resident Vulkan execution and intentionally retained offline "
            "training/oracle/postprocessing surfaces."
        ),
        "capabilities": rows,
        "reference_source_audit": reference_report,
        "implementation_sha256": implementation_digests,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }
