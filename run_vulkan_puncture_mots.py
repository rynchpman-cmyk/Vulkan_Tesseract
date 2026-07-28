"""Run the production head-on puncture/MOTS campaign on resident Vulkan CCZ4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tesseract_nr.vulkan_ccz4_vacuum import (
    VulkanCCZ4VacuumEvolutionBackend,
    VulkanCCZ4VacuumEvolutionSnapshot,
)
from tesseract_nr.vulkan_mots import (
    VulkanCCZ4Layout,
    VulkanPunctureMOTSObserver,
    VulkanPunctureMOTSParameters,
    brill_lindquist_puncture_ccz4,
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def save_snapshot(
    path: Path, snapshot: VulkanCCZ4VacuumEvolutionSnapshot
) -> None:
    """Save an observation readback without performing a second transfer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.array("vulkan-tesseract.ccz4-observer-snapshot.v1"),
        shape=np.asarray(snapshot.shape, dtype=np.int64),
        lengths=np.asarray(snapshot.lengths, dtype=np.float64),
        origins=np.asarray(snapshot.origins, dtype=np.float64),
        derivative_order=np.array(
            snapshot.derivative_order, dtype=np.int64
        ),
        ccz4_state=snapshot.ccz4_state,
        time=np.array(snapshot.time),
        steps=np.array(snapshot.steps, dtype=np.int64),
    )


def classified_names(record: dict[str, object]) -> list[str]:
    classified = record["common_classified"]
    assert isinstance(classified, dict)
    return [
        name
        for name in ("common_outer", "common_inner")
        if classified.get(name) is not None
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=120)
    parser.add_argument("--box-length", type=float, default=3.0)
    parser.add_argument("--separation", type=float, default=0.78)
    parser.add_argument("--total-bare-mass", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--cfl", type=float, default=0.04)
    parser.add_argument("--observation-cadence", type=int, default=10)
    parser.add_argument("--checkpoint-cadence", type=int, default=50)
    parser.add_argument("--maximum-degree", type=int, default=16)
    parser.add_argument("--individual-maximum-degree", type=int, default=8)
    parser.add_argument("--maximum-iterations", type=int, default=60)
    parser.add_argument("--expansion-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--branch-split-log-radius", type=float, default=0.2)
    parser.add_argument("--mask-cells", type=float, default=5.0)
    parser.add_argument("--boundary-mask-cells", type=float, default=5.0)
    parser.add_argument("--skip-individual", action="store_true")
    parser.add_argument("--skip-constraints", action="store_true")
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument(
        "--observer-restart",
        type=Path,
        help="observer JSON; defaults to final_observer.json beside restart",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("runs/vulkan_ccz4_puncture_mots_n120"),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and write the production contract without allocating Vulkan",
    )
    args = parser.parse_args()

    if args.resolution < 16 or args.box_length <= 0.0:
        parser.error("resolution and box length must be positive")
    if args.separation <= 0.0 or args.total_bare_mass <= 0.0:
        parser.error("separation and total bare mass must be positive")
    if args.steps < 0 or args.cfl <= 0.0:
        parser.error("steps and CFL are invalid")
    if args.observation_cadence < 1 or args.checkpoint_cadence < 1:
        parser.error("cadences must be positive")

    layout = VulkanCCZ4Layout(
        (args.resolution,) * 3,
        (args.box_length,) * 3,
        derivative_order=4,
    )
    observer_parameters = VulkanPunctureMOTSParameters(
        separation=args.separation,
        maximum_degree=args.maximum_degree,
        individual_maximum_degree=args.individual_maximum_degree,
        expansion_tolerance=args.expansion_tolerance,
        maximum_iterations=args.maximum_iterations,
        branch_split_log_radius=args.branch_split_log_radius,
        track_individual=not args.skip_individual,
        mask_cells=args.mask_cells,
        boundary_mask_cells=args.boundary_mask_cells,
        compute_constraints=not args.skip_constraints,
    )
    dt = args.cfl * min(layout.spacing)
    layout_report = layout.diagnostics()
    # Seven resident state buffers, one RHS, and one three-scalar audit.
    resident_bytes = (
        layout.points * (8 * 31 + 3) * np.dtype(np.float32).itemsize
    )
    configuration = {
        "schema": "vulkan-tesseract.puncture-mots-campaign.v1",
        "layout": layout_report,
        "separation": args.separation,
        "total_bare_mass": args.total_bare_mass,
        "steps": args.steps,
        "dt": dt,
        "cfl": args.cfl,
        "observation_cadence": args.observation_cadence,
        "checkpoint_cadence": args.checkpoint_cadence,
        "maximum_degree": args.maximum_degree,
        "individual_maximum_degree": args.individual_maximum_degree,
        "maximum_iterations": args.maximum_iterations,
        "expansion_tolerance": args.expansion_tolerance,
        "branch_split_log_radius": args.branch_split_log_radius,
        "track_individual": not args.skip_individual,
        "compute_constraints": not args.skip_constraints,
        "boundary_mode": "radiative_outflow",
        "slicing": "1+log",
        "precollapsed_lapse": "alpha=W=psi^-2",
        "resident_storage_bytes": resident_bytes,
        "resident_storage_gib": resident_bytes / 1024**3,
        "evolution_readback_policy": (
            "none between explicit observation/checkpoint requests"
        ),
    }
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "configuration.json", configuration)
    if args.plan_only:
        print(
            f"N={args.resolution} L={args.box_length:g} "
            f"dx={layout.spacing[0]:.8f} "
            f"reference_cells/r="
            f"{layout_report['reference_cells_per_radius']:.3f}"
        )
        print(
            "reference 16-cell gate="
            f"{layout_report['reference_resolution_gate_passed']}"
        )
        print(f"resident buffers={resident_bytes / 1024**3:.3f} GiB")
        print(f"wrote {output / 'configuration.json'}")
        return

    backend = VulkanCCZ4VacuumEvolutionBackend(
        layout.shape,
        lengths=layout.lengths,
        origins=layout.origins,
        boundary_mode="radiative_outflow",
        slicing="1+log",
        evolve_shift=True,
        derivative_order=layout.derivative_order,
    )
    observer = VulkanPunctureMOTSObserver(layout, observer_parameters)
    trajectory_path = output / "trajectory.json"
    records: list[dict[str, object]] = []

    if args.restart_from is None:
        packed = brill_lindquist_puncture_ccz4(
            layout,
            separation=args.separation,
            total_bare_mass=args.total_bare_mass,
        )
        state = backend.initialize(packed)
        starting_step = 0
    else:
        state = backend.load_checkpoint(args.restart_from)
        starting_step = state.steps
        observer_restart = args.observer_restart or (
            args.restart_from.parent / "final_observer.json"
        )
        payload = json.loads(observer_restart.read_text(encoding="utf-8"))
        if int(payload.get("step", -1)) != starting_step:
            raise ValueError(
                "evolution and observer restart steps do not match"
            )
        if payload.get("configuration") != configuration:
            raise ValueError("restart campaign configuration does not match")
        observer_payload = payload.get("observer")
        if not isinstance(observer_payload, dict):
            raise ValueError("restart is missing observer continuation")
        observer.restore(observer_payload)
        if trajectory_path.exists():
            records = json.loads(
                trajectory_path.read_text(encoding="utf-8")
            )

    def observe() -> None:
        snapshot = backend.snapshot(state)
        record = observer.observe_snapshot(snapshot)
        record["vulkan"] = {
            "maximum_determinant_residual": float(
                np.max(snapshot.projection_audit[:, 0])
            ),
            "maximum_trace_a_residual": float(
                np.max(snapshot.projection_audit[:, 1])
            ),
            "invalid_projected_geometries": int(
                np.count_nonzero(snapshot.projection_audit[:, 2] < 0.5)
            ),
            "host_reads": state.host_reads,
            "resident_steps": state.steps,
        }
        records.append(record)
        atomic_json(trajectory_path, records)
        if args.save_snapshots:
            save_snapshot(
                output / "snapshots" / f"step_{state.steps:07d}.npz",
                snapshot,
            )
        found = classified_names(record)
        constraints = record["constraints"]
        hamiltonian = (
            "disabled"
            if constraints is None
            else f"{constraints['hamiltonian_l2_interior']:.3e}"
        )
        health = record["health"]
        print(
            f"step={state.steps} t={state.time:.8f} "
            f"puncture_d={record['coordinate_separation']:.8f} "
            f"common={found or '-'} Hint={hamiltonian} "
            f"alpha_min={health['minimum_lapse']:.3e}"
        )
        classified = record["common_classified"]
        for name in found:
            value = classified[name]
            stability = value["normal_stability"]
            qualification = value["cartesian_qualification"]
            print(
                f"  {name}: r={value['coordinate_radius_mean']:.6f} "
                f"lambda0={stability['principal_eigenvalue_real']:.6e} "
                f"cells/r={qualification['cells_per_mean_radius']:.2f} "
                "sign_gate="
                f"{qualification['principal_mode_sign_gate_passed']}"
            )
        if (
            not health["finite"]
            or health["minimum_lapse"] <= 0.0
            or health["minimum_conformal_factor"] <= 0.0
            or health["minimum_conformal_metric_eigenvalue"] <= 0.0
        ):
            raise FloatingPointError(
                f"CCZ4 health gate failed at step {state.steps}: {health}"
            )

    def checkpoint() -> None:
        backend.save_checkpoint(state, output / "final_checkpoint.npz")
        atomic_json(
            output / "final_observer.json",
            {
                "format": (
                    "vulkan-tesseract.puncture-mots-restart-pair.v1"
                ),
                "step": state.steps,
                "time": state.time,
                "configuration": configuration,
                "observer": observer.checkpoint(),
            },
        )

    if not records or int(records[-1]["step"]) < starting_step:
        observe()
    for step in range(starting_step + 1, args.steps + 1):
        backend.step(state, dt)
        if step % args.observation_cadence == 0 or step == args.steps:
            observe()
        if step % args.checkpoint_cadence == 0 or step == args.steps:
            checkpoint()

    summary = {
        "format": "vulkan-tesseract.puncture-mots-summary.v1",
        "configuration": configuration,
        "completed_steps": state.steps,
        "final_time": state.time,
        "records": len(records),
        "host_reads": state.host_reads,
        "qualified_sign_observations": sum(
            int(
                value is not None
                and value["cartesian_qualification"][
                    "principal_mode_sign_gate_passed"
                ]
            )
            for record in records
            for value in (
                record["common_classified"].get("common_outer"),
                record["common_classified"].get("common_inner"),
            )
        ),
    }
    atomic_json(output / "summary.json", summary)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
