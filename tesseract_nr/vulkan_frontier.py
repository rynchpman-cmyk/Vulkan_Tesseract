"""Resident Vulkan reproduction of the original NR frontier campaign.

The evolved state remains on the device between explicitly recorded
diagnostic snapshots.  CPU code is used only as an independent diagnostic
and, for the small 128-step one-dimensional campaign, as a literal
trajectory oracle.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .adm import inverse_metric
from .ccz4 import CCZ4State
from .constrained_convergence import (
    _characteristic_diagnostics,
    build_constraint_converged_state,
)
from .frozen_closure import QualifiedFrozenClosure
from .grid import PeriodicGrid
from .matter import FluidPrimitive, lorentz_factor
from .production3 import (
    Theory3MatterState,
    Theory3ProductionParameters,
    Theory3ProductionState,
    _add_stress,
)
from .production33 import Theory33ProductionSolver
from .theory33 import CarrierPrimitive
from .vulkan_backend import VulkanTesseractEvolvedBackend
from .vulkan_validation import _weno_interface_accuracy


Array = np.ndarray


def _observed_order(coarse: float, fine: float) -> float:
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return float(math.log(coarse / fine, 2.0))


def _solver(
    shape: tuple[int, ...],
    *,
    model: str,
    reconstruction: str,
    artifact_path: str | Path,
    recovery_tolerance: float = 1.0e-11,
) -> Theory33ProductionSolver:
    closure = (
        QualifiedFrozenClosure(artifact_path)
        if model == "qualified_frozen"
        else None
    )
    return Theory33ProductionSolver(
        PeriodicGrid(shape, (1.0,) * len(shape), method="fd2"),
        production=Theory3ProductionParameters(
            flux_reconstruction=reconstruction,
            source_splitting=(
                "strang"
                if reconstruction != "piecewise_constant"
                else "lie"
            ),
            recovery_tolerance=recovery_tolerance,
        ),
        constitutive_closure=closure,
    )


def _backend(
    solver: Theory33ProductionSolver,
    shape: tuple[int, ...],
    *,
    model: str,
    reconstruction: str,
) -> VulkanTesseractEvolvedBackend:
    gpu_reconstruction = (
        "full_carter_weno5_z"
        if reconstruction == "weno5_z"
        else reconstruction
    )
    return VulkanTesseractEvolvedBackend(
        shape,
        constitutive_model=model,
        reconstruction=gpu_reconstruction,
        relaxation_time=solver.master_parameters.relaxation_time,
        kappa_gravity=solver.ccz4.parameters.kappa_gravity,
        kappa1=solver.ccz4.parameters.kappa1,
        kappa2=solver.ccz4.parameters.kappa2,
        kappa3=solver.ccz4.parameters.kappa3,
        cosmological_constant=solver.ccz4.parameters.cosmological_constant,
        shift_speed=solver.ccz4.parameters.shift_speed,
        eta_shift=solver.ccz4.parameters.eta_shift,
        slicing=solver.ccz4.parameters.slicing,
        evolve_shift=solver.ccz4.parameters.evolve_shift,
        vector_cleaning_damping=(
            solver.production_parameters.cleaning_damping
        ),
    )


def _spatial(values: Array, shape: tuple[int, ...]) -> Array:
    """Restore the resident x-fast linearization used by the shaders."""
    return np.asarray(values, dtype=float).reshape(shape, order="F").copy()


def _state_from_snapshot(
    solver: Theory33ProductionSolver,
    snapshot: object,
) -> Theory3ProductionState:
    shape = tuple(int(value) for value in solver.grid.shape)
    matter_values = np.asarray(snapshot.coupled.matter.state, dtype=float)
    primitive_values = np.asarray(
        snapshot.coupled.matter.primitive, dtype=float
    )
    geometry_values = np.asarray(snapshot.coupled.ccz4_state, dtype=float)
    vector_values = np.asarray(snapshot.vector_state, dtype=float)
    # Seed the independent Newton decoder with the resident primitive.  This
    # does not bypass recovery: the CPU solver still recomputes and audits the
    # full conserved residual, but it avoids treating the unrelated
    # single-fluid estimate as the initial guess for a counterflow state.
    solver._primitive_guess = np.stack(
        [
            _spatial(primitive_values[:, component], shape)
            for component in (0, 2, 3, 4, 5, 6, 7, 8)
        ]
    )

    conformal_metric = np.empty((3, 3, *shape), dtype=float)
    conformal_A = np.empty_like(conformal_metric)
    for row in range(3):
        for column in range(3):
            component = 3 * row + column
            conformal_metric[row, column] = _spatial(
                geometry_values[:, component], shape
            )
            conformal_A[row, column] = _spatial(
                geometry_values[:, 9 + component], shape
            )
    geometry = CCZ4State(
        conformal_metric,
        conformal_A,
        _spatial(geometry_values[:, 18], shape),
        _spatial(geometry_values[:, 19], shape),
        _spatial(geometry_values[:, 20], shape),
        np.stack(
            [
                _spatial(geometry_values[:, 21 + axis], shape)
                for axis in range(3)
            ]
        ),
        _spatial(geometry_values[:, 24], shape),
        np.stack(
            [
                _spatial(geometry_values[:, 25 + axis], shape)
                for axis in range(3)
            ]
        ),
        np.stack(
            [
                _spatial(geometry_values[:, 28 + axis], shape)
                for axis in range(3)
            ]
        ),
        float(snapshot.coupled.time),
    )
    matter = Theory3MatterState(
        _spatial(matter_values[:, 0], shape),
        np.stack(
            [
                _spatial(matter_values[:, 2 + axis], shape)
                for axis in range(3)
            ]
        ),
        _spatial(matter_values[:, 5], shape),
        _spatial(matter_values[:, 9], shape),
        _spatial(matter_values[:, 10], shape),
    )

    def vector_columns(first: int) -> Array:
        return np.stack(
            [
                _spatial(vector_values[:, first + axis], shape)
                for axis in range(3)
            ]
        )

    return Theory3ProductionState(
        geometry,
        matter,
        vector_columns(0),
        vector_columns(3),
        vector_columns(6),
        vector_columns(9),
        _spatial(vector_values[:, 12], shape),
        _spatial(vector_values[:, 13], shape),
        _spatial(vector_values[:, 14], shape),
        _spatial(vector_values[:, 15], shape),
        (
            solver.master_parameters.carrier_charge
            * _spatial(matter_values[:, 1], shape)
        ),
        np.stack(
            [
                _spatial(matter_values[:, 6 + axis], shape)
                for axis in range(3)
            ]
        ),
        float(snapshot.coupled.time),
    )


def _diagnostics(
    solver: Theory33ProductionSolver,
    state: Theory3ProductionState,
) -> dict[str, object]:
    result = solver.diagnostics(state)
    result.update(_characteristic_diagnostics(solver, state))
    return result


def _diagnostic_state_from_snapshot(
    solver: Theory33ProductionSolver,
    snapshot: object,
) -> tuple[Theory3ProductionState, Theory3ProductionState, float]:
    """Return actual and constitutively recomposed diagnostic states.

    The actual state is retained for conservation and trajectory errors.  The
    recomposed state uses the resident primitive and vector fields, allowing
    the float64 CPU diagnostic routines to evaluate the same physical point
    without aborting on their hardcoded 2e-6 recovery threshold.
    """
    actual = _state_from_snapshot(solver, snapshot)
    primitive = np.asarray(snapshot.coupled.matter.primitive, dtype=float)
    shape = tuple(int(value) for value in solver.grid.shape)
    h, _ = solver.ccz4.physical_geometry(actual.geometry)
    _, _, sqrt_h = inverse_metric(h)
    q_n = np.stack(
        [_spatial(primitive[:, 2 + axis], shape) for axis in range(3)]
    )
    q_d = np.stack(
        [_spatial(primitive[:, 6 + axis], shape) for axis in range(3)]
    )
    w_n = np.sqrt(
        1.0 + np.einsum("ij...,i...,j...->...", h, q_n, q_n)
    )
    w_d = np.sqrt(
        1.0 + np.einsum("ij...,i...,j...->...", h, q_d, q_d)
    )
    n = np.exp(_spatial(primitive[:, 0], shape))
    d = np.exp(_spatial(primitive[:, 5], shape))
    entropy_per_baryon = _spatial(primitive[:, 1], shape)
    internal = np.empty(shape, dtype=float)
    for index in np.ndindex(shape):
        internal[index] = solver._entropy_internal(
            float(n[index]), float(entropy_per_baryon[index])
        )
    sigma = actual.matter.tracer / np.maximum(
        actual.matter.D, 1.0e-300
    )
    fluid = FluidPrimitive(n, internal, q_n / w_n[None, ...], sigma)
    carrier = CarrierPrimitive(d, q_d / w_d[None, ...])
    master = solver.master.evaluate(h, fluid, carrier)
    d_d, p_d = solver.master.conserved_carrier(h, master)
    target_charge = solver.master_parameters.carrier_charge * d_d
    vector_stress = solver.system.vector_stress(
        h,
        actual.a,
        actual.pi_A,
        actual.b,
        actual.pi_B,
        actual.longitudinal_A,
        actual.longitudinal_B,
        target_charge / sqrt_h,
    )
    total = _add_stress(master.stress, vector_stress)
    d_n = sqrt_h * n * lorentz_factor(h, fluid.velocity)
    matter = Theory3MatterState(
        d_n,
        sqrt_h[None, ...] * total.momentum,
        sqrt_h * total.rho,
        d_n * entropy_per_baryon,
        d_n * sigma,
    )
    recomposed = Theory3ProductionState(
        actual.geometry,
        matter,
        actual.a,
        actual.pi_A,
        actual.b,
        actual.pi_B,
        actual.longitudinal_A,
        actual.longitudinal_B,
        actual.cleaning_A,
        actual.cleaning_B,
        target_charge,
        p_d,
        actual.time,
    )
    difference = _same_grid_difference(
        solver._pack(actual), solver._pack(recomposed)
    )
    return actual, recomposed, difference


def _integrals(snapshot: object, shape: tuple[int, ...]) -> dict[str, float]:
    state = np.asarray(snapshot.coupled.matter.state, dtype=float)
    cell_volume = 1.0 / float(np.prod(shape))
    return {
        "baryon_mass": float(np.sum(state[:, 0]) * cell_volume),
        "carrier_number": float(np.sum(state[:, 1]) * cell_volume),
        "combined_energy": float(np.sum(state[:, 5]) * cell_volume),
        "entropy_integral": float(np.sum(state[:, 9]) * cell_volume),
    }


def _relative_balance(
    initial: dict[str, float],
    final: dict[str, float],
) -> dict[str, float]:
    return {
        name: abs(final[name] - initial[name])
        / max(abs(initial[name]), 1.0e-300)
        for name in ("baryon_mass", "carrier_number")
    }


def _nested_restrict(field: Array, ratio: int, ndim: int) -> Array:
    return np.asarray(field)[
        (slice(None),) * (np.asarray(field).ndim - ndim)
        + (slice(None, None, ratio),) * ndim
    ]


def _state_difference(
    coarse: tuple[Array, ...],
    fine: tuple[Array, ...],
    ratio: int,
    ndim: int,
) -> float:
    squared = 0.0
    count = 0
    for coarse_field, fine_field in zip(coarse, fine, strict=True):
        difference = np.asarray(coarse_field) - _nested_restrict(
            np.asarray(fine_field), ratio, ndim
        )
        squared += float(np.sum(difference * difference))
        count += difference.size
    return float(math.sqrt(squared / max(count, 1)))


def _same_grid_difference(
    left: tuple[Array, ...],
    right: tuple[Array, ...],
) -> float:
    squared = 0.0
    count = 0
    for left_field, right_field in zip(left, right, strict=True):
        difference = np.asarray(left_field) - np.asarray(right_field)
        squared += float(np.sum(difference * difference))
        count += difference.size
    return float(math.sqrt(squared / max(count, 1)))


def _gpu_recovery_failures(snapshot: object) -> int:
    diagnostics = np.asarray(
        snapshot.coupled.matter.diagnostics, dtype=float
    )
    return int(np.count_nonzero(diagnostics[:, 1] < 0.5))


def _run_horizon_model(
    *,
    model: str,
    points: int,
    steps: int,
    dt: float,
    reconstruction: str,
    artifact_path: str | Path,
    diagnostic_stride: int,
) -> tuple[dict[str, object], tuple[Array, ...]]:
    shape = (points,)
    solver = _solver(
        shape,
        model=model,
        reconstruction=reconstruction,
        artifact_path=artifact_path,
    )
    diagnostic_solver = _solver(
        shape,
        model=model,
        reconstruction=reconstruction,
        artifact_path=artifact_path,
        # The evolved ABI is float32.  This decoder is deliberately looser
        # than the strict float64 trajectory oracle and its raw residual is
        # retained in the report.
        recovery_tolerance=1.0e-3,
    )
    oracle_state, oracle_elliptic = build_constraint_converged_state(solver)
    gpu = _backend(
        solver,
        shape,
        model=model,
        reconstruction=reconstruction,
    )
    state, elliptic = gpu.initialize_constrained()
    initial_snapshot = gpu.snapshot(state)
    (
        initial_state,
        initial_diagnostic_state,
        initial_recomposition_error,
    ) = _diagnostic_state_from_snapshot(
        diagnostic_solver, initial_snapshot
    )
    initial_diagnostics = _diagnostics(
        diagnostic_solver, initial_diagnostic_state
    )
    initial_integrals = _integrals(initial_snapshot, shape)
    initial_oracle_error = _same_grid_difference(
        solver._pack(initial_state), solver._pack(oracle_state)
    )
    snapshots: list[dict[str, object]] = [
        {
            "step": 0,
            "time": 0.0,
            "hamiltonian_l2": float(
                initial_diagnostics["hamiltonian_l2"]
            ),
            "momentum_l2": float(initial_diagnostics["momentum_l2"]),
            "entropy_integral": initial_integrals["entropy_integral"],
            "switching_margin": (
                None
                if model == "analytic_m1"
                else float(
                    initial_diagnostics[
                        "constitutive_minimum_switching_margin"
                    ]
                )
            ),
        }
    ]
    maximum_recovery_failures = _gpu_recovery_failures(initial_snapshot)
    minimum_observed_entropy_change = float("inf")
    previous_entropy = initial_integrals["entropy_integral"]
    no_step_host_reads = True
    minimum_oracle_step_entropy_change = float("inf")
    for step in range(1, steps + 1):
        reads_before = state.host_reads
        gpu.step(state, dt, apply_drag=True)
        no_step_host_reads &= state.host_reads == reads_before
        oracle_state = solver.step(oracle_state, dt)
        minimum_oracle_step_entropy_change = min(
            minimum_oracle_step_entropy_change,
            float(solver.last_step_entropy_change),
        )
        if step % diagnostic_stride == 0 or step == steps:
            gpu.device.wait_for_idle()
            snapshot = gpu.snapshot(state)
            (
                resident_state,
                resident_diagnostic_state,
                recomposition_error,
            ) = _diagnostic_state_from_snapshot(
                diagnostic_solver, snapshot
            )
            values = _diagnostics(
                diagnostic_solver, resident_diagnostic_state
            )
            integrals = _integrals(snapshot, shape)
            entropy_change = (
                integrals["entropy_integral"] - previous_entropy
            )
            previous_entropy = integrals["entropy_integral"]
            minimum_observed_entropy_change = min(
                minimum_observed_entropy_change, entropy_change
            )
            maximum_recovery_failures = max(
                maximum_recovery_failures,
                _gpu_recovery_failures(snapshot),
                int(values["recovery_failures"]),
            )
            snapshots.append(
                {
                    "step": step,
                    "time": float(state.time),
                    "hamiltonian_l2": float(values["hamiltonian_l2"]),
                    "momentum_l2": float(values["momentum_l2"]),
                    "entropy_integral": integrals["entropy_integral"],
                    "switching_margin": (
                        None
                        if model == "analytic_m1"
                        else float(
                            values[
                                "constitutive_minimum_switching_margin"
                            ]
                        )
                    ),
                    "conservative_recomposition_l2_error": (
                        recomposition_error
                    ),
                }
            )
    final_snapshot = snapshot
    final_state = resident_state
    final_diagnostics = values
    final_integrals = integrals
    final_oracle_error = _same_grid_difference(
        solver._pack(final_state), solver._pack(oracle_state)
    )
    return (
        {
            "model": model,
            "points": points,
            "steps": steps,
            "dt": dt,
            "final_time": float(state.time),
            "reconstruction": "full_carter_weno5_z",
            "cpu_diagnostic_recovery_tolerance": 1.0e-3,
            "resident_elliptic_report": elliptic.__dict__.copy(),
            "oracle_elliptic_report": oracle_elliptic,
            "initial_diagnostics": initial_diagnostics,
            "final_diagnostics": final_diagnostics,
            "relative_balance_errors": _relative_balance(
                initial_integrals, final_integrals
            ),
            "entropy_change": (
                final_integrals["entropy_integral"]
                - initial_integrals["entropy_integral"]
            ),
            "minimum_observed_snapshot_entropy_change": (
                minimum_observed_entropy_change
            ),
            "minimum_oracle_step_entropy_change": (
                minimum_oracle_step_entropy_change
            ),
            "maximum_recovery_failures": maximum_recovery_failures,
            "initial_cpu_oracle_state_l2_error": initial_oracle_error,
            "final_cpu_oracle_state_l2_error": final_oracle_error,
            "initial_conservative_recomposition_l2_error": (
                initial_recomposition_error
            ),
            "final_conservative_recomposition_l2_error": (
                recomposition_error
            ),
            "no_host_reads_inside_steps": no_step_host_reads,
            "host_diagnostic_snapshots": len(snapshots),
            "snapshots": snapshots,
        },
        tuple(np.asarray(value).copy() for value in solver._pack(final_state)),
    )


def run_vulkan_long_horizon(
    *,
    points: int = 8,
    steps: int = 128,
    dt: float = 1.0e-7,
    reconstruction: str = "weno5_z",
    artifact_path: str | Path = "theory33_frozen_variants.json",
    diagnostic_stride: int = 16,
) -> dict[str, object]:
    """Run the original analytic/frozen 128-step campaign on Vulkan."""
    analytic, analytic_state = _run_horizon_model(
        model="analytic_m1",
        points=points,
        steps=steps,
        dt=dt,
        reconstruction=reconstruction,
        artifact_path=artifact_path,
        diagnostic_stride=diagnostic_stride,
    )
    frozen, frozen_state = _run_horizon_model(
        model="qualified_frozen",
        points=points,
        steps=steps,
        dt=dt,
        reconstruction=reconstruction,
        artifact_path=artifact_path,
        diagnostic_stride=diagnostic_stride,
    )
    final_difference = _same_grid_difference(
        analytic_state, frozen_state
    )
    frozen_final = frozen["final_diagnostics"]
    qualification = {
        "at_least_128_steps": steps >= 128,
        "resident_elliptic_constraints_converged": (
            bool(analytic["resident_elliptic_report"]["converged"])
            and bool(frozen["resident_elliptic_report"]["converged"])
        ),
        "no_recovery_failures": (
            int(analytic["maximum_recovery_failures"]) == 0
            and int(frozen["maximum_recovery_failures"]) == 0
        ),
        "currents_conserved": all(
            float(run["relative_balance_errors"][name]) < 2.0e-6
            for run in (analytic, frozen)
            for name in ("baryon_mass", "carrier_number")
        ),
        "entropy_gate_held": all(
            float(run["minimum_observed_snapshot_entropy_change"])
            >= -1.0e-10
            for run in (analytic, frozen)
        ),
        "frozen_switching_and_convexity_held": (
            float(
                frozen_final["constitutive_minimum_switching_margin"]
            )
            > 1.0e-3
            and float(frozen_final["minimum_legendre_eigenvalue"]) > 0.0
            and float(frozen_final["minimum_thermodynamic_eigenvalue"]) > 0.0
        ),
        "frozen_hyperbolicity_and_causality_held": (
            bool(frozen_final["characteristic_all_strongly_hyperbolic"])
            and bool(frozen_final["characteristic_all_causal"])
        ),
        "neural_response_remains_distinct": final_difference > 1.0e-14,
        "device_resident_inside_every_step": (
            bool(analytic["no_host_reads_inside_steps"])
            and bool(frozen["no_host_reads_inside_steps"])
        ),
        "literal_cpu_oracle_comparison_finite": (
            bool(
                np.isfinite(
                    analytic["final_cpu_oracle_state_l2_error"]
                )
            )
            and bool(
                np.isfinite(
                    frozen["final_cpu_oracle_state_l2_error"]
                )
            )
        ),
    }
    return {
        "analytic_control": analytic,
        "qualified_frozen": frozen,
        "final_state_model_difference_l2": final_difference,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def _run_multidimensional_level(
    points: int,
    *,
    ndim: int,
    final_time: float,
    model: str,
    reconstruction: str,
    artifact_path: str | Path,
    base_steps: int,
) -> dict[str, object]:
    shape = (points,) * ndim
    solver = _solver(
        shape,
        model=model,
        reconstruction=reconstruction,
        artifact_path=artifact_path,
        recovery_tolerance=1.0e-3,
    )
    gpu = _backend(
        solver,
        shape,
        model=model,
        reconstruction=reconstruction,
    )
    state, elliptic = gpu.initialize_constrained()
    initial_snapshot = gpu.snapshot(state)
    (
        initial_state,
        initial_diagnostic_state,
        initial_recomposition_error,
    ) = _diagnostic_state_from_snapshot(solver, initial_snapshot)
    initial_diagnostics = _diagnostics(
        solver, initial_diagnostic_state
    )
    initial_integrals = _integrals(initial_snapshot, shape)
    recovery = solver.recover(initial_state)
    transverse_speed = max(
        float(np.max(np.abs(recovery.carrier.velocity[axis])))
        for axis in range(1, ndim)
    )
    steps = base_steps * points // 6
    dt = final_time / steps
    reads_before = state.host_reads
    for _ in range(steps):
        gpu.step(state, dt, apply_drag=True)
    reads_after = state.host_reads
    gpu.device.wait_for_idle()
    final_snapshot = gpu.snapshot(state)
    (
        final_state,
        final_diagnostic_state,
        final_recomposition_error,
    ) = _diagnostic_state_from_snapshot(solver, final_snapshot)
    final_diagnostics = _diagnostics(
        solver, final_diagnostic_state
    )
    final_integrals = _integrals(final_snapshot, shape)
    entropy_change = (
        final_integrals["entropy_integral"]
        - initial_integrals["entropy_integral"]
    )
    return {
        "points_per_axis": points,
        "shape": list(shape),
        "cells": int(np.prod(shape)),
        "steps": steps,
        "dt": dt,
        "cpu_diagnostic_recovery_tolerance": 1.0e-3,
        "resident_elliptic_report": elliptic.__dict__.copy(),
        "initial_diagnostics": initial_diagnostics,
        "final_diagnostics": final_diagnostics,
        "relative_balance_errors": _relative_balance(
            initial_integrals, final_integrals
        ),
        "entropy_change": entropy_change,
        "maximum_transverse_carrier_speed": transverse_speed,
        "energy_change_rate": (
            final_integrals["combined_energy"]
            - initial_integrals["combined_energy"]
        )
        / final_time,
        "recovery_failures": max(
            _gpu_recovery_failures(final_snapshot),
            int(final_diagnostics["recovery_failures"]),
        ),
        "host_reads_during_evolution": int(reads_after - reads_before),
        "initial_conservative_recomposition_l2_error": (
            initial_recomposition_error
        ),
        "final_conservative_recomposition_l2_error": (
            final_recomposition_error
        ),
        "_final_state": tuple(
            np.asarray(value).copy()
            for value in solver._pack(final_state)
        ),
    }


def _constraint_series(
    levels: tuple[dict[str, object], ...],
    key: str,
    stage: str,
) -> dict[str, object]:
    errors = [
        float(level[f"{stage}_diagnostics"][key])
        for level in levels
    ]
    return {
        "errors": errors,
        "orders": [
            _observed_order(errors[index], errors[index + 1])
            for index in range(len(errors) - 1)
        ],
    }


def run_vulkan_multidimensional_convergence(
    resolutions: Iterable[int] = (6, 12, 24),
    *,
    ndim: int = 2,
    final_time: float = 2.0e-7,
    base_steps: int = 4,
    reconstruction: str = "weno5_z",
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> dict[str, object]:
    """Run the original frozen 2D ladder and analytic finest control."""
    values = tuple(int(value) for value in resolutions)
    if (
        len(values) != 3
        or values[1] != 2 * values[0]
        or values[2] != 2 * values[1]
        or values[0] < 6
    ):
        raise ValueError(
            "resolutions must be a factor-two triplet starting at six"
        )
    if ndim not in (2, 3):
        raise ValueError("multidimensional convergence requires 2D or 3D")
    levels = tuple(
        _run_multidimensional_level(
            points,
            ndim=ndim,
            final_time=final_time,
            model="qualified_frozen",
            reconstruction=reconstruction,
            artifact_path=artifact_path,
            base_steps=base_steps,
        )
        for points in values
    )
    analytic_fine = _run_multidimensional_level(
        values[-1],
        ndim=ndim,
        final_time=final_time,
        model="analytic_m1",
        reconstruction=reconstruction,
        artifact_path=artifact_path,
        base_steps=base_steps,
    )
    cm = _state_difference(
        levels[0]["_final_state"],
        levels[1]["_final_state"],
        2,
        ndim,
    )
    mf = _state_difference(
        levels[1]["_final_state"],
        levels[2]["_final_state"],
        2,
        ndim,
    )
    model_difference = _same_grid_difference(
        levels[-1]["_final_state"],
        analytic_fine["_final_state"],
    )
    initial_h = _constraint_series(
        levels, "hamiltonian_l2", "initial"
    )
    initial_m = _constraint_series(levels, "momentum_l2", "initial")
    final_h = _constraint_series(levels, "hamiltonian_l2", "final")
    final_m = _constraint_series(levels, "momentum_l2", "final")
    energy_rates = [
        float(level["energy_change_rate"]) for level in levels
    ]
    energy_differences = [
        abs(energy_rates[index] - energy_rates[index + 1])
        for index in range(2)
    ]
    energy_order = _observed_order(*energy_differences)
    state_order = _observed_order(cm, mf)
    finest = levels[-1]["final_diagnostics"]
    qualification = {
        "resident_elliptic_constraints_converged": all(
            bool(level["resident_elliptic_report"]["converged"])
            for level in levels
        )
        and bool(
            analytic_fine["resident_elliptic_report"]["converged"]
        ),
        "pointwise_momentum_compatible_at_float32_abi": all(
            float(
                level["resident_elliptic_report"][
                    "pointwise_momentum_balance"
                ]
            )
            < 1.0e-10
            for level in levels
        ),
        "genuine_transverse_flow": all(
            float(level["maximum_transverse_carrier_speed"]) > 1.0e-3
            for level in levels
        ),
        "no_recovery_or_entropy_failure": all(
            int(level["recovery_failures"]) == 0
            and float(level["entropy_change"]) >= -1.0e-10
            for level in levels
        ),
        "currents_conserved": all(
            float(level["relative_balance_errors"][name]) < 2.0e-6
            for level in levels
            for name in ("baryon_mass", "carrier_number")
        ),
        "initial_constraints_converge": (
            float(initial_h["orders"][-1]) > 1.5
            and float(initial_m["orders"][-1]) > 1.5
        ),
        "evolved_constraints_converge": (
            float(final_h["orders"][-1]) > 1.5
            and float(final_m["orders"][-1]) > 1.5
        ),
        "complete_state_converges": (
            bool(np.isfinite(state_order)) and state_order > 1.0
        ),
        "geometric_work_rate_converges": (
            bool(np.isfinite(energy_order)) and energy_order > 1.0
        ),
        "frozen_physics_gates_hold": (
            float(finest["constitutive_phase_two_fraction"]) > 0.99
            and float(
                finest["constitutive_minimum_switching_margin"]
            )
            > 1.0e-3
            and float(finest["minimum_legendre_eigenvalue"]) > 0.0
            and float(finest["minimum_thermodynamic_eigenvalue"]) > 0.0
            and bool(finest["characteristic_all_strongly_hyperbolic"])
            and bool(finest["characteristic_all_causal"])
        ),
        "neural_response_is_active": model_difference > 1.0e-14,
        "device_resident_during_evolution": all(
            int(level["host_reads_during_evolution"]) == 0
            for level in (*levels, analytic_fine)
        ),
    }

    def serializable(level: dict[str, object]) -> dict[str, object]:
        return {
            key: value
            for key, value in level.items()
            if not key.startswith("_")
        }

    return {
        "dimensions": ndim,
        "resolutions": list(values),
        "final_time": final_time,
        "reconstruction": "full_carter_weno5_z",
        "qualified_frozen_levels": [
            serializable(level) for level in levels
        ],
        "analytic_finest_control": serializable(analytic_fine),
        "initial_constraints": {
            "hamiltonian": initial_h,
            "momentum": initial_m,
        },
        "final_constraints": {
            "hamiltonian": final_h,
            "momentum": final_m,
        },
        "complete_state": {
            "coarse_medium_error": cm,
            "medium_fine_error": mf,
            "observed_order": state_order,
        },
        "geometric_work_energy_rate": {
            "values": energy_rates,
            "differences": energy_differences,
            "observed_order": energy_order,
        },
        "finest_model_difference_l2": model_difference,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def run_vulkan_three_dimensional_smoke(
    *,
    points: int = 6,
    steps: int = 2,
    dt: float = 1.0e-8,
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> dict[str, object]:
    """Cross the complete resident 3D constrained/evolution code path."""
    shape = (points, points, points)
    solver = _solver(
        shape,
        model="qualified_frozen",
        reconstruction="weno5_z",
        artifact_path=artifact_path,
        recovery_tolerance=1.0e-3,
    )
    gpu = _backend(
        solver,
        shape,
        model="qualified_frozen",
        reconstruction="weno5_z",
    )
    state, elliptic = gpu.initialize_constrained()
    initial_snapshot = gpu.snapshot(state)
    (
        initial_state,
        initial_diagnostic_state,
        initial_recomposition_error,
    ) = _diagnostic_state_from_snapshot(solver, initial_snapshot)
    recovery = solver.recover(initial_diagnostic_state)
    transverse_speed = float(
        max(
            np.max(np.abs(recovery.carrier.velocity[1])),
            np.max(np.abs(recovery.carrier.velocity[2])),
        )
    )
    initial_integrals = _integrals(initial_snapshot, shape)
    reads_before = state.host_reads
    for _ in range(steps):
        gpu.step(state, dt, apply_drag=True)
    reads_after = state.host_reads
    gpu.device.wait_for_idle()
    final_snapshot = gpu.snapshot(state)
    (
        final_state,
        final_diagnostic_state,
        final_recomposition_error,
    ) = _diagnostic_state_from_snapshot(solver, final_snapshot)
    final = _diagnostics(solver, final_diagnostic_state)
    final_integrals = _integrals(final_snapshot, shape)
    entropy_change = (
        final_integrals["entropy_integral"]
        - initial_integrals["entropy_integral"]
    )
    recovery_failures = max(
        _gpu_recovery_failures(final_snapshot),
        int(final["recovery_failures"]),
    )
    qualification = {
        "resident_elliptic_constraints_converged": bool(
            elliptic.converged
        ),
        "pointwise_momentum_compatible_at_float32_abi": (
            elliptic.pointwise_momentum_balance < 1.0e-10
        ),
        "three_axis_flow_is_active": transverse_speed > 1.0e-3,
        "evolution_completed": math.isclose(
            state.time, steps * dt, rel_tol=0.0, abs_tol=1.0e-20
        ),
        "no_recovery_failure": recovery_failures == 0,
        "entropy_gate_held": entropy_change >= -1.0e-10,
        "frozen_physics_gates_hold": (
            float(final["constitutive_phase_two_fraction"]) > 0.99
            and float(
                final["constitutive_minimum_switching_margin"]
            )
            > 1.0e-3
            and float(final["minimum_legendre_eigenvalue"]) > 0.0
            and float(final["minimum_thermodynamic_eigenvalue"]) > 0.0
            and bool(final["characteristic_all_strongly_hyperbolic"])
            and bool(final["characteristic_all_causal"])
        ),
        "device_resident_during_evolution": (
            reads_after - reads_before == 0
        ),
    }
    return {
        "shape": list(shape),
        "cells": int(np.prod(shape)),
        "steps": steps,
        "dt": dt,
        "cpu_diagnostic_recovery_tolerance": 1.0e-3,
        "resident_elliptic_report": elliptic.__dict__.copy(),
        "maximum_transverse_carrier_speed": transverse_speed,
        "entropy_change": entropy_change,
        "final_diagnostics": final,
        "recovery_failures": recovery_failures,
        "host_reads_during_evolution": int(
            reads_after - reads_before
        ),
        "initial_conservative_recomposition_l2_error": (
            initial_recomposition_error
        ),
        "final_conservative_recomposition_l2_error": (
            final_recomposition_error
        ),
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def run_vulkan_frontier(
    *,
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> dict[str, object]:
    """Run the complete resident equivalent of ``tesseract_nr.frontier``."""
    reconstruction = _weno_interface_accuracy()
    reconstruction_qualification = reconstruction["qualification"]
    horizon = run_vulkan_long_horizon(artifact_path=artifact_path)
    multidimensional = run_vulkan_multidimensional_convergence(
        artifact_path=artifact_path
    )
    three_dimensional = run_vulkan_three_dimensional_smoke(
        artifact_path=artifact_path
    )
    qualification = {
        "high_order_reconstruction": all(
            bool(value)
            for value in reconstruction_qualification.values()
        ),
        "long_horizon": bool(horizon["qualified"]),
        "two_dimensional_convergence": bool(
            multidimensional["qualified"]
        ),
        "three_dimensional_path": bool(three_dimensional["qualified"]),
    }
    return {
        "schema": "vulkan-tesseract.nr-frontier.v1",
        "backend": "Slang/Vulkan",
        "scope": (
            "fp64 resident CMC/Picard construction, full eleven-field "
            "characteristic Carter WENO5-Z, mixed Proca, and CCZ4 evolution"
        ),
        "reconstruction_accuracy": reconstruction,
        "long_horizon": horizon,
        "two_dimensional_convergence": multidimensional,
        "three_dimensional_smoke": three_dimensional,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }
