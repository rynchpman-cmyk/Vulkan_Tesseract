"""Independent CPU oracle for the first GPU-resident Theory 3.3 slice."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np

from .frozen_closure import QualifiedFrozenClosure
from .ccz4 import CCZ4Parameters, CCZ4Solver, CCZ4State
from .grid import PeriodicGrid
from .matter import FluidPrimitive
from .production3 import (
    Theory3MatterState,
    Theory3ProductionParameters,
    Theory3ProductionState,
)
from .production33 import Theory33ProductionSolver
from .theory33 import CarrierPrimitive
from .vulkan_backend import (
    DIAGNOSTIC_FIELDS,
    STATE_FIELDS,
    VulkanCCZ4GeometryBackend,
    VulkanMixedProcaBackend,
    VulkanTesseractEvolvedBackend,
    VulkanTesseractCoupledBackend,
    VulkanTheory33Backend,
    VulkanTheory33MultidimensionalBackend,
    VulkanTheory33SourceBackend,
)


Array = np.ndarray


def _source_solver(constitutive_model: str) -> Theory33ProductionSolver:
    if constitutive_model not in {"analytic_m1", "qualified_frozen"}:
        raise ValueError("unknown constitutive model")
    return Theory33ProductionSolver(
        PeriodicGrid((4,), (1.0,), method="fd2"),
        constitutive_closure=(
            QualifiedFrozenClosure()
            if constitutive_model == "qualified_frozen"
            else None
        ),
    )


def full_material_prediction(
    solver: Theory33ProductionSolver,
    primitive: Array,
    phase_override: bool | None = None,
) -> Array:
    """Independent float64 full three-vector Carter conserved prediction."""
    x = np.asarray(primitive, dtype=float)
    if x.shape != (9,):
        raise ValueError("full primitive must contain nine entries")
    production_order = np.concatenate(
        ([x[0]], x[2:5], [x[5]], x[6:9])
    )
    return solver._cell_predictions(
        production_order,
        np.eye(3),
        1.0,
        float(x[1]),
        phase_override,
    )


def reference_full_principal(
    solver: Theory33ProductionSolver,
    primitive: Array,
    geometry: Array,
    *,
    axis: int,
    phase_override: bool | None = None,
) -> tuple[Array, Array, Array]:
    """CPU float64 curved-space nine-field U/F/path evaluation."""
    x = np.asarray(primitive, dtype=float)
    geometric = np.asarray(geometry, dtype=float)
    h = geometric[:9].reshape(3, 3)
    lapse = geometric[9]
    shift = geometric[10:13]
    production_order = np.concatenate(
        ([x[0]], x[2:5], [x[5]], x[6:9], [x[1]])
    )
    fields = solver._carter_principal_fields(
        production_order,
        h,
        lapse,
        shift,
        axis,
        phase_override,
    )
    q_d = x[6:9]
    w_d = float(np.sqrt(1.0 + q_d @ h @ q_d))
    velocity_d = q_d / w_d
    transport_d = lapse * velocity_d[axis] - shift[axis]
    path = np.asarray(
        (fields[2], fields[3], fields[4], transport_d),
        dtype=float,
    )
    return (
        np.asarray(fields[0], dtype=float),
        np.asarray(fields[1], dtype=float),
        path,
    )


def reference_full_carter_symbol(
    solver: Theory33ProductionSolver,
    primitive: Array,
    geometry: Array,
    *,
    axis: int,
) -> dict[str, Array | float | bool]:
    """Numerical CPU flux/path symbol with the hard phase held fixed."""
    x = np.asarray(primitive, dtype=float)
    geometric = np.asarray(geometry, dtype=float)
    h = geometric[:9].reshape(3, 3)
    production_order = np.concatenate(
        ([x[0]], x[2:5], [x[5]], x[6:9])
    )
    phase = solver._cell_phase(
        production_order, h, float(x[1])
    )
    base_u, _, base_path = reference_full_principal(
        solver,
        x,
        geometric,
        axis=axis,
        phase_override=phase,
    )
    time_jacobian = np.empty((9, 9))
    flux_path_jacobian = np.empty((9, 9))
    for column in range(9):
        step = 2.0e-6 * max(1.0, abs(float(x[column])))
        plus = x.copy()
        minus = x.copy()
        plus[column] += step
        minus[column] -= step
        plus_u, plus_flux, plus_path = reference_full_principal(
            solver,
            plus,
            geometric,
            axis=axis,
            phase_override=phase,
        )
        minus_u, minus_flux, minus_path = reference_full_principal(
            solver,
            minus,
            geometric,
            axis=axis,
            phase_override=phase,
        )
        time_jacobian[:, column] = (
            plus_u - minus_u
        ) / (2.0 * step)
        flux_path_jacobian[:, column] = (
            plus_flux - minus_flux
        ) / (2.0 * step)
        chi_zero_derivative = (
            plus_path[0] - minus_path[0]
        ) / (2.0 * step)
        chi_axis_derivative = (
            plus_path[1] - minus_path[1]
        ) / (2.0 * step)
        flux_path_jacobian[6 + axis, column] -= base_path[2] * (
            chi_zero_derivative
            + base_path[3] * chi_axis_derivative
        )
    symbol = flux_path_jacobian @ np.linalg.inv(time_jacobian)
    eigenvalues = np.linalg.eigvals(symbol)
    order = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[order]
    return {
        "conserved": base_u,
        "symbol": symbol,
        "eigenvalues": eigenvalues,
        "maximum_imaginary_part": float(
            np.max(np.abs(eigenvalues.imag))
        ),
        "phase_two": bool(phase) if phase is not None else False,
    }


def _classify_full_primitive(
    solver: Theory33ProductionSolver,
    primitive: Array,
) -> bool | None:
    x = np.asarray(primitive, dtype=float)
    production_order = np.concatenate(
        ([x[0]], x[2:5], [x[5]], x[6:9])
    )
    return solver._cell_phase(
        production_order, np.eye(3), float(x[1])
    )


def reference_drag_step(
    solver: Theory33ProductionSolver,
    primitive: Array,
    *,
    dt: float,
    relaxation_time: float,
    lapse: float = 1.0,
) -> tuple[Array, dict[str, float]]:
    """Float64 CPU Newton oracle for local entropy-producing carrier drag."""
    initial = np.asarray(primitive, dtype=float).copy()
    before = full_material_prediction(solver, initial)
    q_n = initial[2:5]
    q_d = initial[6:9]
    w_n = float(np.sqrt(1.0 + q_n @ q_n))
    w_d = float(np.sqrt(1.0 + q_d @ q_d))
    old_gamma = max(float(w_n * w_d - q_n @ q_d), 1.0)
    if old_gamma - 1.0 <= 1.0e-3:
        return initial, {
            "relative_gamma_before": old_gamma,
            "relative_gamma_after": old_gamma,
            "entropy_increase": 0.0,
            "residual": 0.0,
            "phase_crossings": 0.0,
        }
    decay = float(
        np.exp(-dt * max(lapse, 0.0) / relaxation_time)
    )
    target = np.concatenate(
        (before[:6], decay * (q_d - q_n))
    )
    energy_scale = max(
        abs(float(before[5])),
        float(np.linalg.norm(before[2:5])),
        1.0e-10,
    )
    scale = np.asarray(
        [
            max(abs(float(before[0])), 1.0e-10),
            max(abs(float(before[1])), 1.0e-10),
            energy_scale,
            energy_scale,
            energy_scale,
            energy_scale,
            1.0,
            1.0,
            1.0,
        ]
    )
    working = initial.copy()
    phase_override = _classify_full_primitive(solver, working)
    phase_crossings = 0
    final_norm = np.inf
    for _ in range(3):
        def function(value: Array) -> Array:
            return np.concatenate(
                (
                    full_material_prediction(
                        solver, value, phase_override
                    )[:6],
                    value[6:9] - value[2:5],
                )
            )

        for _ in range(32):
            residual = (function(working) - target) / scale
            final_norm = float(np.max(np.abs(residual)))
            if final_norm < 2.0e-10:
                break
            jacobian = np.empty((9, 9))
            for column in range(9):
                step = 2.0e-6 * max(
                    1.0, abs(float(working[column]))
                )
                plus = working.copy()
                minus = working.copy()
                plus[column] += step
                minus[column] -= step
                jacobian[:, column] = (
                    function(plus) - function(minus)
                ) / (2.0 * step) / scale
            delta = np.linalg.solve(jacobian, -residual)
            for power in range(14):
                trial = working + (0.5**power) * delta
                trial[0] = np.clip(trial[0], -700.0, 4.0)
                trial[1] = np.clip(trial[1], -20.0, 20.0)
                trial[5] = np.clip(trial[5], -700.0, 4.0)
                trial_norm = float(
                    np.max(np.abs((function(trial) - target) / scale))
                )
                if trial_norm < final_norm:
                    working = trial
                    break
            else:
                break
        final_phase = _classify_full_primitive(solver, working)
        if phase_override is None or final_phase == phase_override:
            break
        phase_override = final_phase
        phase_crossings += 1
    after = full_material_prediction(
        solver, working, phase_override
    )
    final = np.concatenate(
        (after[:6], working[6:9] - working[2:5])
    )
    final_norm = float(np.max(np.abs((final - target) / scale)))
    q_n_after = working[2:5]
    q_d_after = working[6:9]
    w_n_after = float(np.sqrt(1.0 + q_n_after @ q_n_after))
    w_d_after = float(np.sqrt(1.0 + q_d_after @ q_d_after))
    new_gamma = max(
        float(w_n_after * w_d_after - q_n_after @ q_d_after),
        1.0,
    )
    return working, {
        "relative_gamma_before": old_gamma,
        "relative_gamma_after": new_gamma,
        "entropy_increase": float(working[1] - initial[1]),
        "residual": final_norm,
        "phase_crossings": float(phase_crossings),
    }


def reference_vector_stress(
    solver: Theory33ProductionSolver,
    primitive: Array,
    vector_input: Array,
) -> Array:
    """Independent flat-metric mixed-Proca 3+1 stress calculation."""
    values = np.asarray(vector_input, dtype=float)
    material = full_material_prediction(solver, primitive)
    a = values[0:3]
    pi_a = values[3:6]
    b = values[6:9]
    pi_b = values[9:12]
    right_a = float(values[12])
    right_b = float(values[13] + material[1])
    phi_a = (8.0 * right_a + 4.0 * right_b) / 3.0
    phi_b = (4.0 * right_a + 8.0 * right_b) / 3.0

    def single(
        potential: Array,
        canonical: Array,
        magnetic: Array,
        phi: float,
    ) -> tuple[float, Array, Array]:
        electric = -canonical
        field = np.zeros((3, 3))
        field[1, 2] = magnetic[0]
        field[2, 1] = -magnetic[0]
        field[2, 0] = magnetic[1]
        field[0, 2] = -magnetic[1]
        field[0, 1] = magnetic[2]
        field[1, 0] = -magnetic[2]
        e2 = float(electric @ electric)
        a2 = float(potential @ potential)
        f2 = float(np.sum(field * field))
        mass2 = 0.25
        rho = 0.5 * e2 + 0.25 * f2 + 0.5 * mass2 * (
            phi**2 + a2
        )
        momentum = electric @ field.T + mass2 * phi * potential
        stress = (
            -np.outer(electric, electric)
            + field @ field.T
            + np.eye(3) * (0.5 * e2 - 0.25 * f2)
            + mass2
            * (
                np.outer(potential, potential)
                - 0.5 * np.eye(3) * (a2 - phi**2)
            )
        )
        return rho, momentum, stress

    rho_a, momentum_a, stress_a = single(
        a, pi_a, values[14:17], phi_a
    )
    rho_b, momentum_b, stress_b = single(
        b, pi_b, values[17:20], phi_b
    )
    difference = a - b
    phi_difference = phi_a - phi_b
    mixing_rho = 0.125 * (
        phi_difference**2 + difference @ difference
    )
    mixing_momentum = 0.25 * phi_difference * difference
    mixing_stress = 0.25 * (
        np.outer(difference, difference)
        - 0.5
        * np.eye(3)
        * (difference @ difference - phi_difference**2)
    )
    return np.concatenate(
        (
            [rho_a + rho_b + mixing_rho],
            momentum_a + momentum_b + mixing_momentum,
            (stress_a + stress_b + mixing_stress).reshape(-1),
            [phi_a, phi_b],
        )
    )


def reference_curved_vector_stress(
    solver: Theory33ProductionSolver,
    primitive: Array,
    vector_input: Array,
    geometry: Array,
) -> Array:
    """Independent curved-metric mixed-Proca stress oracle."""
    x = np.asarray(primitive, dtype=float)
    values = np.asarray(vector_input, dtype=float)
    h = np.asarray(geometry[:9], dtype=float).reshape(3, 3)
    h_inv = np.linalg.inv(h)
    sqrt_h = float(np.sqrt(np.linalg.det(h)))
    d = float(np.exp(x[5]))
    q_d = x[6:9]
    w_d = float(np.sqrt(1.0 + q_d @ h @ q_d))
    target_charge = d * w_d
    right_a = float(values[12] / sqrt_h)
    right_b = float(values[13] / sqrt_h + target_charge)
    phi_a = (8.0 * right_a + 4.0 * right_b) / 3.0
    phi_b = (4.0 * right_a + 8.0 * right_b) / 3.0

    def single(
        potential: Array,
        pi: Array,
        magnetic: Array,
        phi: float,
    ) -> tuple[float, Array, Array]:
        electric_up = -pi / sqrt_h
        electric_down = h @ electric_up
        potential_up = h_inv @ potential
        field = np.zeros((3, 3))
        field[1, 2] = magnetic[0]
        field[2, 1] = -magnetic[0]
        field[2, 0] = magnetic[1]
        field[0, 2] = -magnetic[1]
        field[0, 1] = magnetic[2]
        field[1, 0] = -magnetic[2]
        field_up = h_inv @ field @ h_inv
        electric_squared = float(electric_down @ electric_up)
        field_squared = float(np.sum(field * field_up))
        potential_squared = float(potential @ potential_up)
        rho = (
            0.5 * electric_squared
            + 0.25 * field_squared
            + 0.125 * (phi**2 + potential_squared)
        )
        momentum = electric_up @ field.T + 0.25 * phi * potential
        magnetic_tensor = field @ h_inv @ field.T
        stress = (
            -np.outer(electric_down, electric_down)
            + magnetic_tensor
            + h * (0.5 * electric_squared - 0.25 * field_squared)
            + 0.25
            * (
                np.outer(potential, potential)
                - 0.5 * h * (potential_squared - phi**2)
            )
        )
        return rho, momentum, stress

    a = values[0:3]
    b = values[6:9]
    rho_a, momentum_a, stress_a = single(
        a, values[3:6], values[14:17], phi_a
    )
    rho_b, momentum_b, stress_b = single(
        b, values[9:12], values[17:20], phi_b
    )
    difference = a - b
    difference_squared = float(difference @ h_inv @ difference)
    phi_difference = phi_a - phi_b
    mixing_rho = 0.125 * (
        phi_difference**2 + difference_squared
    )
    mixing_momentum = 0.25 * phi_difference * difference
    mixing_stress = 0.25 * (
        np.outer(difference, difference)
        - 0.5 * h * (difference_squared - phi_difference**2)
    )
    return np.concatenate(
        (
            [rho_a + rho_b + mixing_rho],
            momentum_a + momentum_b + mixing_momentum,
            (stress_a + stress_b + mixing_stress).reshape(-1),
            [phi_a, phi_b],
        )
    )


def compose_full_carter_state(
    solver: Theory33ProductionSolver,
    primitive: Array,
    vector_stress: Array,
    tracer: float,
) -> Array:
    material = full_material_prediction(solver, primitive)
    result = np.empty(11)
    result[:2] = material[:2]
    result[2:5] = material[2:5] + vector_stress[1:4]
    result[5] = material[5] + vector_stress[0]
    result[6:9] = material[6:9]
    result[9] = material[0] * primitive[1]
    result[10] = material[0] * tracer
    return result


def qualify_vulkan_mixed_sources(
    *,
    points: int = 12,
    seed: int = 33,
) -> dict[str, object]:
    """Qualify mixed-vector stress and stiff drag against CPU float64."""
    coordinate = (np.arange(points, dtype=float) + 0.5) / points
    gamma_ad = 5.0 / 3.0
    density = 0.10 + 0.012 * np.sin(2.0 * np.pi * coordinate)
    internal = 0.12 + 0.02 * np.cos(2.0 * np.pi * coordinate)
    pressure = (gamma_ad - 1.0) * density * internal
    entropy = np.log(pressure / density**gamma_ad) / (
        gamma_ad - 1.0
    )
    primitive = np.zeros((points, 9))
    primitive[:, 0] = np.log(density)
    primitive[:, 1] = entropy
    primitive[:, 2] = 0.015 * np.sin(2.0 * np.pi * coordinate)
    primitive[:, 3] = -0.01 * np.cos(2.0 * np.pi * coordinate)
    primitive[:, 4] = 0.006 * np.sin(4.0 * np.pi * coordinate)
    primitive[:, 5] = np.log(
        0.005 + 0.0004 * np.cos(2.0 * np.pi * coordinate)
    )
    relative_velocity = np.column_stack(
        (
            0.18 + 0.22 * np.sin(2.0 * np.pi * coordinate) ** 2,
            0.08 * np.cos(2.0 * np.pi * coordinate),
            -0.05 * np.sin(2.0 * np.pi * coordinate),
        )
    )
    relative_speed_squared = np.sum(relative_velocity**2, axis=1)
    relative_q = relative_velocity / np.sqrt(
        1.0 - relative_speed_squared
    )[:, None]
    primitive[:, 6:9] = primitive[:, 2:5] + relative_q
    rng = np.random.default_rng(seed)
    vector_input = rng.normal(0.0, 0.025, (points, 20))
    tracer = 0.2 + 0.05 * np.sin(2.0 * np.pi * coordinate)
    dt = 0.1
    relaxation_time = 0.01

    modes: list[dict[str, object]] = []
    for mode in ("analytic_m1", "qualified_frozen"):
        solver = _source_solver(mode)
        backend = VulkanTheory33SourceBackend(
            points, constitutive_model=mode
        )
        snapshot = backend.evaluate(
            primitive,
            vector_input,
            tracer,
            dt=dt,
            relaxation_time=relaxation_time,
        )
        reference_vector = np.stack(
            [
                reference_vector_stress(
                    solver, primitive[index], vector_input[index]
                )
                for index in range(points)
            ]
        )
        reference_before = np.stack(
            [
                compose_full_carter_state(
                    solver,
                    primitive[index],
                    reference_vector[index],
                    float(tracer[index]),
                )
                for index in range(points)
            ]
        )
        reference_after_primitive = np.empty_like(primitive)
        reference_drag_diagnostics: list[dict[str, float]] = []
        for index in range(points):
            (
                reference_after_primitive[index],
                reference_diagnostic,
            ) = reference_drag_step(
                solver,
                primitive[index],
                dt=dt,
                relaxation_time=relaxation_time,
            )
            reference_drag_diagnostics.append(reference_diagnostic)
        reference_after = np.stack(
            [
                compose_full_carter_state(
                    solver,
                    reference_after_primitive[index],
                    reference_vector[index],
                    float(tracer[index]),
                )
                for index in range(points)
            ]
        )
        physical_delta = np.abs(
            snapshot.conserved_after[:, :6]
            - snapshot.conserved_before[:, :6]
        )
        old_excess = np.maximum(
            snapshot.drag_field("relative_lorentz_factor_before") - 1.0,
            1.0e-30,
        )
        new_excess = np.maximum(
            snapshot.drag_field("relative_lorentz_factor_after") - 1.0,
            0.0,
        )
        modes.append(
            {
                "constitutive_model": mode,
                "maximum_vector_stress_absolute_error": float(
                    np.max(
                        np.abs(snapshot.vector_stress - reference_vector)
                    )
                ),
                "maximum_composed_state_absolute_error": float(
                    np.max(
                        np.abs(
                            snapshot.conserved_before - reference_before
                        )
                    )
                ),
                "maximum_drag_primitive_absolute_error": float(
                    np.max(
                        np.abs(
                            snapshot.primitive_after_drag
                            - reference_after_primitive
                        )
                    )
                ),
                "maximum_dragged_state_absolute_error": float(
                    np.max(
                        np.abs(snapshot.conserved_after - reference_after)
                    )
                ),
                "maximum_gpu_material_conservation_error": float(
                    np.max(physical_delta)
                ),
                "maximum_cpu_drag_residual": float(
                    max(
                        diagnostic["residual"]
                        for diagnostic in reference_drag_diagnostics
                    )
                ),
                "maximum_gpu_drag_residual": float(
                    np.max(
                        snapshot.drag_field(
                            "scaled_conservation_residual"
                        )
                    )
                ),
                "minimum_entropy_increase": float(
                    np.min(
                        snapshot.drag_field(
                            "specific_entropy_increase"
                        )
                    )
                ),
                "maximum_relative_gamma_excess_ratio": float(
                    np.max(new_excess / old_excess)
                ),
                "invalid_drag_points": int(
                    np.count_nonzero(
                        snapshot.drag_field("valid") < 0.5
                    )
                ),
                "gpu_phase_crossings": int(
                    np.sum(snapshot.drag_field("phase_crossings"))
                ),
                "cpu_phase_crossings": int(
                    sum(
                        diagnostic["phase_crossings"]
                        for diagnostic in reference_drag_diagnostics
                    )
                ),
            }
        )

    quiet = primitive.copy()
    quiet[:, 6:9] = quiet[:, 2:5]
    quiet_backend = VulkanTheory33SourceBackend(points)
    quiet_snapshot = quiet_backend.evaluate(
        quiet,
        np.zeros_like(vector_input),
        tracer,
        dt=dt,
        relaxation_time=relaxation_time,
    )
    quiet_error = float(
        np.max(
            np.abs(
                quiet_snapshot.primitive_after_drag
                - quiet_snapshot.primitive_before
            )
        )
    )
    qualification = {
        "mixed_vector_stress_matches_cpu": all(
            item["maximum_vector_stress_absolute_error"] < 2.0e-6
            for item in modes
        ),
        "full_state_composition_matches_cpu": all(
            item["maximum_composed_state_absolute_error"] < 2.0e-6
            for item in modes
        ),
        "implicit_drag_matches_cpu": all(
            item["maximum_drag_primitive_absolute_error"] < 8.0e-5
            and item["maximum_dragged_state_absolute_error"] < 8.0e-5
            for item in modes
        ),
        "material_energy_momentum_conserved": all(
            item["maximum_gpu_material_conservation_error"] < 2.0e-6
            and item["maximum_gpu_drag_residual"] < 5.0e-4
            for item in modes
        ),
        "drag_is_entropy_producing": all(
            item["minimum_entropy_increase"] >= -2.0e-5
            for item in modes
        ),
        "stiff_relative_motion_relaxes": all(
            item["maximum_relative_gamma_excess_ratio"] < 2.0e-3
            for item in modes
        ),
        "phase_transition_tape_matches_cpu": (
            modes[1]["gpu_phase_crossings"]
            == modes[1]["cpu_phase_crossings"]
            and modes[1]["gpu_phase_crossings"] > 0
        ),
        "all_drag_solves_valid": all(
            item["invalid_drag_points"] == 0 for item in modes
        ),
        "comoving_threshold_is_exactly_quiet": quiet_error < 1.0e-7,
    }
    return {
        "schema": "vulkan-tesseract.mixed-sources.v1",
        "backend": "Slang/Vulkan",
        "primitive_block": list(
            (
                "log_n",
                "entropy",
                "q_N_x",
                "q_N_y",
                "q_N_z",
                "log_d",
                "q_D_x",
                "q_D_y",
                "q_D_z",
            )
        ),
        "conserved_block": (
            "D_N,D_D,S_x,S_y,S_z,E,P_D_x,P_D_y,P_D_z,"
            "D_N*s,D_N*Y"
        ),
        "points": points,
        "dt": dt,
        "relaxation_time": relaxation_time,
        "mode_audits": modes,
        "comoving_primitive_error": quiet_error,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def qualify_vulkan_mixed_proca_evolution(
    *,
    shape: tuple[int, int] = (8, 8),
    dt: float = 2.0e-4,
    steps: int = 4,
) -> dict[str, object]:
    """Qualify evolved resident mixed-Proca fields against Theory 3."""
    from .theory3 import Theory3System

    nx, ny = shape
    points = nx * ny
    grid = PeriodicGrid(shape, (1.0, 1.0), method="fd2")
    x, y = grid.coordinates()
    wave = 2.0 * np.pi

    def flatten(value: Array) -> Array:
        return np.asarray(value).reshape(-1, order="F")

    def pack(
        a: Array,
        pi_a: Array,
        b: Array,
        pi_b: Array,
        long_a: Array,
        long_b: Array,
        clean_a: Array,
        clean_b: Array,
    ) -> Array:
        packed = np.zeros((points, 16), dtype=np.float64)
        for axis in range(3):
            packed[:, axis] = flatten(a[axis])
            packed[:, 3 + axis] = flatten(pi_a[axis])
            packed[:, 6 + axis] = flatten(b[axis])
            packed[:, 9 + axis] = flatten(pi_b[axis])
        packed[:, 12] = flatten(long_a)
        packed[:, 13] = flatten(long_b)
        packed[:, 14] = flatten(clean_a)
        packed[:, 15] = flatten(clean_b)
        return packed

    def unpack(packed: Array) -> tuple[Array, ...]:
        def field(column: int) -> Array:
            return np.asarray(packed[:, column], dtype=float).reshape(
                shape, order="F"
            )

        return (
            np.stack(tuple(field(axis) for axis in range(3))),
            np.stack(tuple(field(3 + axis) for axis in range(3))),
            np.stack(tuple(field(6 + axis) for axis in range(3))),
            np.stack(tuple(field(9 + axis) for axis in range(3))),
            field(12),
            field(13),
            field(14),
            field(15),
        )

    h = grid.zeros((3, 3))
    h[0, 0] = h[1, 1] = h[2, 2] = 1.0
    lapse = 1.0 + 0.01 * np.sin(wave * (x - y))
    shift = grid.zeros((3,))
    shift[0] = 0.004 * np.cos(wave * y)
    shift[1] = -0.003 * np.sin(wave * x)
    geometry = np.zeros((points, 13), dtype=np.float32)
    geometry[:, :9] = np.eye(3, dtype=np.float32).reshape(1, 9)
    geometry[:, 9] = flatten(lapse)
    geometry[:, 10] = flatten(shift[0])
    geometry[:, 11] = flatten(shift[1])
    a = grid.zeros((3,))
    pi_a = grid.zeros((3,))
    b = grid.zeros((3,))
    pi_b = grid.zeros((3,))
    a[0] = 8.0e-4 * np.sin(wave * y)
    a[1] = 7.0e-4 * np.cos(wave * x)
    a[2] = 5.0e-4 * np.sin(wave * (x + y))
    b[0] = -6.0e-4 * np.cos(wave * y)
    b[1] = 9.0e-4 * np.sin(wave * x)
    b[2] = 4.0e-4 * np.cos(wave * (x - y))
    pi_a[0] = 5.0e-4 * np.cos(wave * x)
    pi_a[1] = -4.0e-4 * np.sin(wave * y)
    pi_a[2] = 3.0e-4 * np.cos(wave * (x + y))
    pi_b[0] = -4.0e-4 * np.sin(wave * x)
    pi_b[1] = 6.0e-4 * np.cos(wave * y)
    pi_b[2] = -2.0e-4 * np.sin(wave * (x - y))
    system = Theory3System(grid)
    long_a, long_b = system.initial_longitudinal_charges(
        h, pi_a, pi_b
    )
    clean_a = 1.0e-4 * np.sin(wave * (x + y))
    clean_b = -8.0e-5 * np.cos(wave * (x - y))
    initial = pack(
        a, pi_a, b, pi_b, long_a, long_b, clean_a, clean_b
    )
    density_d = 0.005 * (
        1.0 + 0.01 * np.cos(wave * (x + y))
    )
    q_d = grid.zeros((3,))
    q_d[0] = 0.04 + 0.003 * np.sin(wave * y)
    q_d[1] = -0.02 + 0.002 * np.cos(wave * x)
    target_current = density_d[None, ...] * q_d
    matter = np.zeros((points, 11), dtype=np.float32)
    matter[:, 0] = flatten(0.1)
    matter[:, 1] = flatten(density_d * np.sqrt(
        1.0 + np.sum(q_d * q_d, axis=0)
    ))
    primitive = np.zeros((points, 9), dtype=np.float32)
    primitive[:, 0] = np.log(0.1)
    primitive[:, 1] = -2.0
    primitive[:, 5] = flatten(np.log(density_d))
    for axis in range(3):
        primitive[:, 6 + axis] = flatten(q_d[axis])

    def cpu_rhs(packed: Array) -> Array:
        values = unpack(packed)
        result = system.rhs_fields(
            h,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            matter[:, 1].reshape(shape, order="F"),
            target_current,
            lapse,
            shift,
            1.0,
        )
        return pack(*result)

    backend = VulkanMixedProcaBackend(
        shape, cleaning_damping=1.0
    )
    state = backend.initialize(
        initial.astype(np.float32), matter, primitive, geometry
    )
    gpu_initial = backend.snapshot(state)
    oracle_rhs = cpu_rhs(initial)
    rhs_error = np.abs(gpu_initial.rhs.astype(float) - oracle_rhs)
    magnetic_a = system.magnetic_field(h, a)
    magnetic_b = system.magnetic_field(h, b)
    oracle_magnetic = np.stack(
        (
            *(flatten(magnetic_a[axis]) for axis in range(3)),
            *(flatten(magnetic_b[axis]) for axis in range(3)),
        ),
        axis=1,
    )
    magnetic_error = float(
        np.max(
            np.abs(
                gpu_initial.vector_input[:, 14:20]
                - oracle_magnetic
            )
        )
    )

    cpu_state = initial.copy()
    reads_before = state.host_reads
    for _ in range(steps):
        first = cpu_state + dt * cpu_rhs(cpu_state)
        second_euler = first + dt * cpu_rhs(first)
        second = 0.75 * cpu_state + 0.25 * second_euler
        third_euler = second + dt * cpu_rhs(second)
        cpu_state = (
            (1.0 / 3.0) * cpu_state
            + (2.0 / 3.0) * third_euler
        )
        backend.step(state, dt)
    reads_after = state.host_reads
    final = backend.snapshot(state)
    evolution_error = np.abs(final.state.astype(float) - cpu_state)
    gpu_values = unpack(final.state)
    gauss_a, gauss_b = system.gauss_constraints(
        h,
        gpu_values[1],
        gpu_values[3],
        gpu_values[4],
        gpu_values[5],
    )
    report: dict[str, object] = {
        "shape": list(shape),
        "steps": steps,
        "dt": dt,
        "maximum_rhs_absolute_error": float(np.max(rhs_error)),
        "rhs_rms_error": float(np.sqrt(np.mean(rhs_error**2))),
        "maximum_magnetic_absolute_error": magnetic_error,
        "maximum_evolution_absolute_error": float(
            np.max(evolution_error)
        ),
        "evolution_rms_error": float(
            np.sqrt(np.mean(evolution_error**2))
        ),
        "maximum_gauss_constraint": float(
            max(np.max(np.abs(gauss_a)), np.max(np.abs(gauss_b)))
        ),
        "host_reads_during_evolution": int(
            reads_after - reads_before
        ),
    }
    report["qualification"] = {
        "mixed_proca_rhs_matches_float64_oracle": (
            report["maximum_rhs_absolute_error"] < 2.0e-5
        ),
        "magnetic_fields_are_derived_on_device": (
            report["maximum_magnetic_absolute_error"] < 2.0e-6
        ),
        "ssprk3_evolution_matches_float64_oracle": (
            report["maximum_evolution_absolute_error"] < 2.0e-6
        ),
        "gauss_constraints_remain_controlled": (
            report["maximum_gauss_constraint"] < 2.0e-3
        ),
        "evolution_is_device_resident": (
            report["host_reads_during_evolution"] == 0
        ),
    }
    report.update(
        {
            "schema": "vulkan-tesseract.mixed-proca-evolution.v1",
            "backend": "Slang/Vulkan",
            "dynamic_fields": (
                "a_i,pi_A^i,b_i,pi_B^i,Q_A,Q_B,clean_A,clean_B"
            ),
            "qualified": all(report["qualification"].values()),
        }
    )
    return report


def qualify_vulkan_evolved_coupled_stack(
    *,
    points: int = 8,
    dt: float = 1.0e-7,
) -> dict[str, object]:
    """Compare one fully coupled evolved-vector step with the CPU engine."""
    grid = PeriodicGrid((points,), (1.0,), method="fd2")
    coordinate = grid.coordinates()[0]
    wave = 2.0 * np.pi
    solver = Theory33ProductionSolver(
        grid,
        production=Theory3ProductionParameters(
            flux_reconstruction="piecewise_constant",
            source_splitting="strang",
        ),
    )
    fluid = FluidPrimitive(
        np.full(points, 0.1),
        np.full(points, 0.12),
        grid.zeros((3,)),
        grid.zeros(),
    )
    carrier = CarrierPrimitive(
        np.full(points, 0.005), grid.zeros((3,))
    )
    a = grid.zeros((3,))
    pi_a = grid.zeros((3,))
    b = grid.zeros((3,))
    pi_b = grid.zeros((3,))
    a[1] = 7.0e-4 * np.sin(wave * coordinate)
    a[2] = 5.0e-4 * np.cos(wave * coordinate)
    b[1] = -6.0e-4 * np.cos(wave * coordinate)
    b[2] = 4.0e-4 * np.sin(wave * coordinate)
    pi_a[0] = 3.0e-4 * np.sin(wave * coordinate)
    pi_a[1] = 2.0e-4 * np.cos(wave * coordinate)
    pi_b[0] = -2.5e-4 * np.cos(wave * coordinate)
    pi_b[2] = 1.5e-4 * np.sin(wave * coordinate)
    cpu_state = solver.initialize(
        solver.ccz4.flat_state(),
        fluid,
        carrier=carrier,
        a=a,
        pi_A=pi_a,
        b=b,
        pi_B=pi_b,
    )

    def pack_matter(state) -> Array:
        return np.stack(
            (
                state.matter.D,
                state.target_charge,
                state.matter.momentum[0],
                state.matter.momentum[1],
                state.matter.momentum[2],
                state.matter.energy,
                state.target_current[0],
                state.target_current[1],
                state.target_current[2],
                state.matter.entropy,
                state.matter.tracer,
            ),
            axis=1,
        )

    def pack_vector(state) -> Array:
        packed = np.zeros((points, 16), dtype=np.float64)
        packed[:, 0:3] = state.a.T
        packed[:, 3:6] = state.pi_A.T
        packed[:, 6:9] = state.b.T
        packed[:, 9:12] = state.pi_B.T
        packed[:, 12] = state.longitudinal_A
        packed[:, 13] = state.longitudinal_B
        packed[:, 14] = state.cleaning_A
        packed[:, 15] = state.cleaning_B
        return packed

    def pack_geometry(state: CCZ4State) -> Array:
        packed = np.zeros((points, 31), dtype=np.float64)
        for row in range(3):
            for column in range(3):
                packed[:, row * 3 + column] = (
                    state.conformal_metric[row, column]
                )
                packed[:, 9 + row * 3 + column] = (
                    state.conformal_A[row, column]
                )
        packed[:, 18] = state.conformal_factor
        packed[:, 19] = state.trace_K
        packed[:, 20] = state.theta
        packed[:, 24] = state.lapse
        for axis in range(3):
            packed[:, 21 + axis] = state.gamma_hat[axis]
            packed[:, 25 + axis] = state.shift[axis]
            packed[:, 28 + axis] = state.shift_driver[axis]
        return packed

    master = solver.master
    primitive = np.zeros((points, 9), dtype=np.float32)
    primitive[:, 0] = np.log(0.1)
    primitive[:, 1] = master.specific_entropy(
        np.full(points, 0.1), np.full(points, 0.12)
    )
    primitive[:, 5] = np.log(0.005)
    vector_dynamic = pack_vector(cpu_state).astype(np.float32)
    geometry = pack_geometry(cpu_state.geometry).astype(np.float32)
    gpu = VulkanTesseractEvolvedBackend(
        (points,),
        constitutive_model="analytic_m1",
        reconstruction="piecewise_constant",
        relaxation_time=1.0,
        kappa_gravity=solver.ccz4.parameters.kappa_gravity,
        kappa1=solver.ccz4.parameters.kappa1,
        kappa2=solver.ccz4.parameters.kappa2,
        kappa3=solver.ccz4.parameters.kappa3,
        cosmological_constant=(
            solver.ccz4.parameters.cosmological_constant
        ),
        shift_speed=solver.ccz4.parameters.shift_speed,
        eta_shift=solver.ccz4.parameters.eta_shift,
        slicing=solver.ccz4.parameters.slicing,
        evolve_shift=solver.ccz4.parameters.evolve_shift,
        vector_cleaning_damping=(
            solver.production_parameters.cleaning_damping
        ),
    )
    gpu_state = gpu.initialize(
        primitive,
        vector_dynamic,
        geometry,
        np.zeros(points, dtype=np.float32),
    )
    initial_gpu = gpu.snapshot(gpu_state)
    initial_matter_error = float(
        np.max(
            np.abs(
                initial_gpu.coupled.matter.state
                - pack_matter(cpu_state)
            )
        )
    )
    reads_before = gpu_state.host_reads
    cpu_final = solver.step(cpu_state, dt)
    gpu.step(gpu_state, dt, apply_drag=True)
    reads_after = gpu_state.host_reads
    final_gpu = gpu.snapshot(gpu_state)
    matter_error = float(
        np.max(
            np.abs(
                final_gpu.coupled.matter.state
                - pack_matter(cpu_final)
            )
        )
    )
    vector_error = float(
        np.max(
            np.abs(
                final_gpu.vector_state - pack_vector(cpu_final)
            )
        )
    )
    geometry_error = float(
        np.max(
            np.abs(
                final_gpu.coupled.ccz4_state
                - pack_geometry(cpu_final.geometry)
            )
        )
    )
    vector_change = float(
        np.max(np.abs(final_gpu.vector_state - vector_dynamic))
    )
    report: dict[str, object] = {
        "points": points,
        "dt": dt,
        "initial_matter_absolute_error": initial_matter_error,
        "matter_step_absolute_error": matter_error,
        "vector_step_absolute_error": vector_error,
        "geometry_step_absolute_error": geometry_error,
        "maximum_vector_state_change": vector_change,
        "maximum_recovery_residual": float(
            np.max(final_gpu.coupled.matter.diagnostics[:, 0])
        ),
        "invalid_recoveries": int(
            np.count_nonzero(
                final_gpu.coupled.matter.diagnostics[:, 1] < 0.5
            )
        ),
        "host_reads_during_evolution": int(
            reads_after - reads_before
        ),
    }
    report["qualification"] = {
        "initial_total_state_matches_cpu": (
            report["initial_matter_absolute_error"] < 2.0e-6
        ),
        "matter_step_matches_cpu": (
            report["matter_step_absolute_error"] < 3.0e-6
        ),
        "vector_step_matches_cpu": (
            report["vector_step_absolute_error"] < 3.0e-6
        ),
        "geometry_step_matches_cpu": (
            report["geometry_step_absolute_error"] < 3.0e-6
        ),
        "vector_state_is_dynamical": (
            report["maximum_vector_state_change"] > 1.0e-10
        ),
        "recovery_valid": report["invalid_recoveries"] == 0,
        "step_is_device_resident": (
            report["host_reads_during_evolution"] == 0
        ),
    }
    report.update(
        {
            "schema": "vulkan-tesseract.evolved-coupled-stack.v1",
            "backend": "Slang/Vulkan",
            "schedule": (
                "CCZ4/Proca/Carter/Proca/CCZ4 symmetric split"
            ),
            "qualified": all(report["qualification"].values()),
        }
    )
    return report


def qualify_vulkan_multidimensional_full_stack(
    *,
    shape: tuple[int, int] = (8, 8),
    steps: int = 2,
    seed: int = 3344,
    derivative_order: int = 2,
) -> dict[str, object]:
    """Qualify resident curved Carter evolution and the full CCZ4 operator."""
    if min(shape) < 4 or steps < 1:
        raise ValueError("qualification needs at least 4x4 cells and one step")
    if derivative_order not in {2, 4}:
        raise ValueError("derivative_order must be 2 or 4")
    if derivative_order == 4 and min(shape) < 8:
        raise ValueError("fourth-order qualification needs 8x8 cells")
    nx, ny = shape
    points = nx * ny
    grid = PeriodicGrid(
        shape, (1.0, 1.0), method=f"fd{derivative_order}"
    )
    x, y = grid.coordinates()
    wave = 2.0 * np.pi

    def flatten(values: Array) -> Array:
        return np.asarray(values).reshape(-1, order="F")

    def pack_ccz4(state: CCZ4State) -> Array:
        packed = np.zeros((points, 31), dtype=np.float32)
        for row in range(3):
            for column in range(3):
                packed[:, row * 3 + column] = flatten(
                    state.conformal_metric[row, column]
                )
                packed[:, 9 + row * 3 + column] = flatten(
                    state.conformal_A[row, column]
                )
        packed[:, 18] = flatten(state.conformal_factor)
        packed[:, 19] = flatten(state.trace_K)
        packed[:, 20] = flatten(state.theta)
        packed[:, 24] = flatten(state.lapse)
        for axis in range(3):
            packed[:, 21 + axis] = flatten(state.gamma_hat[axis])
            packed[:, 25 + axis] = flatten(state.shift[axis])
            packed[:, 28 + axis] = flatten(
                state.shift_driver[axis]
            )
        return packed

    def unpack_ccz4(packed: Array) -> CCZ4State:
        def field(column: int) -> Array:
            return np.asarray(packed[:, column], dtype=float).reshape(
                shape, order="F"
            )

        metric = grid.zeros((3, 3))
        conformal_a = grid.zeros((3, 3))
        gamma_hat = grid.zeros((3,))
        shift = grid.zeros((3,))
        driver = grid.zeros((3,))
        for row in range(3):
            for column in range(3):
                metric[row, column] = field(row * 3 + column)
                conformal_a[row, column] = field(
                    9 + row * 3 + column
                )
        for axis in range(3):
            gamma_hat[axis] = field(21 + axis)
            shift[axis] = field(25 + axis)
            driver[axis] = field(28 + axis)
        return CCZ4State(
            metric,
            conformal_a,
            field(18),
            field(19),
            field(20),
            gamma_hat,
            field(24),
            shift,
            driver,
        )

    parameters = CCZ4Parameters(
        kappa1=0.1,
        kappa2=0.0,
        kappa3=1.0,
        shift_speed=1.0,
        eta_shift=1.0,
        slicing="1+log",
        evolve_shift=True,
        radiative_boundaries=False,
    )
    cpu_ccz4 = CCZ4Solver(grid, parameters)
    conformal_metric = grid.zeros((3, 3))
    metric_wave = 0.012 * np.sin(wave * x) * np.cos(wave * y)
    conformal_metric[0, 0] = np.exp(metric_wave)
    conformal_metric[1, 1] = np.exp(-metric_wave)
    conformal_metric[2, 2] = 1.0
    conformal_a = grid.zeros((3, 3))
    conformal_a[0, 1] = conformal_a[1, 0] = (
        0.0015 * np.sin(wave * (x + y))
    )
    inverse = np.moveaxis(
        np.linalg.inv(
            np.moveaxis(conformal_metric, (0, 1), (-2, -1))
        ),
        (-2, -1),
        (0, 1),
    )
    trace_a = np.einsum(
        "ij...,ij...->...", inverse, conformal_a
    )
    conformal_a -= (
        conformal_metric * (trace_a / 3.0)[None, None, ...]
    )
    factor = 1.0 + 0.004 * np.cos(wave * x)
    trace_k = 0.002 * np.sin(wave * y)
    theta = 0.0005 * np.cos(wave * (x + y))
    lapse = 1.0 + 0.006 * np.sin(wave * (x - y))
    shift = grid.zeros((3,))
    shift[0] = 0.003 * np.cos(wave * y)
    shift[1] = 0.002 * np.sin(wave * x)
    driver = grid.zeros((3,))
    driver[0] = 0.0005 * np.sin(wave * x)
    driver[1] = -0.0005 * np.cos(wave * y)
    _, _, gamma_tilde = cpu_ccz4.conformal_connection(
        conformal_metric
    )
    gamma_hat = gamma_tilde.copy()
    gamma_hat[0] += 0.001 * np.sin(wave * y)
    geometry_state = CCZ4State(
        conformal_metric,
        conformal_a,
        factor,
        trace_k,
        theta,
        gamma_hat,
        lapse,
        shift,
        driver,
    )
    packed_geometry = pack_ccz4(geometry_state)

    geometry_backend = VulkanCCZ4GeometryBackend(points)
    vacuum = geometry_backend.vacuum_rhs(
        packed_geometry,
        shape=shape,
        lengths=(1.0, 1.0),
        kappa1=parameters.kappa1,
        kappa2=parameters.kappa2,
        kappa3=parameters.kappa3,
        cosmological_constant=parameters.cosmological_constant,
        shift_speed=parameters.shift_speed,
        eta_shift=parameters.eta_shift,
        slicing=parameters.slicing,
        evolve_shift=parameters.evolve_shift,
        derivative_order=derivative_order,
    )
    cpu_state = unpack_ccz4(vacuum.projected_state)
    zero_matter = (
        grid.zeros(),
        grid.zeros((3,)),
        grid.zeros((3, 3)),
    )
    cpu_rhs_fields = cpu_ccz4.rhs(
        0.0,
        (
            cpu_state.conformal_metric,
            cpu_state.conformal_A,
            cpu_state.conformal_factor,
            cpu_state.trace_K,
            cpu_state.theta,
            cpu_state.gamma_hat,
            cpu_state.lapse,
            cpu_state.shift,
            cpu_state.shift_driver,
        ),
        zero_matter,
    )
    cpu_rhs = pack_ccz4(CCZ4State(*cpu_rhs_fields))
    vacuum_error = np.abs(
        np.asarray(vacuum.vacuum_rhs, dtype=float)
        - np.asarray(cpu_rhs, dtype=float)
    )
    vacuum_worst = np.unravel_index(
        int(np.argmax(vacuum_error)), vacuum_error.shape
    )

    rng = np.random.default_rng(seed)
    primitive = np.zeros((points, 9), dtype=np.float32)
    primitive[:, 0] = flatten(
        np.log(0.1 * (1.0 + 0.015 * np.sin(wave * x)))
    )
    primitive[:, 1] = flatten(-2.0 + 0.01 * np.cos(wave * y))
    primitive[:, 2] = flatten(0.008 * np.sin(wave * y))
    primitive[:, 3] = flatten(-0.006 * np.cos(wave * x))
    primitive[:, 4] = 0.002
    primitive[:, 5] = flatten(
        np.log(0.005 * (1.0 + 0.02 * np.cos(wave * x)))
    )
    relative = flatten(0.12 + 0.035 * np.sin(wave * (x + y)))
    primitive[:, 6] = primitive[:, 2] + relative
    primitive[:, 7] = primitive[:, 3] - 0.35 * relative
    primitive[:, 8] = primitive[:, 4] + 0.2 * relative
    vector_input = rng.normal(
        0.0, 0.0008, (points, 20)
    ).astype(np.float32)
    tracer = flatten(
        0.2 + 0.03 * np.sin(wave * x) * np.cos(wave * y)
    ).astype(np.float32)
    coupled = VulkanTesseractCoupledBackend(
        shape,
        constitutive_model="qualified_frozen",
        relaxation_time=0.02,
        kappa_gravity=0.02,
        kappa1=parameters.kappa1,
        kappa2=parameters.kappa2,
        kappa3=parameters.kappa3,
        shift_speed=parameters.shift_speed,
        eta_shift=parameters.eta_shift,
        slicing=parameters.slicing,
        evolve_shift=parameters.evolve_shift,
        derivative_order=derivative_order,
    )
    state = coupled.initialize(
        primitive, vector_input, packed_geometry, tracer
    )
    initial = coupled.snapshot(state)
    initial_sum = np.sum(initial.matter.state, axis=0, dtype=float)
    reads_before = state.host_reads
    dt = 2.0e-4
    for _ in range(steps):
        coupled.step(state, dt, apply_drag=True)
    reads_after = state.host_reads
    final = coupled.snapshot(state)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = coupled.save_checkpoint(
            state, Path(directory) / "coupled_restart.npz"
        )
        restarted_backend = VulkanTesseractCoupledBackend(
            shape,
            constitutive_model="qualified_frozen",
            relaxation_time=0.02,
            kappa_gravity=0.02,
            kappa1=parameters.kappa1,
            kappa2=parameters.kappa2,
            kappa3=parameters.kappa3,
            shift_speed=parameters.shift_speed,
            eta_shift=parameters.eta_shift,
            slicing=parameters.slicing,
            evolve_shift=parameters.evolve_shift,
            derivative_order=derivative_order,
        )
        restarted = restarted_backend.load_checkpoint(checkpoint_path)
        restart_snapshot = restarted_backend.snapshot(restarted)
        restart_state_error = float(
            np.max(
                np.abs(
                    restart_snapshot.matter.state
                    - final.matter.state
                )
            )
        )
        restart_geometry_error = float(
            np.max(
                np.abs(
                    restart_snapshot.ccz4_state
                    - final.ccz4_state
                )
            )
        )
        coupled.step(state, dt, apply_drag=True)
        restarted_backend.step(restarted, dt, apply_drag=True)
        continued = coupled.snapshot(state)
        restarted_continued = restarted_backend.snapshot(restarted)
        restart_continuation_error = float(
            max(
                np.max(
                    np.abs(
                        restarted_continued.matter.state
                        - continued.matter.state
                    )
                ),
                np.max(
                    np.abs(
                        restarted_continued.ccz4_state
                        - continued.ccz4_state
                    )
                ),
            )
        )
    final_sum = np.sum(final.matter.state, axis=0, dtype=float)
    balance_scale = np.maximum(np.abs(initial_sum), 1.0)
    balance_error = np.abs(final_sum - initial_sum) / balance_scale
    metrics = {
        "ccz4_vacuum_maximum_absolute_error": float(
            np.max(vacuum_error)
        ),
        "ccz4_vacuum_rms_error": float(
            np.sqrt(np.mean(vacuum_error**2))
        ),
        "ccz4_vacuum_worst_component": int(
            vacuum_worst[1]
        ),
        "ccz4_vacuum_worst_gpu_value": float(
            vacuum.vacuum_rhs[vacuum_worst]
        ),
        "ccz4_vacuum_worst_cpu_value": float(
            cpu_rhs[vacuum_worst]
        ),
        "initial_maximum_recovery_residual": float(
            np.max(initial.matter.diagnostics[:, 0])
        ),
        "final_maximum_recovery_residual": float(
            np.max(final.matter.diagnostics[:, 0])
        ),
        "invalid_final_recoveries": int(
            np.count_nonzero(final.matter.diagnostics[:, 1] < 0.5)
        ),
        "baryon_balance_error": float(balance_error[0]),
        "carrier_balance_error": float(balance_error[1]),
        "tracer_balance_error": float(balance_error[10]),
        "entropy_change": float(final_sum[9] - initial_sum[9]),
        "maximum_projection_determinant_residual": float(
            np.max(final.ccz4_projection_audit[:, 0])
        ),
        "maximum_projection_trace_a_residual": float(
            np.max(final.ccz4_projection_audit[:, 1])
        ),
        "minimum_final_lapse": float(
            np.min(final.ccz4_state[:, 24])
        ),
        "minimum_absolute_switching_margin": float(
            np.min(np.abs(final.matter.diagnostics[:, 4]))
        ),
        "host_reads_during_evolution": int(reads_after - reads_before),
        "all_outputs_finite": bool(
            np.all(np.isfinite(final.matter.state))
            and np.all(np.isfinite(final.matter.primitive))
            and np.all(np.isfinite(final.ccz4_state))
            and np.all(np.isfinite(final.ccz4_rhs))
            and np.all(np.isfinite(final.matter_stress))
        ),
        "checkpoint_state_roundtrip_error": restart_state_error,
        "checkpoint_geometry_roundtrip_error": restart_geometry_error,
        "checkpoint_continuation_error": restart_continuation_error,
    }
    qualification = {
        f"derivative_ccz4_matches_fd{derivative_order}_cpu": (
            metrics["ccz4_vacuum_maximum_absolute_error"] < 2.0e-5
        ),
        "curved_nine_variable_recovery_valid": (
            metrics["initial_maximum_recovery_residual"] < 5.0e-4
            and metrics["final_maximum_recovery_residual"] < 5.0e-4
            and metrics["invalid_final_recoveries"] == 0
        ),
        "periodic_transport_balances_conserved_scalars": (
            metrics["baryon_balance_error"] < 3.0e-6
            and metrics["carrier_balance_error"] < 3.0e-6
            and metrics["tracer_balance_error"] < 3.0e-6
        ),
        "strang_drag_is_entropy_producing": (
            metrics["entropy_change"] >= -2.0e-6
        ),
        "ccz4_projection_constraints_hold": (
            metrics["maximum_projection_determinant_residual"] < 5.0e-6
            and metrics["maximum_projection_trace_a_residual"] < 5.0e-6
            and metrics["minimum_final_lapse"] > 0.0
        ),
        "frozen_switching_margin_remains_resolved": (
            metrics["minimum_absolute_switching_margin"] > 1.0e-5
        ),
        "full_stack_remains_device_resident": (
            metrics["host_reads_during_evolution"] == 0
        ),
        "checkpoint_restart_is_exact": (
            metrics["checkpoint_state_roundtrip_error"] == 0.0
            and metrics["checkpoint_geometry_roundtrip_error"] < 1.0e-7
            and metrics["checkpoint_continuation_error"] < 2.0e-6
        ),
        "full_stack_outputs_are_finite": metrics["all_outputs_finite"],
    }
    return {
        "schema": "vulkan-tesseract.multidimensional-full-stack.v1",
        "backend": "Slang/Vulkan",
        "shape": list(shape),
        "steps": steps,
        "dt": dt,
        "ccz4_derivative_order": derivative_order,
        "constitutive_model": "qualified_frozen",
        "spatial_operator": (
            "multidirectional Rusanov plus midpoint Carter DLM path"
        ),
        "time_integrator": (
            "CCZ4 half-step / Carter SSP-RK3 with implicit drag / "
            "CCZ4 half-step"
        ),
        "ccz4_derivatives": (
            f"periodic centered FD{derivative_order}"
        ),
        "metrics": metrics,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def qualify_vulkan_full_carter_block(
    *,
    points: int = 6,
    seed: int = 3309,
) -> dict[str, object]:
    """Qualify all three curved directions of the physical nine-field block."""
    rng = np.random.default_rng(seed)
    primitive = np.zeros((points, 9))
    primitive[:, 0] = np.log(rng.uniform(0.09, 0.11, points))
    primitive[:, 1] = rng.uniform(-2.1, -1.9, points)
    primitive[:, 2:5] = rng.normal(0.0, 0.025, (points, 3))
    primitive[:, 5] = np.log(
        rng.uniform(0.0046, 0.0054, points)
    )
    relative = rng.normal(0.0, 1.0, (points, 3))
    relative /= np.linalg.norm(relative, axis=1)[:, None]
    magnitudes = np.where(
        np.arange(points) % 2 == 0, 0.10, 0.18
    )
    relative_velocity = magnitudes[:, None] * relative
    relative_q = relative_velocity / np.sqrt(
        1.0 - magnitudes**2
    )[:, None]
    primitive[:, 6:9] = primitive[:, 2:5] + relative_q
    geometry = np.empty((points, 13))
    for point in range(points):
        factor = np.eye(3) + rng.normal(0.0, 0.035, (3, 3))
        metric = factor.T @ factor
        geometry[point] = np.concatenate(
            (
                metric.reshape(-1),
                [rng.uniform(0.92, 1.08)],
                rng.normal(0.0, 0.015, 3),
            )
        )
    vector_input = rng.normal(0.0, 0.025, (points, 20))
    vector_solver = _source_solver("analytic_m1")
    vector_backend = VulkanTheory33SourceBackend(points)
    gpu_vector_stress = vector_backend.curved_vector_stress(
        primitive, vector_input, geometry
    )
    cpu_vector_stress = np.stack(
        [
            reference_curved_vector_stress(
                vector_solver,
                primitive[point],
                vector_input[point],
                geometry[point],
            )
            for point in range(points)
        ]
    )
    curved_vector_error = float(
        np.max(np.abs(gpu_vector_stress - cpu_vector_stress))
    )

    audits: list[dict[str, object]] = []
    for mode in ("analytic_m1", "qualified_frozen"):
        solver = _source_solver(mode)
        backend = VulkanTheory33SourceBackend(
            points, constitutive_model=mode
        )
        for axis in range(3):
            principal = backend.principal(
                primitive, geometry, axis=axis
            )
            eigensystem = backend.full_eigensystem(
                primitive, geometry, axis=axis
            )
            reference_principal = [
                reference_full_principal(
                    solver,
                    primitive[point],
                    geometry[point],
                    axis=axis,
                )
                for point in range(points)
            ]
            reference_symbols = [
                reference_full_carter_symbol(
                    solver,
                    primitive[point],
                    geometry[point],
                    axis=axis,
                )
                for point in range(points)
            ]
            reference_u = np.stack(
                [item[0] for item in reference_principal]
            )
            reference_flux = np.stack(
                [item[1] for item in reference_principal]
            )
            reference_path = np.stack(
                [item[2] for item in reference_principal]
            )
            reference_symbol = np.stack(
                [item["symbol"] for item in reference_symbols]
            )
            reference_values = np.stack(
                [
                    np.asarray(item["eigenvalues"]).real
                    for item in reference_symbols
                ]
            )
            symbol_scale = max(
                float(np.max(np.abs(reference_symbol))), 1.0
            )
            audits.append(
                {
                    "constitutive_model": mode,
                    "axis": axis,
                    "maximum_conserved_absolute_error": float(
                        np.max(
                            np.abs(
                                principal.conserved - reference_u
                            )
                        )
                    ),
                    "maximum_flux_absolute_error": float(
                        np.max(
                            np.abs(principal.flux - reference_flux)
                        )
                    ),
                    "maximum_path_absolute_error": float(
                        np.max(
                            np.abs(principal.path - reference_path)
                        )
                    ),
                    "maximum_symbol_relative_error": float(
                        np.max(
                            np.abs(
                                eigensystem.symbol - reference_symbol
                            )
                        )
                        / symbol_scale
                    ),
                    "maximum_eigenvalue_absolute_error": float(
                        np.max(
                            np.abs(
                                eigensystem.eigenvalues
                                - reference_values
                            )
                        )
                    ),
                    "maximum_cpu_imaginary_part": float(
                        max(
                            item["maximum_imaginary_part"]
                            for item in reference_symbols
                        )
                    ),
                    "maximum_gpu_eigenpair_residual": float(
                        np.max(eigensystem.audit[:, 1])
                    ),
                    "maximum_gpu_inverse_residual": float(
                        np.max(eigensystem.audit[:, 2])
                    ),
                    "maximum_condition_estimate": float(
                        np.max(eigensystem.audit[:, 3])
                    ),
                    "invalid_geometry_points": int(
                        np.count_nonzero(principal.valid < 0.5)
                    ),
                    "invalid_eigensystems": int(
                        np.count_nonzero(eigensystem.audit[:, 4] < 0.5)
                    ),
                    "maximum_coordinate_speed": float(
                        np.max(eigensystem.audit[:, 0])
                    ),
                    "phase_two_points": int(
                        sum(
                            bool(item["phase_two"])
                            for item in reference_symbols
                        )
                    ),
                }
            )
    qualification = {
        "curved_principal_map_matches_cpu": all(
            item["maximum_conserved_absolute_error"] < 2.0e-6
            and item["maximum_flux_absolute_error"] < 2.0e-6
            and item["maximum_path_absolute_error"] < 2.0e-6
            for item in audits
        ),
        "full_flux_path_symbol_matches_cpu": all(
            item["maximum_symbol_relative_error"] < 3.0e-3
            for item in audits
        ),
        "all_nine_eigenvalues_match_cpu": all(
            item["maximum_eigenvalue_absolute_error"] < 1.5e-3
            for item in audits
        ),
        "cpu_spectra_are_real": all(
            item["maximum_cpu_imaginary_part"] < 1.0e-7
            for item in audits
        ),
        "gpu_bases_are_complete": all(
            item["maximum_gpu_eigenpair_residual"] < 2.0e-3
            and item["maximum_gpu_inverse_residual"] < 2.0e-3
            and item["invalid_eigensystems"] == 0
            for item in audits
        ),
        "all_curved_geometries_are_valid": all(
            item["invalid_geometry_points"] == 0 for item in audits
        ),
        "all_three_spatial_directions_exercised": (
            sorted({item["axis"] for item in audits}) == [0, 1, 2]
        ),
        "both_frozen_neural_branches_exercised": any(
            item["constitutive_model"] == "qualified_frozen"
            and 0 < item["phase_two_points"] < points
            for item in audits
        ),
        "curved_mixed_vector_stress_matches_cpu": (
            curved_vector_error < 2.0e-6
        ),
    }
    return {
        "schema": "vulkan-tesseract.full-carter-block.v1",
        "backend": "Slang/Vulkan",
        "physical_conserved_order": (
            "D_N,D_D,S_x,S_y,S_z,E,P_D_x,P_D_y,P_D_z"
        ),
        "geometry_order": (
            "h_xx,h_xy,h_xz,h_yx,h_yy,h_yz,"
            "h_zx,h_zy,h_zz,lapse,shift_x,shift_y,shift_z"
        ),
        "points": points,
        "directional_audits": audits,
        "maximum_curved_vector_stress_absolute_error": (
            curved_vector_error
        ),
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def qualify_vulkan_ccz4_geometry(
    *,
    points: int = 8,
    seed: int = 334,
) -> dict[str, object]:
    """Qualify packed CCZ4 projection and Carter matter injection."""
    rng = np.random.default_rng(seed)
    state = np.zeros((points, 31))
    matter = np.zeros((points, 13))
    for point in range(points):
        factor = np.eye(3) + rng.normal(0.0, 0.06, (3, 3))
        state[point, :9] = (factor.T @ factor).reshape(-1)
        state[point, 9:18] = rng.normal(0.0, 0.025, 9)
        state[point, 18] = rng.uniform(0.92, 1.08)
        state[point, 19:24] = rng.normal(0.0, 0.015, 5)
        state[point, 24] = rng.uniform(0.9, 1.1)
        state[point, 25:31] = rng.normal(0.0, 0.02, 6)
        matter[point, 0] = rng.uniform(0.05, 0.15)
        matter[point, 1:4] = rng.normal(0.0, 0.01, 3)
        stress = rng.normal(0.0, 0.008, (3, 3))
        stress = 0.5 * (stress + stress.T)
        stress += np.eye(3) * rng.uniform(0.01, 0.04)
        matter[point, 4:13] = stress.reshape(-1)
    backend = VulkanCCZ4GeometryBackend(points)
    snapshot = backend.evaluate(state, matter)
    projected = np.empty_like(state)
    geometry = np.empty((points, 13))
    extrinsic = np.empty((points, 3, 3))
    source = np.zeros_like(state)
    determinant_errors = []
    trace_errors = []
    kappa = 8.0 * np.pi
    for point in range(points):
        metric = state[point, :9].reshape(3, 3)
        metric = 0.5 * (metric + metric.T)
        metric *= np.linalg.det(metric) ** (-1.0 / 3.0)
        inverse = np.linalg.inv(metric)
        conformal_a = state[point, 9:18].reshape(3, 3)
        conformal_a = 0.5 * (conformal_a + conformal_a.T)
        trace_a = float(np.sum(inverse * conformal_a))
        conformal_a -= metric * trace_a / 3.0
        projected[point] = state[point]
        projected[point, :9] = metric.reshape(-1)
        projected[point, 9:18] = conformal_a.reshape(-1)
        projected[point, 18] = max(projected[point, 18], 1.0e-8)
        projected[point, 24] = max(projected[point, 24], 1.0e-8)
        phi = projected[point, 18]
        geometry[point, :9] = (metric / phi**2).reshape(-1)
        geometry[point, 9] = projected[point, 24]
        geometry[point, 10:13] = projected[point, 25:28]
        extrinsic[point] = (
            conformal_a + projected[point, 19] * metric / 3.0
        ) / phi**2
        alpha = projected[point, 24]
        physical_trace = float(
            np.sum(
                phi**2
                * inverse
                * matter[point, 4:13].reshape(3, 3)
            )
        )
        source[point, 9:18] = (
            -kappa
            * alpha
            * phi**2
            * matter[point, 4:13].reshape(3, 3)
            + kappa * alpha * metric * physical_trace / 3.0
        ).reshape(-1)
        rho = matter[point, 0]
        source[point, 19] = (
            0.5 * kappa * alpha * (physical_trace - 3.0 * rho)
        )
        source[point, 20] = -kappa * alpha * rho
        source[point, 21:24] = (
            -2.0 * kappa * alpha * (inverse @ matter[point, 1:4])
        )
        determinant_errors.append(abs(np.linalg.det(metric) - 1.0))
        trace_errors.append(abs(np.sum(inverse * conformal_a)))
    metrics = {
        "maximum_projected_state_absolute_error": float(
            np.max(np.abs(snapshot.projected_state - projected))
        ),
        "maximum_physical_geometry_absolute_error": float(
            np.max(np.abs(snapshot.physical_geometry - geometry))
        ),
        "maximum_extrinsic_curvature_absolute_error": float(
            np.max(np.abs(snapshot.extrinsic_curvature - extrinsic))
        ),
        "maximum_matter_source_absolute_error": float(
            np.max(np.abs(snapshot.matter_source - source))
        ),
        "maximum_gpu_determinant_residual": float(
            np.max(snapshot.projection_audit[:, 0])
        ),
        "maximum_gpu_trace_a_residual": float(
            np.max(snapshot.projection_audit[:, 1])
        ),
        "invalid_gpu_geometries": int(
            np.count_nonzero(snapshot.projection_audit[:, 2] < 0.5)
        ),
        "maximum_cpu_determinant_residual": float(
            max(determinant_errors)
        ),
        "maximum_cpu_trace_a_residual": float(max(trace_errors)),
    }
    qualification = {
        "algebraic_projection_matches_cpu": (
            metrics["maximum_projected_state_absolute_error"] < 2.0e-6
        ),
        "physical_geometry_matches_cpu": (
            metrics["maximum_physical_geometry_absolute_error"] < 2.0e-6
            and metrics[
                "maximum_extrinsic_curvature_absolute_error"
            ]
            < 2.0e-6
        ),
        "matter_source_matches_cpu": (
            metrics["maximum_matter_source_absolute_error"] < 2.0e-5
        ),
        "determinant_constraint_projected": (
            metrics["maximum_gpu_determinant_residual"] < 2.0e-5
        ),
        "tracefree_constraint_projected": (
            metrics["maximum_gpu_trace_a_residual"] < 2.0e-5
        ),
        "all_projected_geometries_valid": (
            metrics["invalid_gpu_geometries"] == 0
        ),
    }
    return {
        "schema": "vulkan-tesseract.ccz4-geometry.v1",
        "backend": "Slang/Vulkan",
        "state_order": (
            "gt_ij,At_ij,phi,K,theta,GammaHat_i,alpha,beta_i,B_i"
        ),
        "points": points,
        "metrics": metrics,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def _restricted_prediction(
    solver: Theory33ProductionSolver,
    primitive: Array,
    phase_override: bool | None,
) -> tuple[Array, Array, Array, bool, float]:
    log_n, q_n, log_d, q_d, entropy = np.asarray(
        primitive, dtype=float
    )
    n = float(np.exp(log_n))
    d = float(np.exp(log_d))
    w_n = float(np.sqrt(1.0 + q_n**2))
    w_d = float(np.sqrt(1.0 + q_d**2))
    velocity_n = q_n / w_n
    velocity_d = q_d / w_d
    relative_gamma = max(w_n * w_d - q_n * q_d, 1.0)
    cross = n * d * relative_gamma
    value, gradient, phase, threshold, _ = (
        solver.master.invariant_response(
            n**2,
            d**2,
            cross,
            entropy,
            phase_override=phase_override,
        )
    )
    b_n = float(-2.0 * gradient[..., 0])
    b_d = float(-2.0 * gradient[..., 1])
    entrainment = float(-gradient[..., 2])
    generalized_pressure = float(
        value
        + b_n * n**2
        + b_d * d**2
        + 2.0 * entrainment * cross
    )
    d_n = n * w_n
    d_d = d * w_d
    momentum = (
        b_n * d_n**2 * velocity_n
        + b_d * d_d**2 * velocity_d
        + entrainment * d_n * d_d * (velocity_n + velocity_d)
    )
    energy = (
        -generalized_pressure
        + b_n * d_n**2
        + b_d * d_d**2
        + 2.0 * entrainment * d_n * d_d
    )
    chi_down = (
        b_d * d_d * velocity_d
        + entrainment * d_n * velocity_n
    )
    chi_zero = -(b_d * d_d + entrainment * d_n)
    canonical = d_d * chi_down
    stress = (
        generalized_pressure
        + b_n * d_n**2 * velocity_n**2
        + b_d * d_d**2 * velocity_d**2
        + 2.0
        * entrainment
        * d_n
        * d_d
        * velocity_n
        * velocity_d
    )
    conserved = np.asarray((d_n, d_d, momentum, energy, canonical))
    flux = np.asarray(
        (
            d_n * velocity_n,
            d_d * velocity_d,
            stress,
            momentum,
            canonical * velocity_d,
        )
    )
    path = np.asarray(
        (d_d, chi_zero, d_d * velocity_d, chi_down)
    )
    return (
        conserved,
        flux,
        path,
        bool(np.asarray(phase)),
        float(relative_gamma - 1.0 - np.asarray(threshold)),
    )


def restricted_carter_eigensystem(
    solver: Theory33ProductionSolver,
    primitive: Array,
    *,
    phase_override: bool | None = None,
) -> dict[str, Array | float | bool]:
    """Independent float64 5x5 flux/path symbol for the resident ABI."""
    point = np.asarray(primitive, dtype=float)
    base_conserved, base_flux, base_path, phase, margin = (
        _restricted_prediction(solver, point, phase_override)
    )
    if phase_override is None and solver.master.constitutive_closure is not None:
        phase_override = phase
    time_jacobian = np.empty((5, 5))
    flux_path_jacobian = np.empty((5, 5))
    for column in range(5):
        step = 2.0e-6 * max(1.0, abs(float(point[column])))
        plus = point.copy()
        minus = point.copy()
        plus[column] += step
        minus[column] -= step
        plus_u, plus_f, plus_path, _, _ = _restricted_prediction(
            solver, plus, phase_override
        )
        minus_u, minus_f, minus_path, _, _ = _restricted_prediction(
            solver, minus, phase_override
        )
        time_jacobian[:, column] = (
            plus_u - minus_u
        ) / (2.0 * step)
        flux_path_jacobian[:, column] = (
            plus_f - minus_f
        ) / (2.0 * step)
        chi_zero_derivative = (
            plus_path[1] - minus_path[1]
        ) / (2.0 * step)
        chi_down_derivative = (
            plus_path[3] - minus_path[3]
        ) / (2.0 * step)
        flux_path_jacobian[4, column] -= (
            base_path[0] * chi_zero_derivative
            + base_path[2] * chi_down_derivative
        )
    symbol = flux_path_jacobian @ np.linalg.inv(time_jacobian)
    values, right = np.linalg.eig(symbol)
    order = np.argsort(values.real)
    values = values[order]
    right = right[:, order]
    right /= np.maximum(np.linalg.norm(right, axis=0), 1.0e-300)
    return {
        "symbol": symbol,
        "eigenvalues": values,
        "right": right,
        "maximum_imaginary_part": float(np.max(np.abs(values.imag))),
        "condition": float(np.linalg.cond(right)),
        "phase_two": phase,
        "switching_margin": margin,
    }


def smooth_theory33_primitives(points: int) -> Array:
    coordinate = (np.arange(points, dtype=float) + 0.5) / points
    phase = 2.0 * np.pi * coordinate
    values = np.zeros((points, 6), dtype=np.float64)
    values[:, 0] = 0.10 + 0.005 * np.sin(phase)
    values[:, 1] = 0.12 + 0.003 * np.cos(phase)
    values[:, 2] = 0.01 * np.sin(phase)
    values[:, 3] = 0.005 + 0.0002 * np.cos(phase)
    values[:, 4] = 0.02 * np.cos(phase)
    values[:, 5] = 0.2 + 0.01 * np.sin(phase)
    return values


def neural_phase_contact_primitives(points: int) -> Array:
    """Return the moving two-branch contact used by Vulkan stress tests."""
    coordinate = (np.arange(points, dtype=float) + 0.5) / points
    left = coordinate < 0.5
    values = np.zeros((points, 6), dtype=np.float64)
    values[:, 0] = 0.10
    values[:, 1] = 0.12
    values[:, 2] = 0.08
    values[:, 3] = 0.005
    relative_velocity = np.where(left, 0.112, 0.103)
    values[:, 4] = np.tanh(
        np.arctanh(values[:, 2]) + np.arctanh(relative_velocity)
    )
    values[:, 5] = 0.2
    return values


def original_neural_phase_riemann_primitives(points: int) -> Array:
    """Return the literal initial state of the original Riemann campaign."""
    coordinate = (np.arange(points, dtype=float) + 0.5) / points
    left = coordinate < 0.5
    values = np.zeros((points, 6), dtype=np.float64)
    values[:, 0] = 0.10
    values[:, 1] = 0.12
    values[:, 2] = 0.0
    values[:, 3] = 0.005
    values[:, 4] = np.where(left, 0.112, 0.103)
    values[:, 5] = 0.0
    return values


def original_neural_phase_riemann_full_primitives(points: int) -> Array:
    """Return the original Riemann state in the full nine-primitive ABI."""
    compact = original_neural_phase_riemann_primitives(points)
    closure = QualifiedFrozenClosure()
    values = np.zeros((points, 9), dtype=np.float64)
    values[:, 0] = np.log(compact[:, 0])
    values[:, 1] = closure.analytic.specific_entropy(
        compact[:, 0], compact[:, 1]
    )
    fluid_w = 1.0 / np.sqrt(1.0 - compact[:, 2] ** 2)
    carrier_w = 1.0 / np.sqrt(1.0 - compact[:, 4] ** 2)
    values[:, 2] = fluid_w * compact[:, 2]
    values[:, 5] = np.log(compact[:, 3])
    values[:, 6] = carrier_w * compact[:, 4]
    return values


def carter_riemann_full_primitives(
    points: int, problem: str
) -> Array:
    """Return an original Carter Riemann state in the full GPU ABI."""
    coordinate = (np.arange(points, dtype=float) + 0.5) / points
    left = coordinate < 0.5
    if problem == "counterflow_shock":
        data = (0.11, 0.07, 0.17, 0.09, 0.02, 0.0,
                0.006, 0.0035, 0.08, -0.02)
    elif problem == "colliding_streams":
        data = (0.11, 0.065, 0.17, 0.09, 0.025, -0.02,
                0.007, 0.003, 0.06, -0.01)
    elif problem == "neural_phase_contact":
        return original_neural_phase_riemann_full_primitives(points)
    else:
        raise ValueError(f"unknown Carter Riemann problem {problem!r}")
    (
        n_left, n_right, e_left, e_right, u_left, u_right,
        d_left, d_right, v_left, v_right,
    ) = data
    density = np.where(left, n_left, n_right)
    internal = np.where(left, e_left, e_right)
    velocity = np.where(left, u_left, u_right)
    carrier = np.where(left, d_left, d_right)
    carrier_velocity = np.where(left, v_left, v_right)
    master = _source_solver("analytic_m1").master
    values = np.zeros((points, 9), dtype=np.float64)
    values[:, 0] = np.log(density)
    values[:, 1] = master.specific_entropy(density, internal)
    values[:, 2] = velocity / np.sqrt(1.0 - velocity**2)
    values[:, 5] = np.log(carrier)
    values[:, 6] = carrier_velocity / np.sqrt(
        1.0 - carrier_velocity**2
    )
    return values


@dataclass
class CpuMaterialReference:
    """Material-only flat 1D projection and Rusanov oracle.

    It deliberately invokes the production solver's independent energy
    recovery and master evaluation while excluding the mixed-vector stress.
    That is the exact vertical slice implemented by ``tesseract_nr_gpu.slang``.
    """

    solver: Theory33ProductionSolver
    state: Array
    metric: Array
    dx: float
    reconstruction: str = "piecewise_constant"
    positivity_preserving: bool = False
    positivity_floor_factor: float = 1.001
    time: float = 0.0
    steps: int = 0
    minimum_positivity_theta: float = 1.0

    @classmethod
    def initialize(
        cls,
        primitives: Array,
        *,
        constitutive_model: str = "analytic_m1",
        reconstruction: str = "piecewise_constant",
        positivity_preserving: bool = False,
        positivity_floor_factor: float = 1.001,
    ) -> "CpuMaterialReference":
        points = int(primitives.shape[0])
        if constitutive_model not in {
            "analytic_m1",
            "qualified_frozen",
        }:
            raise ValueError("unknown constitutive model")
        if reconstruction not in {
            "piecewise_constant",
            "weno5_z",
            "carter_characteristic_weno5_z",
            "carter_characteristic_roe",
        }:
            raise ValueError("unknown reconstruction")
        if positivity_floor_factor < 1.0:
            raise ValueError("positivity_floor_factor must be at least one")
        solver = Theory33ProductionSolver(
            PeriodicGrid((points,), (1.0,), method="fd2"),
            production=Theory3ProductionParameters(
                flux_reconstruction="piecewise_constant",
                positivity_preserving=False,
            ),
            constitutive_closure=(
                QualifiedFrozenClosure()
                if constitutive_model == "qualified_frozen"
                else None
            ),
        )
        velocity_n = np.zeros((3, points))
        velocity_d = np.zeros((3, points))
        velocity_n[0] = primitives[:, 2]
        velocity_d[0] = primitives[:, 4]
        production_state = solver.initialize(
            solver.ccz4.flat_state(),
            FluidPrimitive(
                primitives[:, 0],
                primitives[:, 1],
                velocity_n,
                primitives[:, 5],
            ),
            carrier=CarrierPrimitive(
                primitives[:, 3], velocity_d
            ),
        )
        recovery = solver.recover(production_state)
        state = np.stack(
            (
                production_state.matter.D,
                production_state.target_charge,
                recovery.master.stress.momentum[0],
                recovery.master.stress.rho,
                production_state.target_current[0],
                production_state.matter.entropy,
                production_state.matter.tracer,
            ),
            axis=-1,
        )
        metric = np.zeros((3, 3, points))
        metric[0, 0] = 1.0
        metric[1, 1] = 1.0
        metric[2, 2] = 1.0
        return cls(
            solver,
            state,
            metric,
            1.0 / points,
            reconstruction,
            positivity_preserving,
            positivity_floor_factor,
        )

    def _recover(
        self, state: Array
    ) -> tuple[FluidPrimitive, CarrierPrimitive, object, Array]:
        points = state.shape[0]
        momentum = np.zeros((3, points))
        canonical = np.zeros((3, points))
        momentum[0] = state[:, 2]
        canonical[0] = state[:, 4]
        (
            fluid,
            carrier,
            entropy,
            residual,
            failed,
            _phase_crossings,
        ) = self.solver._recover_cells_from_energy(
            self.metric,
            state[:, 0],
            state[:, 1],
            momentum,
            state[:, 3],
            canonical,
            state[:, 5] / np.maximum(state[:, 0], 1.0e-300),
        )
        if failed:
            raise FloatingPointError(
                "CPU material recovery failed: "
                f"cells={failed}, residual={np.max(residual):.3e}"
            )
        master = self.solver.master.evaluate(
            self.metric, fluid, carrier
        )
        state[:, 5] = state[:, 0] * entropy
        return fluid, carrier, master, entropy

    @staticmethod
    def _weno5_z(
        v0: Array,
        v1: Array,
        v2: Array,
        v3: Array,
        v4: Array,
    ) -> Array:
        q0 = (2.0 * v0 - 7.0 * v1 + 11.0 * v2) / 6.0
        q1 = (-v1 + 5.0 * v2 + 2.0 * v3) / 6.0
        q2 = (2.0 * v2 + 5.0 * v3 - v4) / 6.0
        beta0 = (
            (13.0 / 12.0) * (v0 - 2.0 * v1 + v2) ** 2
            + 0.25 * (v0 - 4.0 * v1 + 3.0 * v2) ** 2
        )
        beta1 = (
            (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2
            + 0.25 * (v1 - v3) ** 2
        )
        beta2 = (
            (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2
            + 0.25 * (3.0 * v2 - 4.0 * v3 + v4) ** 2
        )
        tau5 = np.abs(beta0 - beta2)
        scale = np.maximum.reduce(
            (np.abs(v0), np.abs(v1), np.abs(v2), np.abs(v3), np.abs(v4))
        )
        epsilon = 1.0e-26 + 1.0e-12 * scale**2
        alpha0 = 0.1 * (1.0 + (tau5 / (beta0 + epsilon)) ** 2)
        alpha1 = 0.6 * (1.0 + (tau5 / (beta1 + epsilon)) ** 2)
        alpha2 = 0.3 * (1.0 + (tau5 / (beta2 + epsilon)) ** 2)
        return (
            alpha0 * q0 + alpha1 * q1 + alpha2 * q2
        ) / (alpha0 + alpha1 + alpha2)

    def _interfaces(
        self,
        state: Array,
        flux: Array,
        primitive: Array | None = None,
    ) -> tuple[Array, Array]:
        following_state = np.roll(state, -1, axis=0)
        following_flux = np.roll(flux, -1, axis=0)
        low = (
            0.5 * (flux + following_flux)
            - 0.5 * (following_state - state)
        )
        if self.reconstruction == "piecewise_constant":
            return low, low.copy()
        if self.reconstruction == "carter_characteristic_roe":
            if primitive is None:
                raise ValueError(
                    "characteristic reconstruction needs primitives"
                )
            high = low.copy()
            count = state.shape[0]
            for face in range(count):
                following = (face + 1) % count
                left_x = primitive[face]
                right_x = primitive[following]
                face_x = 0.5 * (left_x + right_x)
                _, _, _, left_phase, left_margin = (
                    _restricted_prediction(self.solver, left_x, None)
                )
                _, _, _, right_phase, right_margin = (
                    _restricted_prediction(self.solver, right_x, None)
                )
                phase_override: bool | None = None
                if (
                    self.solver.master.constitutive_closure is not None
                    and left_phase != right_phase
                ):
                    if abs(left_margin) >= abs(right_margin):
                        face_x = left_x
                        phase_override = left_phase
                    else:
                        face_x = right_x
                        phase_override = right_phase
                eigensystem = restricted_carter_eigensystem(
                    self.solver,
                    face_x,
                    phase_override=phase_override,
                )
                right = np.asarray(eigensystem["right"]).real
                left = np.linalg.inv(right)
                values = np.asarray(
                    eigensystem["eigenvalues"]
                ).real
                speed = min(
                    max(float(np.max(np.abs(values))), 1.0e-4),
                    1.0,
                )
                jump = state[following, :5] - state[face, :5]
                rates = 0.5 * np.abs(values) + 0.5 * speed
                dissipation = right @ (rates * (left @ jump))
                high[face, :5] = (
                    0.5 * (flux[face, :5] + flux[following, :5])
                    - 0.5 * dissipation
                )
                high[face, 5:] = (
                    0.5 * (flux[face, 5:] + flux[following, 5:])
                    - 0.25
                    * speed
                    * (state[following, 5:] - state[face, 5:])
                )
            return low, high
        left_state = self._weno5_z(
            np.roll(state, 2, axis=0),
            np.roll(state, 1, axis=0),
            state,
            np.roll(state, -1, axis=0),
            np.roll(state, -2, axis=0),
        )
        right_state = self._weno5_z(
            np.roll(state, -3, axis=0),
            np.roll(state, -2, axis=0),
            np.roll(state, -1, axis=0),
            state,
            np.roll(state, 1, axis=0),
        )
        left_flux = self._weno5_z(
            np.roll(flux, 2, axis=0),
            np.roll(flux, 1, axis=0),
            flux,
            np.roll(flux, -1, axis=0),
            np.roll(flux, -2, axis=0),
        )
        right_flux = self._weno5_z(
            np.roll(flux, -3, axis=0),
            np.roll(flux, -2, axis=0),
            np.roll(flux, -1, axis=0),
            flux,
            np.roll(flux, 1, axis=0),
        )
        high = (
            0.5 * (left_flux + right_flux)
            - 0.5 * (right_state - left_state)
        )
        if self.reconstruction != "carter_characteristic_weno5_z":
            return low, high
        if primitive is None:
            raise ValueError("characteristic reconstruction needs primitives")
        count = state.shape[0]
        for face in range(count):
            following = (face + 1) % count
            left_x = primitive[face]
            right_x = primitive[following]
            face_x = 0.5 * (left_x + right_x)
            _, _, _, left_phase, left_margin = _restricted_prediction(
                self.solver, left_x, None
            )
            _, _, _, right_phase, right_margin = _restricted_prediction(
                self.solver, right_x, None
            )
            phase_override: bool | None = None
            if (
                self.solver.master.constitutive_closure is not None
                and left_phase != right_phase
            ):
                if abs(left_margin) >= abs(right_margin):
                    face_x = left_x
                    phase_override = left_phase
                else:
                    face_x = right_x
                    phase_override = right_phase
            eigensystem = restricted_carter_eigensystem(
                self.solver,
                face_x,
                phase_override=phase_override,
            )
            right = np.asarray(eigensystem["right"]).real
            left = np.linalg.inv(right)
            speed = min(
                max(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(
                                    eigensystem["eigenvalues"]
                                ).real
                            )
                        )
                    ),
                    1.0e-4,
                ),
                1.0,
            )

            def split(offset: int, sign: float) -> Array:
                sample = (face + offset) % count
                return left @ (
                    flux[sample, :5]
                    + sign * speed * state[sample, :5]
                )

            plus = self._weno5_z(
                split(-2, 1.0),
                split(-1, 1.0),
                split(0, 1.0),
                split(1, 1.0),
                split(2, 1.0),
            )
            minus = self._weno5_z(
                split(3, -1.0),
                split(2, -1.0),
                split(1, -1.0),
                split(0, -1.0),
                split(-1, -1.0),
            )
            high[face, :5] = 0.5 * right @ (plus + minus)
        return low, high

    def _path_source(self, path: Array) -> Array:
        following = np.roll(path, -1, axis=0)
        previous = np.roll(path, 1, axis=0)
        return 0.5 / self.dx * (
            0.5
            * (path[:, 0] + following[:, 0])
            * (following[:, 1] - path[:, 1])
            + 0.5
            * (path[:, 0] + previous[:, 0])
            * (path[:, 1] - previous[:, 1])
            + 0.5
            * (path[:, 2] + following[:, 2])
            * (following[:, 3] - path[:, 3])
            + 0.5
            * (path[:, 2] + previous[:, 2])
            * (path[:, 3] - previous[:, 3])
        )

    def _rhs_from_interface(self, interface: Array, path: Array) -> Array:
        rhs = -(interface - np.roll(interface, 1, axis=0)) / self.dx
        rhs[:, 4] += self._path_source(path)
        return rhs

    @staticmethod
    def _admissibility(state: Array) -> Array:
        return (
            state[:, 3]
            - np.sqrt(
                np.maximum(state[:, 0] ** 2 + state[:, 2] ** 2, 0.0)
            )
            - 1.0e-12
        )

    def _limited_interface(
        self,
        state: Array,
        low: Array,
        high: Array,
        path: Array,
        dt: float,
    ) -> Array:
        if not self.positivity_preserving:
            self.minimum_positivity_theta = 1.0
            return high
        low_state = state + dt * self._rhs_from_interface(low, path)
        high_state = state + dt * self._rhs_from_interface(high, path)
        theta = np.ones(state.shape[0])
        density_floor = 1.0e-8 * self.positivity_floor_factor
        carrier_floor = 1.0e-8 * self.positivity_floor_factor
        for index, floor in (
            (0, density_floor),
            (1, carrier_floor),
        ):
            troubled = high_state[:, index] < floor
            ratio = np.clip(
                (low_state[:, index] - floor)
                / np.maximum(
                    low_state[:, index] - high_state[:, index], 1.0e-12
                ),
                0.0,
                1.0,
            )
            theta = np.where(troubled, np.minimum(theta, ratio), theta)
        energy_floor = 1.0e-12 * self.positivity_floor_factor
        energy_troubled = (
            self._admissibility(high_state) - energy_floor + 1.0e-12
        ) < 0.0
        lower = np.zeros_like(theta)
        upper = theta.copy()
        delta = high_state - low_state
        for _ in range(18):
            middle = 0.5 * (lower + upper)
            trial = low_state + middle[:, None] * delta
            admissible = (
                self._admissibility(trial) - energy_floor + 1.0e-12
            ) >= 0.0
            lower = np.where(admissible, middle, lower)
            upper = np.where(admissible, upper, middle)
        theta = np.where(energy_troubled, lower, theta)
        face_theta = np.minimum(theta, np.roll(theta, -1))
        self.minimum_positivity_theta = min(
            self.minimum_positivity_theta, float(np.min(face_theta))
        )
        return low + face_theta[:, None] * (high - low)

    def _rhs(self, state: Array, dt: float = 0.0) -> Array:
        fluid, carrier, master, entropy = self._recover(state)
        flux = np.stack(
            (
                state[:, 0] * fluid.velocity[0],
                state[:, 1] * carrier.velocity[0],
                master.stress.stress[0, 0],
                master.stress.momentum[0],
                state[:, 4] * carrier.velocity[0],
                state[:, 0] * entropy * fluid.velocity[0],
                state[:, 6] * fluid.velocity[0],
            ),
            axis=-1,
        )
        path = np.stack(
            (
                state[:, 1],
                -master.carrier_momentum_normal,
                state[:, 1] * carrier.velocity[0],
                master.carrier_momentum_down[0],
            ),
            axis=-1,
        )
        w_n = 1.0 / np.sqrt(
            np.maximum(1.0 - fluid.velocity[0] ** 2, 1.0e-300)
        )
        w_d = 1.0 / np.sqrt(
            np.maximum(1.0 - carrier.velocity[0] ** 2, 1.0e-300)
        )
        primitive = np.stack(
            (
                np.log(fluid.baryon_density),
                w_n * fluid.velocity[0],
                np.log(carrier.number_density),
                w_d * carrier.velocity[0],
                entropy,
            ),
            axis=-1,
        )
        low, high = self._interfaces(state, flux, primitive)
        interface = self._limited_interface(state, low, high, path, dt)
        return self._rhs_from_interface(interface, path)

    def step(self, dt: float) -> None:
        self.minimum_positivity_theta = 1.0
        base = self.state.copy()
        first = base + dt * self._rhs(base.copy(), dt)
        self._recover(first)
        second = (
            0.75 * base
            + 0.25 * (first + dt * self._rhs(first.copy(), dt))
        )
        self._recover(second)
        final = (
            (1.0 / 3.0) * base
            + (2.0 / 3.0)
            * (second + dt * self._rhs(second.copy(), dt))
        )
        self._recover(final)
        self.state = final
        self.time += dt
        self.steps += 1

    def evolve(self, final_time: float, cfl: float) -> None:
        while self.time < final_time - 1.0e-15:
            dt = min(cfl * self.dx, final_time - self.time)
            self.step(dt)


def _relative_error(actual: Array, expected: Array) -> float:
    return float(
        np.linalg.norm(actual - expected)
        / max(np.linalg.norm(expected), 1.0e-30)
    )


def run_vulkan_theory33_frontier(
    resolutions: Iterable[int] = (16, 32, 64),
    *,
    final_time: float = 0.002,
    cfl: float = 0.1,
    module_path: str | Path | None = None,
) -> dict[str, object]:
    """Run CPU/GPU parity ladders without host reads during evolution."""
    values = tuple(int(value) for value in resolutions)
    if len(values) < 2 or any(value < 8 for value in values):
        raise ValueError("Vulkan ladder needs at least two resolutions")
    diagnostics: list[dict[str, object]] = []
    for points in values:
        primitives = smooth_theory33_primitives(points)
        cpu = CpuMaterialReference.initialize(primitives)
        initial_baryon = float(np.sum(cpu.state[:, 0]))
        initial_carrier = float(np.sum(cpu.state[:, 1]))
        gpu = VulkanTheory33Backend(
            points, module_path=module_path
        )
        state = gpu.initialize(primitives)
        reads_before = state.host_reads
        gpu.evolve(state, final_time=final_time, cfl=cfl)
        reads_after_evolution = state.host_reads
        snapshot = gpu.snapshot(state)
        cpu.evolve(final_time, cfl)

        field_absolute = {
            name: float(
                np.max(
                    np.abs(
                        snapshot.state[:, index]
                        - cpu.state[:, index]
                    )
                )
            )
            for index, name in enumerate(STATE_FIELDS)
        }
        field_relative = {
            name: _relative_error(
                snapshot.state[:, index], cpu.state[:, index]
            )
            for index, name in enumerate(STATE_FIELDS)
        }
        diagnostic = {
            "points": points,
            "steps": snapshot.steps,
            "host_reads_before_evolution": reads_before,
            "host_reads_after_evolution": reads_after_evolution,
            "host_reads_after_snapshot": state.host_reads,
            "maximum_absolute_field_errors": field_absolute,
            "relative_field_errors": field_relative,
            "physical_state_relative_error": _relative_error(
                snapshot.state[:, :5], cpu.state[:, :5]
            ),
            "maximum_recovery_residual": float(
                np.max(
                    snapshot.diagnostics[
                        :,
                        DIAGNOSTIC_FIELDS.index(
                            "recovery_residual"
                        ),
                    ]
                )
            ),
            "recovery_failures": int(
                np.count_nonzero(
                    snapshot.diagnostics[
                        :,
                        DIAGNOSTIC_FIELDS.index("recovery_valid"),
                    ]
                    < 0.5
                )
            ),
            "minimum_density": float(
                np.min(
                    snapshot.diagnostics[
                        :,
                        DIAGNOSTIC_FIELDS.index("baryon_density"),
                    ]
                )
            ),
            "minimum_carrier_density": float(
                np.min(
                    snapshot.diagnostics[
                        :,
                        DIAGNOSTIC_FIELDS.index("carrier_density"),
                    ]
                )
            ),
            "maximum_relative_lorentz_factor": float(
                np.max(
                    snapshot.diagnostics[
                        :,
                        DIAGNOSTIC_FIELDS.index(
                            "relative_lorentz_factor"
                        ),
                    ]
                )
            ),
            "minimum_legendre_eigenvalue": float(
                np.min(
                    snapshot.diagnostics[
                        :,
                        DIAGNOSTIC_FIELDS.index(
                            "minimum_legendre_eigenvalue"
                        ),
                    ]
                )
            ),
            "minimum_thermodynamic_eigenvalue": float(
                np.min(
                    snapshot.diagnostics[
                        :,
                        DIAGNOSTIC_FIELDS.index(
                            "minimum_thermodynamic_eigenvalue"
                        ),
                    ]
                )
            ),
            "baryon_relative_balance_error": abs(
                float(np.sum(snapshot.state[:, 0])) - initial_baryon
            )
            / abs(initial_baryon),
            "carrier_relative_balance_error": abs(
                float(np.sum(snapshot.state[:, 1])) - initial_carrier
            )
            / abs(initial_carrier),
        }
        diagnostic["qualification"] = {
            "device_resident_between_snapshots": (
                reads_before == reads_after_evolution == 0
            ),
            "cpu_gpu_physical_parity": (
                diagnostic["physical_state_relative_error"] < 2.0e-5
            ),
            "recovery_valid": (
                diagnostic["recovery_failures"] == 0
                and diagnostic["maximum_recovery_residual"] < 5.0e-4
            ),
            "densities_positive": (
                diagnostic["minimum_density"] > 0.0
                and diagnostic["minimum_carrier_density"] > 0.0
            ),
            "counterflow_domain_valid": (
                diagnostic["maximum_relative_lorentz_factor"] <= 2.0
            ),
            "legendre_margin_positive": (
                diagnostic["minimum_legendre_eigenvalue"] > 0.0
            ),
            "thermodynamic_margin_positive": (
                diagnostic["minimum_thermodynamic_eigenvalue"] > 0.0
            ),
            "periodic_number_balances_close": (
                diagnostic["baryon_relative_balance_error"] < 2.0e-6
                and diagnostic["carrier_relative_balance_error"] < 2.0e-6
            ),
        }
        diagnostics.append(diagnostic)

    qualification = {
        f"{points}_{name}": bool(value)
        for points, diagnostic in zip(values, diagnostics, strict=True)
        for name, value in diagnostic["qualification"].items()
    }
    return {
        "schema": "vulkan-tesseract.theory33-gpu-resident.v1",
        "backend": "Slang/Vulkan",
        "precision": "float32 device / float64 CPU oracle",
        "spacetime": "flat periodic 1D",
        "constitutive_model": "analytic_m1",
        "reconstruction": "piecewise_constant_rusanov",
        "path": "literal midpoint carrier canonical-momentum product",
        "integrator": "SSP-RK3 with stage-local energy recovery",
        "state_fields": list(STATE_FIELDS),
        "resolutions": list(values),
        "final_time": final_time,
        "cfl": cfl,
        "diagnostics": diagnostics,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def run_vulkan_frozen_neural_frontier(
    resolutions: Iterable[int] = (16, 32, 64),
    *,
    final_time: float = 0.002,
    cfl: float = 0.1,
    reconstruction: str = "piecewise_constant",
    positivity_preserving: bool = False,
    problem: str = "moving_phase_contact",
    module_path: str | Path | None = None,
) -> dict[str, object]:
    """Qualify GPU-resident frozen-neural phase evolution against the CPU."""
    values = tuple(int(value) for value in resolutions)
    if len(values) < 2 or any(value < 8 for value in values):
        raise ValueError("Vulkan ladder needs at least two resolutions")
    if problem not in {
        "moving_phase_contact",
        "original_neural_phase_riemann",
    }:
        raise ValueError("unknown frozen-neural phase problem")
    primitive_factory = (
        original_neural_phase_riemann_primitives
        if problem == "original_neural_phase_riemann"
        else neural_phase_contact_primitives
    )
    closure = QualifiedFrozenClosure()
    diagnostics: list[dict[str, object]] = []
    for points in values:
        primitives = primitive_factory(points)
        cpu = CpuMaterialReference.initialize(
            primitives,
            constitutive_model="qualified_frozen",
            reconstruction=reconstruction,
            positivity_preserving=positivity_preserving,
        )
        initial_baryon = float(np.sum(cpu.state[:, 0]))
        initial_carrier = float(np.sum(cpu.state[:, 1]))
        gpu = VulkanTheory33Backend(
            points,
            constitutive_model="qualified_frozen",
            reconstruction=reconstruction,
            positivity_preserving=positivity_preserving,
            module_path=module_path,
        )
        state = gpu.initialize(primitives)
        initial_snapshot = gpu.snapshot(state)
        initial_phase = (
            initial_snapshot.diagnostic_field(
                "constitutive_phase_two"
            )
            >= 0.5
        )
        reads_before_evolution = state.host_reads
        gpu.evolve(state, final_time=final_time, cfl=cfl)
        reads_after_evolution = state.host_reads
        snapshot = gpu.snapshot(state)
        cpu.evolve(final_time, cfl)
        fluid, carrier, master, _ = cpu._recover(cpu.state)

        gpu_phase = (
            snapshot.diagnostic_field("constitutive_phase_two") >= 0.5
        )
        cpu_phase = np.asarray(master.phase_two, dtype=bool)
        gpu_margin = snapshot.diagnostic_field(
            "constitutive_switching_margin"
        )
        cpu_margin = (
            np.asarray(master.relative_lorentz_factor)
            - 1.0
            - np.asarray(master.phase_threshold)
        )
        physical_error = _relative_error(
            snapshot.state[:, :5], cpu.state[:, :5]
        )
        phase_disagreements = int(
            np.count_nonzero(gpu_phase != cpu_phase)
        )
        disagreement = gpu_phase != cpu_phase
        disagreement_distance = (
            float(
                np.max(
                    np.minimum(
                        np.abs(gpu_margin[disagreement]),
                        np.abs(cpu_margin[disagreement]),
                    )
                )
            )
            if phase_disagreements
            else 0.0
        )
        initial_phase_count = int(np.count_nonzero(initial_phase))
        final_phase_count = int(np.count_nonzero(gpu_phase))
        diagnostic: dict[str, object] = {
            "points": points,
            "steps": snapshot.steps,
            "host_reads_before_evolution": reads_before_evolution,
            "host_reads_after_evolution": reads_after_evolution,
            "physical_state_relative_error": physical_error,
            "phase_disagreements_with_cpu": phase_disagreements,
            "maximum_disagreement_switching_distance": (
                disagreement_distance
            ),
            "initial_phase_two_cells": initial_phase_count,
            "final_phase_two_cells": final_phase_count,
            "phase_cells_changed_from_initial": int(
                np.count_nonzero(gpu_phase != initial_phase)
            ),
            "maximum_switching_margin_error": float(
                np.max(np.abs(gpu_margin - cpu_margin))
            ),
            "minimum_absolute_switching_margin": float(
                np.min(np.abs(gpu_margin))
            ),
            "maximum_recovery_residual": float(
                np.max(
                    snapshot.diagnostic_field("recovery_residual")
                )
            ),
            "recovery_failures": int(
                np.count_nonzero(
                    snapshot.diagnostic_field("recovery_valid") < 0.5
                )
            ),
            "recovery_phase_crossings": int(
                np.sum(
                    snapshot.diagnostic_field(
                        "recovery_phase_crossings"
                    )
                )
            ),
            "minimum_density": float(
                np.min(snapshot.diagnostic_field("baryon_density"))
            ),
            "minimum_carrier_density": float(
                np.min(snapshot.diagnostic_field("carrier_density"))
            ),
            "minimum_legendre_eigenvalue": float(
                np.min(
                    snapshot.diagnostic_field(
                        "minimum_legendre_eigenvalue"
                    )
                )
            ),
            "minimum_thermodynamic_eigenvalue": float(
                np.min(
                    snapshot.diagnostic_field(
                        "minimum_thermodynamic_eigenvalue"
                    )
                )
            ),
            "baryon_relative_balance_error": abs(
                float(np.sum(snapshot.state[:, 0])) - initial_baryon
            )
            / abs(initial_baryon),
            "carrier_relative_balance_error": abs(
                float(np.sum(snapshot.state[:, 1])) - initial_carrier
            )
            / abs(initial_carrier),
            "cpu_velocity_norm": float(
                max(
                    np.max(np.abs(fluid.velocity)),
                    np.max(np.abs(carrier.velocity)),
                )
            ),
        }
        diagnostic["qualification"] = {
            "device_resident_between_snapshots": (
                reads_before_evolution == reads_after_evolution
            ),
            "cpu_gpu_physical_parity": physical_error < 3.0e-5,
            "hard_phase_agrees_away_from_boundary": (
                disagreement_distance < 1.0e-4
            ),
            "opposite_phases_remain_represented": (
                0 < final_phase_count < points
            ),
            "recovery_valid": (
                diagnostic["recovery_failures"] == 0
                and diagnostic["maximum_recovery_residual"] < 5.0e-4
            ),
            "densities_positive": (
                diagnostic["minimum_density"] > 0.0
                and diagnostic["minimum_carrier_density"] > 0.0
            ),
            "legendre_margin_positive": (
                diagnostic["minimum_legendre_eigenvalue"] > 0.0
            ),
            "thermodynamic_margin_positive": (
                diagnostic["minimum_thermodynamic_eigenvalue"] > 0.0
            ),
            "periodic_number_balances_close": (
                diagnostic["baryon_relative_balance_error"] < 2.0e-6
                and diagnostic["carrier_relative_balance_error"] < 2.0e-6
            ),
        }
        diagnostics.append(diagnostic)

    qualification = {
        f"{points}_{name}": bool(value)
        for points, diagnostic in zip(values, diagnostics, strict=True)
        for name, value in diagnostic["qualification"].items()
    }
    qualification["phase_crossing_exercised"] = any(
        int(diagnostic["phase_cells_changed_from_initial"]) > 0
        for diagnostic in diagnostics
    )
    return {
        "schema": "vulkan-tesseract.frozen-neural-gpu-resident.v1",
        "backend": "Slang/Vulkan",
        "precision": "float32 device / float64 CPU oracle",
        "spacetime": "flat periodic 1D",
        "problem": problem,
        "constitutive_model": "qualified_frozen",
        "variant_digest": closure.variant_digest,
        "phase_semantics": (
            "hard branch frozen within Newton/Jacobian; reclassified "
            "between recovery transition iterations"
        ),
        "reconstruction": reconstruction,
        "positivity_preserving": positivity_preserving,
        "path": "literal midpoint carrier canonical-momentum product",
        "integrator": "SSP-RK3 with stage-local energy recovery",
        "resolutions": list(values),
        "final_time": final_time,
        "cfl": cfl,
        "diagnostics": diagnostics,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def positivity_torture_primitives(points: int = 32) -> Array:
    coordinate = (np.arange(points, dtype=float) + 0.5) / points
    left = coordinate < 0.5
    values = np.zeros((points, 6), dtype=np.float64)
    values[:, 0] = np.where(left, 0.18, 0.06)
    values[:, 1] = np.where(left, 0.5, 0.05)
    values[:, 2] = np.where(left, 0.4, -0.3)
    values[:, 3] = np.where(left, 0.009, 0.003)
    values[:, 4] = np.where(left, 0.5, -0.4)
    values[:, 5] = 0.2
    return values


def _weno_interface_accuracy() -> dict[str, object]:
    resolutions = (8, 16, 32)
    errors: list[float] = []
    for points in resolutions:
        dx = 1.0 / points
        coordinate = (np.arange(points, dtype=float) + 0.5) / points
        cell_average_mode = np.sinc(dx) * np.sin(
            2.0 * np.pi * coordinate
        )
        primitives = np.zeros((points, 6), dtype=np.float64)
        primitives[:, 0] = 0.1
        primitives[:, 1] = 0.12
        primitives[:, 2] = 0.2
        primitives[:, 3] = 0.005
        primitives[:, 4] = 0.2
        primitives[:, 5] = 0.2 + 0.01 * cell_average_mode
        backend = VulkanTheory33Backend(
            points, reconstruction="weno5_z"
        )
        state = backend.initialize(primitives)
        interface = backend.interface_snapshot(state).high[:, 6]
        face = (np.arange(points, dtype=float) + 1.0) / points
        conserved_density = 0.1 / np.sqrt(1.0 - 0.2**2)
        exact = (
            conserved_density
            * (0.2 + 0.01 * np.sin(2.0 * np.pi * face))
            * 0.2
        )
        errors.append(float(np.max(np.abs(interface - exact))))
    orders = [
        float(np.log2(coarse / fine))
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
    ]
    return {
        "resolutions": list(resolutions),
        "maximum_errors": errors,
        "observed_orders": orders,
        "qualification": {
            "pre_roundoff_order_exceeds_4_5": min(orders) > 4.5,
            "error_decreases": all(
                fine < coarse
                for coarse, fine in zip(
                    errors[:-1], errors[1:], strict=True
                )
            ),
        },
    }


def run_vulkan_weno_frontier(
    smooth_resolutions: Iterable[int] = (16, 32, 64),
    phase_resolutions: Iterable[int] = (16, 32),
    *,
    smooth_final_time: float = 0.002,
    phase_final_time: float = 0.1,
) -> dict[str, object]:
    """Qualify device WENO5-Z and conservative positivity blending."""
    smooth_values = tuple(int(value) for value in smooth_resolutions)
    phase_values = tuple(int(value) for value in phase_resolutions)
    accuracy = _weno_interface_accuracy()
    smooth: list[dict[str, object]] = []
    for points in smooth_values:
        primitives = smooth_theory33_primitives(points)
        cpu = CpuMaterialReference.initialize(
            primitives,
            reconstruction="weno5_z",
            positivity_preserving=True,
        )
        gpu = VulkanTheory33Backend(
            points,
            reconstruction="weno5_z",
            positivity_preserving=True,
        )
        state = gpu.initialize(primitives)
        initial_baryon = float(np.sum(cpu.state[:, 0]))
        initial_carrier = float(np.sum(cpu.state[:, 1]))
        gpu.evolve(state, final_time=smooth_final_time, cfl=0.1)
        reads = state.host_reads
        snapshot = gpu.snapshot(state)
        cpu.evolve(smooth_final_time, 0.1)
        smooth.append(
            {
                "points": points,
                "physical_state_relative_error": _relative_error(
                    snapshot.state[:, :5], cpu.state[:, :5]
                ),
                "minimum_positivity_theta": float(
                    np.min(snapshot.positivity_theta)
                ),
                "recovery_failures": int(
                    np.count_nonzero(
                        snapshot.diagnostic_field("recovery_valid") < 0.5
                    )
                ),
                "host_reads_after_evolution": reads,
                "baryon_relative_balance_error": abs(
                    float(np.sum(snapshot.state[:, 0])) - initial_baryon
                )
                / abs(initial_baryon),
                "carrier_relative_balance_error": abs(
                    float(np.sum(snapshot.state[:, 1])) - initial_carrier
                )
                / abs(initial_carrier),
            }
        )

    factor = 2.8e5
    torture_primitives = positivity_torture_primitives()
    torture_cpu = CpuMaterialReference.initialize(
        torture_primitives,
        reconstruction="weno5_z",
        positivity_preserving=True,
        positivity_floor_factor=factor,
    )
    torture_gpu = VulkanTheory33Backend(
        32,
        reconstruction="weno5_z",
        positivity_preserving=True,
        positivity_floor_factor=factor,
    )
    torture_state = torture_gpu.initialize(torture_primitives)
    initial_torture_baryon = float(np.sum(torture_cpu.state[:, 0]))
    initial_torture_carrier = float(np.sum(torture_cpu.state[:, 1]))
    torture_dt = 0.8 / 32.0
    torture_gpu.step(torture_state, torture_dt)
    torture_snapshot = torture_gpu.snapshot(torture_state)
    torture_cpu.step(torture_dt)
    torture = {
        "points": 32,
        "cfl": 0.8,
        "positivity_floor_factor": factor,
        "physical_state_relative_error": _relative_error(
            torture_snapshot.state[:, :5], torture_cpu.state[:, :5]
        ),
        "gpu_minimum_theta": float(
            np.min(torture_snapshot.positivity_theta)
        ),
        "cpu_minimum_theta": torture_cpu.minimum_positivity_theta,
        "recovery_failures": int(
            np.count_nonzero(
                torture_snapshot.diagnostic_field("recovery_valid") < 0.5
            )
        ),
        "minimum_baryon_conserved": float(
            np.min(torture_snapshot.state[:, 0])
        ),
        "minimum_carrier_conserved": float(
            np.min(torture_snapshot.state[:, 1])
        ),
        "baryon_balance_error": abs(
            float(np.sum(torture_snapshot.state[:, 0]))
            - initial_torture_baryon
        ),
        "carrier_balance_error": abs(
            float(np.sum(torture_snapshot.state[:, 1]))
            - initial_torture_carrier
        ),
    }

    phase: list[dict[str, object]] = []
    for points in phase_values:
        primitives = neural_phase_contact_primitives(points)
        options = {
            "constitutive_model": "qualified_frozen",
            "reconstruction": "weno5_z",
            "positivity_preserving": True,
        }
        cpu = CpuMaterialReference.initialize(primitives, **options)
        gpu = VulkanTheory33Backend(points, **options)
        state = gpu.initialize(primitives)
        initial = gpu.snapshot(state).diagnostic_field(
            "constitutive_phase_two"
        ) >= 0.5
        reads_before = state.host_reads
        gpu.evolve(state, final_time=phase_final_time, cfl=0.12)
        reads_after = state.host_reads
        snapshot = gpu.snapshot(state)
        cpu.evolve(phase_final_time, 0.12)
        _, _, master, _ = cpu._recover(cpu.state)
        gpu_phase = (
            snapshot.diagnostic_field("constitutive_phase_two") >= 0.5
        )
        phase.append(
            {
                "points": points,
                "physical_state_relative_error": _relative_error(
                    snapshot.state[:, :5], cpu.state[:, :5]
                ),
                "phase_disagreements_with_cpu": int(
                    np.count_nonzero(
                        gpu_phase != np.asarray(master.phase_two)
                    )
                ),
                "phase_cells_changed_from_initial": int(
                    np.count_nonzero(gpu_phase != initial)
                ),
                "minimum_positivity_theta": float(
                    np.min(snapshot.positivity_theta)
                ),
                "recovery_failures": int(
                    np.count_nonzero(
                        snapshot.diagnostic_field("recovery_valid") < 0.5
                    )
                ),
                "host_reads_before_evolution": reads_before,
                "host_reads_after_evolution": reads_after,
            }
        )

    qualification = {
        **{
            f"interface_{name}": bool(value)
            for name, value in accuracy["qualification"].items()
        },
        "smooth_cpu_gpu_parity": all(
            item["physical_state_relative_error"] < 3.0e-5
            for item in smooth
        ),
        "smooth_recovery_valid": all(
            item["recovery_failures"] == 0 for item in smooth
        ),
        "smooth_unlimited_when_admissible": all(
            item["minimum_positivity_theta"] == 1.0 for item in smooth
        ),
        "smooth_number_balance": all(
            item["baryon_relative_balance_error"] < 2.0e-6
            and item["carrier_relative_balance_error"] < 2.0e-6
            for item in smooth
        ),
        "torture_limiter_activates": (
            0.0 < torture["gpu_minimum_theta"] < 1.0
        ),
        "torture_cpu_gpu_parity": (
            torture["physical_state_relative_error"] < 3.0e-5
        ),
        "torture_recovery_and_positivity": (
            torture["recovery_failures"] == 0
            and torture["minimum_baryon_conserved"] > 0.0
            and torture["minimum_carrier_conserved"] > 0.0
        ),
        "torture_conservation": (
            torture["baryon_balance_error"] < 2.0e-6
            and torture["carrier_balance_error"] < 2.0e-6
        ),
        "phase_cpu_gpu_parity": all(
            item["physical_state_relative_error"] < 3.0e-5
            and item["phase_disagreements_with_cpu"] == 0
            for item in phase
        ),
        "phase_crossing_exercised": any(
            item["phase_cells_changed_from_initial"] > 0
            for item in phase
        ),
        "device_resident_between_snapshots": all(
            item["host_reads_before_evolution"]
            == item["host_reads_after_evolution"]
            for item in phase
        )
        and all(item["host_reads_after_evolution"] == 0 for item in smooth),
    }
    return {
        "schema": "vulkan-tesseract.weno-positivity.v1",
        "backend": "Slang/Vulkan",
        "reconstruction": "componentwise_weno5_z",
        "low_order_control": "piecewise_constant_rusanov",
        "positivity": (
            "cell forward-Euler admissibility with minimum adjacent "
            "face blending"
        ),
        "interface_accuracy": accuracy,
        "smooth_ladder": smooth,
        "positivity_torture": torture,
        "frozen_neural_phase_ladder": phase,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def _cpu_face_eigenvalues(
    solver: Theory33ProductionSolver,
    recovery: Array,
) -> tuple[Array, int, float, float]:
    count = recovery.shape[0]
    values = np.empty((count, 5))
    phase_faces = 0
    maximum_imaginary = 0.0
    maximum_condition = 0.0
    for face in range(count):
        following = (face + 1) % count
        left_x = recovery[face]
        right_x = recovery[following]
        face_x = 0.5 * (left_x + right_x)
        _, _, _, left_phase, left_margin = _restricted_prediction(
            solver, left_x, None
        )
        _, _, _, right_phase, right_margin = _restricted_prediction(
            solver, right_x, None
        )
        phase_override: bool | None = None
        if (
            solver.master.constitutive_closure is not None
            and left_phase != right_phase
        ):
            phase_faces += 1
            if abs(left_margin) >= abs(right_margin):
                face_x = left_x
                phase_override = left_phase
            else:
                face_x = right_x
                phase_override = right_phase
        system = restricted_carter_eigensystem(
            solver, face_x, phase_override=phase_override
        )
        eigenvalues = np.asarray(system["eigenvalues"])
        values[face] = eigenvalues.real
        maximum_imaginary = max(
            maximum_imaginary,
            float(system["maximum_imaginary_part"]),
        )
        maximum_condition = max(
            maximum_condition, float(system["condition"])
        )
    return values, phase_faces, maximum_imaginary, maximum_condition


def run_vulkan_carter_eigensystem_frontier(
    smooth_resolutions: Iterable[int] = (16, 32, 64),
    shock_resolutions: Iterable[int] = (16, 32),
    phase_resolutions: Iterable[int] = (16, 32),
) -> dict[str, object]:
    """Qualify the numerical Carter flux/path basis and characteristic WENO."""
    smooth_values = tuple(int(value) for value in smooth_resolutions)
    shock_values = tuple(int(value) for value in shock_resolutions)
    phase_values = tuple(int(value) for value in phase_resolutions)
    options = {
        "reconstruction": "carter_characteristic_weno5_z",
        "positivity_preserving": True,
    }

    symbol_audits: list[dict[str, object]] = []
    for model, primitives in (
        ("analytic_m1", smooth_theory33_primitives(16)),
        ("qualified_frozen", neural_phase_contact_primitives(16)),
    ):
        backend = VulkanTheory33Backend(
            16, constitutive_model=model, **options
        )
        snapshot = backend.snapshot(backend.initialize(primitives))
        cpu = CpuMaterialReference.initialize(
            primitives, constitutive_model=model, **options
        )
        cpu_values, cpu_phase_faces, imaginary, cpu_condition = (
            _cpu_face_eigenvalues(cpu.solver, snapshot.recovery)
        )
        symbol_audits.append(
            {
                "constitutive_model": model,
                "maximum_eigenvalue_error": float(
                    np.max(np.abs(snapshot.eigenvalues - cpu_values))
                ),
                "maximum_cpu_imaginary_part": imaginary,
                "maximum_cpu_condition": cpu_condition,
                "maximum_gpu_speed": float(
                    np.max(
                        snapshot.eigen_audit_field(
                            "maximum_characteristic_speed"
                        )
                    )
                ),
                "maximum_gpu_eigenpair_residual": float(
                    np.max(
                        snapshot.eigen_audit_field(
                            "maximum_eigenpair_residual"
                        )
                    )
                ),
                "maximum_gpu_inverse_residual": float(
                    np.max(
                        snapshot.eigen_audit_field(
                            "inverse_basis_residual"
                        )
                    )
                ),
                "maximum_gpu_condition_estimate": float(
                    np.max(
                        snapshot.eigen_audit_field(
                            "condition_estimate"
                        )
                    )
                ),
                "minimum_gpu_separation": float(
                    np.min(
                        snapshot.eigen_audit_field(
                            "minimum_eigenvalue_separation"
                        )
                    )
                ),
                "gpu_invalid_faces": int(
                    np.count_nonzero(
                        snapshot.eigen_audit_field(
                            "eigensystem_valid"
                        )
                        < 0.5
                    )
                ),
                "gpu_phase_boundary_faces": int(
                    np.count_nonzero(
                        snapshot.eigen_audit_field(
                            "phase_boundary_face"
                        )
                    )
                ),
                "cpu_phase_boundary_faces": cpu_phase_faces,
            }
        )

    smooth: list[dict[str, object]] = []
    for points in smooth_values:
        primitives = smooth_theory33_primitives(points)
        cpu = CpuMaterialReference.initialize(primitives, **options)
        gpu = VulkanTheory33Backend(points, **options)
        state = gpu.initialize(primitives)
        gpu.evolve(state, final_time=0.002, cfl=0.1)
        reads = state.host_reads
        snapshot = gpu.snapshot(state)
        cpu.evolve(0.002, 0.1)
        smooth.append(
            {
                "points": points,
                "physical_state_relative_error": _relative_error(
                    snapshot.state[:, :5], cpu.state[:, :5]
                ),
                "maximum_speed": float(
                    np.max(
                        snapshot.eigen_audit_field(
                            "maximum_characteristic_speed"
                        )
                    )
                ),
                "maximum_eigenpair_residual": float(
                    np.max(
                        snapshot.eigen_audit_field(
                            "maximum_eigenpair_residual"
                        )
                    )
                ),
                "invalid_eigensystems": int(
                    np.count_nonzero(
                        snapshot.eigen_audit_field(
                            "eigensystem_valid"
                        )
                        < 0.5
                    )
                ),
                "recovery_failures": int(
                    np.count_nonzero(
                        snapshot.diagnostic_field("recovery_valid") < 0.5
                    )
                ),
                "minimum_positivity_theta": float(
                    np.min(snapshot.positivity_theta)
                ),
                "host_reads_after_evolution": reads,
            }
        )

    shock: list[dict[str, object]] = []
    for points in shock_values:
        primitives = positivity_torture_primitives(points)
        cpu = CpuMaterialReference.initialize(primitives, **options)
        gpu = VulkanTheory33Backend(points, **options)
        state = gpu.initialize(primitives)
        initial_baryon = float(np.sum(cpu.state[:, 0]))
        initial_carrier = float(np.sum(cpu.state[:, 1]))
        gpu.evolve(state, final_time=0.005, cfl=0.1)
        snapshot = gpu.snapshot(state)
        cpu.evolve(0.005, 0.1)
        shock.append(
            {
                "points": points,
                "physical_state_relative_error": _relative_error(
                    snapshot.state[:, :5], cpu.state[:, :5]
                ),
                "maximum_speed": float(
                    np.max(
                        snapshot.eigen_audit_field(
                            "maximum_characteristic_speed"
                        )
                    )
                ),
                "maximum_eigenpair_residual": float(
                    np.max(
                        snapshot.eigen_audit_field(
                            "maximum_eigenpair_residual"
                        )
                    )
                ),
                "invalid_eigensystems": int(
                    np.count_nonzero(
                        snapshot.eigen_audit_field(
                            "eigensystem_valid"
                        )
                        < 0.5
                    )
                ),
                "recovery_failures": int(
                    np.count_nonzero(
                        snapshot.diagnostic_field("recovery_valid") < 0.5
                    )
                ),
                "baryon_balance_error": abs(
                    float(np.sum(snapshot.state[:, 0]))
                    - initial_baryon
                ),
                "carrier_balance_error": abs(
                    float(np.sum(snapshot.state[:, 1]))
                    - initial_carrier
                ),
            }
        )

    phase: list[dict[str, object]] = []
    phase_options = {
        **options,
        "constitutive_model": "qualified_frozen",
    }
    for points in phase_values:
        primitives = neural_phase_contact_primitives(points)
        cpu = CpuMaterialReference.initialize(primitives, **phase_options)
        gpu = VulkanTheory33Backend(points, **phase_options)
        state = gpu.initialize(primitives)
        initial_snapshot = gpu.snapshot(state)
        initial_phase = (
            initial_snapshot.diagnostic_field(
                "constitutive_phase_two"
            )
            >= 0.5
        )
        reads_before = state.host_reads
        gpu.evolve(state, final_time=0.1, cfl=0.1)
        reads_after = state.host_reads
        snapshot = gpu.snapshot(state)
        cpu.evolve(0.1, 0.1)
        _, _, master, _ = cpu._recover(cpu.state)
        gpu_phase = (
            snapshot.diagnostic_field("constitutive_phase_two") >= 0.5
        )
        phase.append(
            {
                "points": points,
                "physical_state_relative_error": _relative_error(
                    snapshot.state[:, :5], cpu.state[:, :5]
                ),
                "phase_disagreements_with_cpu": int(
                    np.count_nonzero(
                        gpu_phase != np.asarray(master.phase_two)
                    )
                ),
                "phase_cells_changed_from_initial": int(
                    np.count_nonzero(gpu_phase != initial_phase)
                ),
                "phase_boundary_faces": int(
                    np.count_nonzero(
                        snapshot.eigen_audit_field(
                            "phase_boundary_face"
                        )
                    )
                ),
                "maximum_speed": float(
                    np.max(
                        snapshot.eigen_audit_field(
                            "maximum_characteristic_speed"
                        )
                    )
                ),
                "maximum_eigenpair_residual": float(
                    np.max(
                        snapshot.eigen_audit_field(
                            "maximum_eigenpair_residual"
                        )
                    )
                ),
                "invalid_eigensystems": int(
                    np.count_nonzero(
                        snapshot.eigen_audit_field(
                            "eigensystem_valid"
                        )
                        < 0.5
                    )
                ),
                "recovery_failures": int(
                    np.count_nonzero(
                        snapshot.diagnostic_field("recovery_valid") < 0.5
                    )
                ),
                "host_reads_before_evolution": reads_before,
                "host_reads_after_evolution": reads_after,
            }
        )

    qualification = {
        "cpu_gpu_eigenvalues_match": all(
            item["maximum_eigenvalue_error"] < 1.0e-3
            for item in symbol_audits
        ),
        "symbols_real_and_causal": all(
            item["maximum_cpu_imaginary_part"] < 1.0e-8
            and item["maximum_gpu_speed"] <= 1.0 + 2.0e-5
            for item in symbol_audits
        ),
        "basis_residuals_and_conditioning_valid": all(
            item["gpu_invalid_faces"] == 0
            and item["maximum_gpu_eigenpair_residual"] < 2.0e-3
            and item["maximum_gpu_inverse_residual"] < 2.0e-3
            and item["minimum_gpu_separation"] > 1.0e-5
            for item in symbol_audits
        ),
        "branch_aware_face_selection_matches": (
            symbol_audits[1]["gpu_phase_boundary_faces"]
            == symbol_audits[1]["cpu_phase_boundary_faces"]
            == 2
        ),
        "smooth_characteristic_parity": all(
            item["physical_state_relative_error"] < 3.0e-5
            and item["invalid_eigensystems"] == 0
            and item["recovery_failures"] == 0
            and item["maximum_speed"] <= 1.0 + 2.0e-5
            for item in smooth
        ),
        "shock_characteristic_parity": all(
            item["physical_state_relative_error"] < 3.0e-5
            and item["invalid_eigensystems"] == 0
            and item["recovery_failures"] == 0
            and item["maximum_speed"] <= 1.0 + 2.0e-5
            and item["baryon_balance_error"] < 2.0e-6
            and item["carrier_balance_error"] < 2.0e-6
            for item in shock
        ),
        "neural_phase_characteristic_parity": all(
            item["physical_state_relative_error"] < 3.0e-5
            and item["phase_disagreements_with_cpu"] == 0
            and item["phase_boundary_faces"] >= 2
            and item["invalid_eigensystems"] == 0
            and item["recovery_failures"] == 0
            and item["maximum_speed"] <= 1.0 + 2.0e-5
            for item in phase
        ),
        "dynamic_phase_crossing_exercised": any(
            item["phase_cells_changed_from_initial"] > 0
            for item in phase
        ),
        "device_resident_between_snapshots": all(
            item["host_reads_before_evolution"]
            == item["host_reads_after_evolution"]
            for item in phase
        )
        and all(item["host_reads_after_evolution"] == 0 for item in smooth),
    }
    return {
        "schema": "vulkan-tesseract.carter-eigensystem.v1",
        "backend": "Slang/Vulkan",
        "physical_block": (
            "five-field flat-1D Carter conserved principal system"
        ),
        "symbol": "(F_,p - B_,p) @ inverse(U_,p)",
        "device_eigensolver": (
            "shifted QR eigenvalues plus inverse-iteration right vectors"
        ),
        "reconstruction": "branch-aware_carter_characteristic_weno5_z",
        "symbol_audits": symbol_audits,
        "smooth_ladder": smooth,
        "shock_ladder": shock,
        "frozen_neural_phase_ladder": phase,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def _full_neural_phase_snapshot(
    primitive: Array,
    phase_two: Array,
    gamma_ad: float,
) -> dict[str, Array]:
    density = np.exp(primitive[:, 0])
    entropy = primitive[:, 1]
    q_n = primitive[:, 2:5]
    carrier_density = np.exp(primitive[:, 5])
    q_d = primitive[:, 6:9]
    w_n = np.sqrt(1.0 + np.sum(q_n * q_n, axis=1))
    w_d = np.sqrt(1.0 + np.sum(q_d * q_d, axis=1))
    velocity_n = q_n[:, 0] / w_n
    velocity_d = q_d[:, 0] / w_d
    pressure = (
        np.exp((gamma_ad - 1.0) * entropy)
        * density**gamma_ad
    )
    return {
        "baryon_density": density,
        "fluid_velocity": velocity_n,
        "pressure": pressure,
        "carrier_density": carrier_density,
        "carrier_velocity": velocity_d,
        "relative_lorentz_factor": (
            w_n * w_d - np.sum(q_n * q_d, axis=1)
        ),
        "phase_two": phase_two.astype(float),
    }


def _full_neural_phase_error(
    coarse: dict[str, Array],
    fine: dict[str, Array],
) -> tuple[float, dict[str, float]]:
    fields: dict[str, float] = {}
    for name, coarse_value in coarse.items():
        fine_value = fine[name][::2]
        scale = max(
            float(np.mean(np.abs(fine_value))),
            float(np.max(fine_value) - np.min(fine_value)),
            1.0e-8,
        )
        fields[name] = float(
            np.mean(np.abs(coarse_value - fine_value)) / scale
        )
    physical = [
        value for name, value in fields.items() if name != "phase_two"
    ]
    return float(np.mean(physical)), fields


def vulkan_neural_phase_corridor_audit(
    samples: int = 129,
    *,
    module_path: str | Path | None = None,
) -> dict[str, object]:
    """Audit the straight conserved learned-phase path on Vulkan."""
    if samples < 17:
        raise ValueError("neural phase corridor needs at least 17 samples")
    primitive = original_neural_phase_riemann_full_primitives(
        samples
    ).astype(np.float32)
    geometry = np.zeros((samples, 13), dtype=np.float32)
    geometry[:, :9] = np.eye(3, dtype=np.float32).reshape(1, 9)
    geometry[:, 9] = 1.0
    backend = VulkanTheory33MultidimensionalBackend(
        (samples,),
        constitutive_model="qualified_frozen",
        reconstruction="full_carter_roe",
        module_path=module_path,
    )
    state = backend.initialize(
        primitive,
        np.zeros((samples, 20), dtype=np.float32),
        geometry,
        np.zeros(samples, dtype=np.float32),
    )
    backend.linearize_conserved_corridor(state)
    snapshot = backend.snapshot(state)
    backend._rhs_dispatch(state.tensor, state.primitive)
    phase = snapshot.diagnostics[:, 3] >= 0.5
    margin = snapshot.diagnostics[:, 4].astype(float)
    face_values = backend._full_face_eigenvalues.to_numpy()
    face_audit = backend._full_face_audit.to_numpy()
    path = backend._full_face_path.to_numpy().reshape(
        samples, 9, 9
    )
    density = np.exp(snapshot.primitive[:, 0].astype(float))
    carrier = np.exp(snapshot.primitive[:, 5].astype(float))
    entropy = snapshot.primitive[:, 1].astype(float)
    q_n = snapshot.primitive[:, 2:5].astype(float)
    q_d = snapshot.primitive[:, 6:9].astype(float)
    w_n = np.sqrt(1.0 + np.sum(q_n * q_n, axis=1))
    w_d = np.sqrt(1.0 + np.sum(q_d * q_d, axis=1))
    metric = np.zeros((3, 3, samples), dtype=float)
    metric[0, 0] = metric[1, 1] = metric[2, 2] = 1.0
    source = _source_solver("qualified_frozen")
    gamma_ad = source.master_parameters.gamma_ad
    pressure = (
        np.exp((gamma_ad - 1.0) * entropy) * density**gamma_ad
    )
    master = source.master.evaluate(
        metric,
        FluidPrimitive(
            density,
            pressure / ((gamma_ad - 1.0) * density),
            (q_n / w_n[:, None]).T,
            np.zeros(samples),
        ),
        CarrierPrimitive(carrier, (q_d / w_d[:, None]).T),
    )
    legendre = np.stack(
        (
            np.stack((master.B_N, master.entrainment), axis=-1),
            np.stack((master.entrainment, master.B_D), axis=-1),
        ),
        axis=-2,
    )
    thermo = np.stack(
        (
            np.stack(
                (
                    master.thermodynamic_hessian_nn,
                    master.thermodynamic_hessian_nd,
                ),
                axis=-1,
            ),
            np.stack(
                (
                    master.thermodynamic_hessian_nd,
                    master.thermodynamic_hessian_dd,
                ),
                axis=-1,
            ),
        ),
        axis=-2,
    )
    report: dict[str, object] = {
        "samples": samples,
        "left_phase_two": bool(phase[0]),
        "right_phase_two": bool(phase[-1]),
        "phase_transitions": int(
            np.count_nonzero(phase[1:] != phase[:-1])
        ),
        "left_switching_margin": float(margin[0]),
        "right_switching_margin": float(margin[-1]),
        "minimum_absolute_switching_margin": float(
            np.min(np.abs(margin))
        ),
        "maximum_physical_speed": float(
            np.max(np.abs(face_values))
        ),
        "maximum_eigenpair_residual": float(
            np.max(face_audit[:, 1])
        ),
        "maximum_eigenvector_condition": float(
            np.max(face_audit[:, 3])
        ),
        "invalid_eigensystems": int(
            np.count_nonzero(face_audit[:, 4] < 0.5)
        ),
        "minimum_legendre_eigenvalue": float(
            np.min(np.linalg.eigvalsh(legendre))
        ),
        "minimum_thermodynamic_eigenvalue": float(
            np.min(np.linalg.eigvalsh(thermo))
        ),
        "maximum_recovery_residual": float(
            np.max(snapshot.diagnostics[:, 0])
        ),
        "path_symbol_minimum_norm": float(
            np.min(np.linalg.norm(path, axis=(1, 2)))
        ),
    }
    report["qualification"] = {
        "endpoints_on_opposite_learned_phases": (
            report["left_phase_two"]
            and not report["right_phase_two"]
        ),
        "single_hard_transition_on_path": (
            report["phase_transitions"] == 1
        ),
        "endpoint_switching_margins_healthy": (
            report["left_switching_margin"] > 2.0e-4
            and report["right_switching_margin"] < -2.0e-4
        ),
        "path_recovery_valid": (
            report["maximum_recovery_residual"] < 2.1e-6
        ),
        "path_symbols_causal": (
            report["maximum_physical_speed"] <= 1.0 + 2.0e-3
        ),
        "path_eigensystems_valid": (
            report["invalid_eigensystems"] == 0
        ),
        "path_eigenvectors_conditioned": (
            report["maximum_eigenvector_condition"] < 1.0e8
        ),
        "path_legendre_margin_positive": (
            report["minimum_legendre_eigenvalue"] > 0.0
        ),
        "path_thermodynamic_margin_positive": (
            report["minimum_thermodynamic_eigenvalue"] > 0.0
        ),
        "nonconservative_path_active": (
            report["path_symbol_minimum_norm"] > 1.0e-8
        ),
    }
    return report


def run_vulkan_original_neural_phase_riemann_frontier(
    resolutions: Iterable[int] = (16, 32, 64, 128),
    *,
    final_time: float = 0.02,
    cfl: float = 0.12,
    reference_path: str | Path = (
        "tesseract_neural_phase_riemann_results.json"
    ),
    module_path: str | Path | None = None,
) -> dict[str, object]:
    """Run the original learned-phase problem through the full GPU ABI."""
    values = tuple(int(value) for value in resolutions)
    if len(values) < 3 or any(value < 16 for value in values):
        raise ValueError("full Carter Riemann ladder is invalid")
    if any(fine != 2 * coarse for coarse, fine in zip(values, values[1:])):
        raise ValueError("resolutions must form a factor-two ladder")
    closure = QualifiedFrozenClosure()
    corridor = vulkan_neural_phase_corridor_audit(
        129, module_path=module_path
    )
    gamma_ad = closure.parameters.gamma_ad
    snapshots: list[dict[str, Array]] = []
    diagnostics: list[dict[str, object]] = []
    for points in values:
        primitive = original_neural_phase_riemann_full_primitives(
            points
        ).astype(np.float32)
        vector_input = np.zeros((points, 20), dtype=np.float32)
        geometry = np.zeros((points, 13), dtype=np.float32)
        geometry[:, :9] = np.eye(3, dtype=np.float32).reshape(1, 9)
        geometry[:, 9] = 1.0
        tracer = np.zeros(points, dtype=np.float32)
        backend = VulkanTheory33MultidimensionalBackend(
            (points,),
            constitutive_model="qualified_frozen",
            reconstruction="full_carter_roe",
            module_path=module_path,
        )
        state = backend.initialize(
            primitive, vector_input, geometry, tracer
        )
        initial = backend.snapshot(state)
        initial_phase = initial.diagnostics[:, 3] >= 0.5
        initial_baryon = float(np.sum(initial.state[:, 0]))
        initial_carrier = float(np.sum(initial.state[:, 1]))
        reads_before = state.host_reads
        while state.time < final_time - 1.0e-15:
            dt = min(
                cfl / points,
                final_time - state.time,
            )
            backend.step(state, dt, apply_drag=False)
        reads_after = state.host_reads
        final = backend.snapshot(state)
        face_eigenvalues = backend._full_face_eigenvalues.to_numpy()
        face_audit = backend._full_face_audit.to_numpy()
        phase = final.diagnostics[:, 3] >= 0.5
        snapshots.append(
            _full_neural_phase_snapshot(
                final.primitive, phase, gamma_ad
            )
        )

        density = np.exp(final.primitive[:, 0].astype(float))
        entropy = final.primitive[:, 1].astype(float)
        carrier_density = np.exp(
            final.primitive[:, 5].astype(float)
        )
        q_n = final.primitive[:, 2:5].astype(float)
        q_d = final.primitive[:, 6:9].astype(float)
        w_n = np.sqrt(1.0 + np.sum(q_n * q_n, axis=1))
        w_d = np.sqrt(1.0 + np.sum(q_d * q_d, axis=1))
        velocity_n = (q_n / w_n[:, None]).T
        velocity_d = (q_d / w_d[:, None]).T
        pressure = (
            np.exp((gamma_ad - 1.0) * entropy)
            * density**gamma_ad
        )
        internal = pressure / (
            (gamma_ad - 1.0) * density
        )
        metric = np.zeros((3, 3, points), dtype=float)
        metric[0, 0] = 1.0
        metric[1, 1] = 1.0
        metric[2, 2] = 1.0
        master = _source_solver("qualified_frozen").master.evaluate(
            metric,
            FluidPrimitive(
                density,
                internal,
                velocity_n,
                np.zeros(points, dtype=float),
            ),
            CarrierPrimitive(carrier_density, velocity_d),
        )
        legendre = np.empty((points, 2, 2), dtype=float)
        legendre[:, 0, 0] = master.B_N
        legendre[:, 0, 1] = master.entrainment
        legendre[:, 1, 0] = master.entrainment
        legendre[:, 1, 1] = master.B_D
        thermo = np.empty((points, 2, 2), dtype=float)
        thermo[:, 0, 0] = master.thermodynamic_hessian_nn
        thermo[:, 0, 1] = master.thermodynamic_hessian_nd
        thermo[:, 1, 0] = master.thermodynamic_hessian_nd
        thermo[:, 1, 1] = master.thermodynamic_hessian_dd
        diagnostic: dict[str, object] = {
            "points": points,
            "steps": final.steps,
            "host_reads_before_evolution": reads_before,
            "host_reads_after_evolution": reads_after,
            "phase_two_fraction": float(np.mean(phase)),
            "phase_boundary_faces": int(
                np.count_nonzero(phase != np.roll(phase, -1))
            ),
            "phase_cells_changed_from_initial": int(
                np.count_nonzero(phase != initial_phase)
            ),
            "minimum_switching_margin": float(
                np.min(np.abs(final.diagnostics[:, 4]))
            ),
            "maximum_recovery_residual": float(
                np.max(final.diagnostics[:, 0])
            ),
            "recovery_failures": int(
                np.count_nonzero(final.diagnostics[:, 1] < 0.5)
            ),
            "minimum_density": float(np.min(density)),
            "minimum_carrier_density": float(
                np.min(carrier_density)
            ),
            "minimum_legendre_eigenvalue": float(
                np.min(np.linalg.eigvalsh(legendre))
            ),
            "minimum_thermodynamic_eigenvalue": float(
                np.min(np.linalg.eigvalsh(thermo))
            ),
            "maximum_characteristic_speed": float(
                np.max(np.abs(face_eigenvalues))
            ),
            "maximum_eigenpair_residual": float(
                np.max(face_audit[:, 1])
            ),
            "maximum_inverse_residual": float(
                np.max(face_audit[:, 2])
            ),
            "maximum_condition_estimate": float(
                np.max(face_audit[:, 3])
            ),
            "invalid_eigensystems": int(
                np.count_nonzero(face_audit[:, 4] < 0.5)
            ),
            "eigensystem_phase_boundary_faces": int(
                np.count_nonzero(face_audit[:, 5] >= 0.5)
            ),
            "baryon_relative_balance_error": abs(
                float(np.sum(final.state[:, 0])) - initial_baryon
            )
            / abs(initial_baryon),
            "carrier_relative_balance_error": abs(
                float(np.sum(final.state[:, 1])) - initial_carrier
            )
            / abs(initial_carrier),
        }
        diagnostics.append(diagnostic)

    errors: list[float] = []
    field_errors: list[dict[str, float]] = []
    for coarse, fine in zip(snapshots, snapshots[1:]):
        error, per_field = _full_neural_phase_error(coarse, fine)
        errors.append(error)
        field_errors.append(per_field)
    orders = [
        math.log(errors[index] / errors[index + 1], 2.0)
        for index in range(len(errors) - 1)
        if errors[index] > 0.0 and errors[index + 1] > 0.0
    ]
    reference_file = Path(reference_path)
    reference = json.loads(reference_file.read_text(encoding="utf-8"))
    reference_ladder = reference["resolution_ladder"]
    reference_phase_changes = reference_ladder[
        "phase_changes_from_initial_count"
    ]
    phase_changes = [
        item["phase_cells_changed_from_initial"]
        for item in diagnostics
    ]
    qualification = {
        **{
            f"corridor_{name}": bool(value)
            for name, value in corridor["qualification"].items()
        },
        "device_resident_between_snapshots": all(
            item["host_reads_before_evolution"]
            == item["host_reads_after_evolution"]
            for item in diagnostics
        ),
        "all_recoveries_valid": all(
            item["recovery_failures"] == 0 for item in diagnostics
        ),
        "all_eigensystems_valid": all(
            item["invalid_eigensystems"] == 0 for item in diagnostics
        ),
        "all_symbols_causal": all(
            item["maximum_characteristic_speed"] <= 1.0 + 2.0e-3
            for item in diagnostics
        ),
        "legendre_margins_positive": all(
            item["minimum_legendre_eigenvalue"] > 0.0
            for item in diagnostics
        ),
        "thermodynamic_margins_positive": all(
            item["minimum_thermodynamic_eigenvalue"] > 0.0
            for item in diagnostics
        ),
        "positive_densities": all(
            item["minimum_density"] > 0.0
            and item["minimum_carrier_density"] > 0.0
            for item in diagnostics
        ),
        "both_learned_phases_present": all(
            0.0 < item["phase_two_fraction"] < 1.0
            for item in diagnostics
        ),
        "phase_interfaces_resolved": all(
            item["phase_boundary_faces"] >= 2
            for item in diagnostics
        ),
        "final_switching_margins_nonzero": all(
            item["minimum_switching_margin"] > 0.0
            for item in diagnostics
        ),
        "periodic_number_balances_close": all(
            item["baryon_relative_balance_error"] < 2.0e-6
            and item["carrier_relative_balance_error"] < 2.0e-6
            for item in diagnostics
        ),
        "successive_error_decreases": all(
            later < earlier
            for earlier, later in zip(errors, errors[1:])
        ),
        "shock_order_resolved": bool(orders) and orders[-1] > 0.35,
        "physical_phase_crossing_exercised": max(phase_changes) > 0,
        "reference_phase_change_pattern_matches": (
            phase_changes == reference_phase_changes
        ),
    }
    return {
        "schema": "vulkan-tesseract.full-neural-phase-riemann.v1",
        "backend": "Slang/Vulkan",
        "precision": "float32 device / float64 physical audit",
        "problem": "original_neural_phase_riemann",
        "physical_block": "nine Carter fields plus entropy and tracer",
        "reconstruction": "full_carter_roe",
        "path": "resident midpoint canonical-momentum DLM fluctuation",
        "variant_digest": closure.variant_digest,
        "reference_schema": reference["schema"],
        "corridor": corridor,
        "reference_phase_changes": reference_phase_changes,
        "resolutions": list(values),
        "final_time": final_time,
        "cfl": cfl,
        "successive_l1_errors": errors,
        "successive_field_l1_errors": field_errors,
        "observed_orders": orders,
        "phase_changes_from_initial_count": phase_changes,
        "diagnostics": diagnostics,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def _run_vulkan_analytic_riemann_ladder(
    problem: str,
    resolutions: tuple[int, ...],
    *,
    final_time: float,
    cfl: float,
    module_path: str | Path | None,
) -> dict[str, object]:
    gamma_ad = _source_solver("analytic_m1").master_parameters.gamma_ad
    snapshots: list[dict[str, Array]] = []
    diagnostics: list[dict[str, object]] = []
    for points in resolutions:
        primitive = carter_riemann_full_primitives(
            points, problem
        ).astype(np.float32)
        geometry = np.zeros((points, 13), dtype=np.float32)
        geometry[:, :9] = np.eye(3, dtype=np.float32).reshape(1, 9)
        geometry[:, 9] = 1.0
        backend = VulkanTheory33MultidimensionalBackend(
            (points,),
            constitutive_model="analytic_m1",
            reconstruction="full_carter_roe",
            module_path=module_path,
        )
        state = backend.initialize(
            primitive,
            np.zeros((points, 20), dtype=np.float32),
            geometry,
            np.zeros(points, dtype=np.float32),
        )
        initial = backend.snapshot(state)
        initial_baryon = float(np.sum(initial.state[:, 0]))
        initial_carrier = float(np.sum(initial.state[:, 1]))
        reads_before = state.host_reads
        while state.time < final_time - 1.0e-15:
            backend.step(
                state,
                min(cfl / points, final_time - state.time),
                apply_drag=False,
            )
        reads_after = state.host_reads
        final = backend.snapshot(state)
        phase = final.diagnostics[:, 3] >= 0.5
        snapshots.append(
            _full_neural_phase_snapshot(
                final.primitive, phase, gamma_ad
            )
        )
        density = np.exp(final.primitive[:, 0].astype(float))
        carrier = np.exp(final.primitive[:, 5].astype(float))
        entropy = final.primitive[:, 1].astype(float)
        q_n = final.primitive[:, 2:5].astype(float)
        q_d = final.primitive[:, 6:9].astype(float)
        w_n = np.sqrt(1.0 + np.sum(q_n * q_n, axis=1))
        w_d = np.sqrt(1.0 + np.sum(q_d * q_d, axis=1))
        metric = np.zeros((3, 3, points), dtype=float)
        metric[0, 0] = metric[1, 1] = metric[2, 2] = 1.0
        pressure = (
            np.exp((gamma_ad - 1.0) * entropy)
            * density**gamma_ad
        )
        master = _source_solver("analytic_m1").master.evaluate(
            metric,
            FluidPrimitive(
                density,
                pressure / (
                    (gamma_ad - 1.0) * density
                ),
                (q_n / w_n[:, None]).T,
                np.zeros(points),
            ),
            CarrierPrimitive(
                carrier, (q_d / w_d[:, None]).T
            ),
        )
        legendre = np.stack(
            (
                np.stack((master.B_N, master.entrainment), axis=-1),
                np.stack((master.entrainment, master.B_D), axis=-1),
            ),
            axis=-2,
        )
        thermo = np.stack(
            (
                np.stack(
                    (
                        master.thermodynamic_hessian_nn,
                        master.thermodynamic_hessian_nd,
                    ),
                    axis=-1,
                ),
                np.stack(
                    (
                        master.thermodynamic_hessian_nd,
                        master.thermodynamic_hessian_dd,
                    ),
                    axis=-1,
                ),
            ),
            axis=-2,
        )
        face_values = backend._full_face_eigenvalues.to_numpy()
        face_audit = backend._full_face_audit.to_numpy()
        diagnostics.append(
            {
                "points": points,
                "steps": final.steps,
                "host_reads_before_evolution": reads_before,
                "host_reads_after_evolution": reads_after,
                "recovery_failures": int(
                    np.count_nonzero(final.diagnostics[:, 1] < 0.5)
                ),
                "maximum_recovery_residual": float(
                    np.max(final.diagnostics[:, 0])
                ),
                "invalid_eigensystems": int(
                    np.count_nonzero(face_audit[:, 4] < 0.5)
                ),
                "maximum_characteristic_speed": float(
                    np.max(np.abs(face_values))
                ),
                "maximum_eigenpair_residual": float(
                    np.max(face_audit[:, 1])
                ),
                "minimum_density": float(np.min(density)),
                "minimum_carrier_density": float(np.min(carrier)),
                "minimum_legendre_eigenvalue": float(
                    np.min(np.linalg.eigvalsh(legendre))
                ),
                "minimum_thermodynamic_eigenvalue": float(
                    np.min(np.linalg.eigvalsh(thermo))
                ),
                "baryon_relative_balance_error": abs(
                    float(np.sum(final.state[:, 0])) - initial_baryon
                )
                / abs(initial_baryon),
                "carrier_relative_balance_error": abs(
                    float(np.sum(final.state[:, 1])) - initial_carrier
                )
                / abs(initial_carrier),
            }
        )
    errors: list[float] = []
    field_errors: list[dict[str, float]] = []
    for coarse, fine in zip(snapshots, snapshots[1:]):
        error, per_field = _full_neural_phase_error(coarse, fine)
        errors.append(error)
        field_errors.append(per_field)
    orders = [
        math.log(errors[index] / errors[index + 1], 2.0)
        for index in range(len(errors) - 1)
        if errors[index] > 0.0 and errors[index + 1] > 0.0
    ]
    qualification = {
        "device_resident_between_snapshots": all(
            item["host_reads_before_evolution"]
            == item["host_reads_after_evolution"]
            for item in diagnostics
        ),
        "all_recoveries_valid": all(
            item["recovery_failures"] == 0 for item in diagnostics
        ),
        "all_eigensystems_valid": all(
            item["invalid_eigensystems"] == 0 for item in diagnostics
        ),
        "all_symbols_causal": all(
            item["maximum_characteristic_speed"] <= 1.0 + 2.0e-3
            for item in diagnostics
        ),
        "legendre_margins_positive": all(
            item["minimum_legendre_eigenvalue"] > 0.0
            for item in diagnostics
        ),
        "thermodynamic_margins_positive": all(
            item["minimum_thermodynamic_eigenvalue"] > 0.0
            for item in diagnostics
        ),
        "positive_densities": all(
            item["minimum_density"] > 0.0
            and item["minimum_carrier_density"] > 0.0
            for item in diagnostics
        ),
        "periodic_number_balances_close": all(
            item["baryon_relative_balance_error"] < 2.0e-6
            and item["carrier_relative_balance_error"] < 2.0e-6
            for item in diagnostics
        ),
        "successive_error_decreases": all(
            later < earlier
            for earlier, later in zip(errors, errors[1:])
        ),
        "shock_order_resolved": bool(orders) and orders[-1] > 0.35,
    }
    return {
        "problem": problem,
        "resolutions": list(resolutions),
        "final_time": final_time,
        "cfl": cfl,
        "constitutive_model": "analytic_m1",
        "reconstruction": "full_carter_roe",
        "successive_l1_errors": errors,
        "successive_field_l1_errors": field_errors,
        "observed_orders": orders,
        "diagnostics": diagnostics,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def run_vulkan_carter_riemann_frontier(
    resolutions: Iterable[int] = (16, 32, 64, 128),
    *,
    final_time: float = 0.02,
    cfl: float = 0.15,
    reference_path: str | Path = "tesseract_carter_riemann_results.json",
    module_path: str | Path | None = None,
) -> dict[str, object]:
    """Reproduce both original analytic Carter shock ladders on Vulkan."""
    values = tuple(int(value) for value in resolutions)
    if len(values) < 3 or any(value < 16 for value in values):
        raise ValueError("full Carter Riemann ladder is invalid")
    if any(fine != 2 * coarse for coarse, fine in zip(values, values[1:])):
        raise ValueError("resolutions must form a factor-two ladder")
    counterflow = _run_vulkan_analytic_riemann_ladder(
        "counterflow_shock",
        values,
        final_time=final_time,
        cfl=cfl,
        module_path=module_path,
    )
    colliding = _run_vulkan_analytic_riemann_ladder(
        "colliding_streams",
        values,
        final_time=final_time,
        cfl=cfl,
        module_path=module_path,
    )
    reference = json.loads(
        Path(reference_path).read_text(encoding="utf-8")
    )
    order_matches = {}
    for name, result in (
        ("counterflow_shock", counterflow),
        ("colliding_streams", colliding),
    ):
        reference_orders = reference[name]["observed_orders"]
        result_orders = result["observed_orders"]
        order_matches[name] = (
            bool(result_orders)
            and abs(result_orders[-1] - reference_orders[-1]) < 0.35
        )
    qualification = {
        **{
            f"counterflow_{name}": bool(value)
            for name, value in counterflow["qualification"].items()
        },
        **{
            f"colliding_{name}": bool(value)
            for name, value in colliding["qualification"].items()
        },
        "counterflow_reference_order_matches": order_matches[
            "counterflow_shock"
        ],
        "colliding_reference_order_matches": order_matches[
            "colliding_streams"
        ],
    }
    return {
        "schema": "vulkan-tesseract.full-carter-riemann-frontier.v1",
        "backend": "Slang/Vulkan",
        "precision": "float32 device / float64 physical audit",
        "principal_system": {
            "physical_fields": 9,
            "extended_reconstruction_fields": 11,
            "path": "straight-state DLM midpoint fluctuation",
            "riemann_flux": "conditioned full-eigensystem Roe-Rusanov blend",
            "time_integrator": "SSP-RK3 with constitutive projection",
        },
        "reference_schema": reference["schema"],
        "counterflow_shock": counterflow,
        "colliding_streams": colliding,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def _smooth_full_carter_primitives(points: int) -> tuple[Array, Array]:
    coordinate = (np.arange(points, dtype=float) + 0.5) / points
    phase = 2.0 * np.pi * coordinate
    density = 0.1 + 0.004 * np.sin(phase)
    internal = 0.12 + 0.003 * np.cos(phase)
    velocity = 0.015 * np.sin(phase)
    carrier = 0.005 + 0.0002 * np.cos(phase)
    carrier_velocity = 0.025 * np.cos(phase)
    master = _source_solver("analytic_m1").master
    primitive = np.zeros((points, 9), dtype=np.float64)
    primitive[:, 0] = np.log(density)
    primitive[:, 1] = master.specific_entropy(density, internal)
    primitive[:, 2] = velocity / np.sqrt(1.0 - velocity**2)
    primitive[:, 5] = np.log(carrier)
    primitive[:, 6] = carrier_velocity / np.sqrt(
        1.0 - carrier_velocity**2
    )
    return primitive, internal


def qualify_vulkan_full_carter_weno(
    resolutions: Iterable[int] = (16, 32, 64),
    *,
    final_time: float = 0.005,
    cfl: float = 0.1,
    module_path: str | Path | None = None,
) -> dict[str, object]:
    """Qualify full eleven-field characteristic WENO reconstruction."""
    from .carter_riemann import _principal_rhs, _solver

    values = tuple(int(value) for value in resolutions)
    if len(values) < 3:
        raise ValueError("full Carter WENO needs three resolutions")
    snapshots: list[Array] = []
    diagnostics: list[dict[str, object]] = []
    rhs_audits: list[dict[str, float]] = []
    for points in values:
        primitive, internal = _smooth_full_carter_primitives(points)
        q_n = primitive[:, 2:5]
        q_d = primitive[:, 6:9]
        w_n = np.sqrt(1.0 + np.sum(q_n * q_n, axis=1))
        w_d = np.sqrt(1.0 + np.sum(q_d * q_d, axis=1))
        cpu = _solver(
            points,
            reconstruction="full_carter_weno5_z",
            artifact_path="theory33_frozen_variants.json",
            model="analytic_m1",
        )
        cpu_state = cpu.initialize(
            cpu.ccz4.flat_state(),
            FluidPrimitive(
                np.exp(primitive[:, 0]),
                internal,
                (q_n / w_n[:, None]).T,
                np.zeros(points),
            ),
            carrier=CarrierPrimitive(
                np.exp(primitive[:, 5]),
                (q_d / w_d[:, None]).T,
            ),
        )
        cpu_rhs_fields = _principal_rhs(cpu, cpu_state)
        cpu_rhs = np.stack(
            (
                cpu_rhs_fields[0],
                cpu_rhs_fields[5],
                cpu_rhs_fields[1][0],
                cpu_rhs_fields[1][1],
                cpu_rhs_fields[1][2],
                cpu_rhs_fields[2],
                cpu_rhs_fields[6][0],
                cpu_rhs_fields[6][1],
                cpu_rhs_fields[6][2],
                cpu_rhs_fields[3],
                cpu_rhs_fields[4],
            ),
            axis=1,
        )
        geometry = np.zeros((points, 13), dtype=np.float32)
        geometry[:, :9] = np.eye(3, dtype=np.float32).reshape(1, 9)
        geometry[:, 9] = 1.0
        backend = VulkanTheory33MultidimensionalBackend(
            (points,),
            constitutive_model="analytic_m1",
            reconstruction="full_carter_weno5_z",
            module_path=module_path,
        )
        state = backend.initialize(
            primitive.astype(np.float32),
            np.zeros((points, 20), dtype=np.float32),
            geometry,
            np.zeros(points, dtype=np.float32),
        )
        backend._rhs_dispatch(state.tensor, state.primitive)
        gpu_rhs = backend._rhs.to_numpy()
        rhs_audits.append(
            {
                "points": points,
                "maximum_absolute_error": float(
                    np.max(np.abs(gpu_rhs - cpu_rhs))
                ),
                "rms_error": float(
                    np.sqrt(np.mean((gpu_rhs - cpu_rhs) ** 2))
                ),
            }
        )
        reads_before = state.host_reads
        while state.time < final_time - 1.0e-15:
            backend.step(
                state,
                min(cfl / points, final_time - state.time),
                apply_drag=False,
            )
        reads_after = state.host_reads
        snapshot = backend.snapshot(state)
        snapshots.append(snapshot.primitive.astype(float))
        diagnostics.append(
            {
                "points": points,
                "maximum_recovery_residual": float(
                    np.max(snapshot.diagnostics[:, 0])
                ),
                "recovery_failures": int(
                    np.count_nonzero(snapshot.diagnostics[:, 1] < 0.5)
                ),
                "invalid_eigensystems": int(
                    np.count_nonzero(
                        backend._full_face_audit.to_numpy()[:, 4] < 0.5
                    )
                ),
                "host_reads_during_evolution": int(
                    reads_after - reads_before
                ),
            }
        )
    errors = [
        float(
            np.mean(
                np.abs(
                    coarse
                    - fine.reshape(len(coarse), 2, 9).mean(axis=1)
                )
            )
        )
        for coarse, fine in zip(snapshots, snapshots[1:])
    ]
    orders = [
        math.log(errors[index] / errors[index + 1], 2.0)
        for index in range(len(errors) - 1)
    ]
    qualification = {
        "full_weno_rhs_matches_float64_oracle": all(
            item["maximum_absolute_error"] < 1.0e-5
            for item in rhs_audits
        ),
        "smooth_evolution_converges": bool(orders) and orders[-1] > 1.8,
        "all_recoveries_valid": all(
            item["recovery_failures"] == 0 for item in diagnostics
        ),
        "all_eigensystems_valid": all(
            item["invalid_eigensystems"] == 0 for item in diagnostics
        ),
        "device_resident_between_snapshots": all(
            item["host_reads_during_evolution"] == 0
            for item in diagnostics
        ),
    }
    return {
        "schema": "vulkan-tesseract.full-carter-weno5-z.v1",
        "backend": "Slang/Vulkan",
        "reconstruction": "full_carter_weno5_z",
        "physical_fields": 9,
        "extended_fields": ["entropy", "tracer"],
        "resolutions": list(values),
        "final_time": final_time,
        "cfl": cfl,
        "rhs_audits": rhs_audits,
        "successive_l1_errors": errors,
        "observed_orders": orders,
        "diagnostics": diagnostics,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }

def run_vulkan_constrained_convergence(
    resolutions: Iterable[int] = (8, 16, 32),
    *,
    final_time: float = 1.0e-6,
    picard_iterations: int = 3,
    elliptic_tolerance: float = 2.0e-8,
) -> dict[str, object]:
    """Run resident CMC/Picard construction and the coupled NR ladder."""
    from .constrained_convergence import (
        _characteristic_diagnostics,
        build_constraint_converged_state,
    )
    from .convergence import (
        _observed_order,
        _tuple_difference_norm,
    )
    from .vulkan_frontier import _diagnostic_state_from_snapshot

    values = tuple(int(value) for value in resolutions)
    if (
        len(values) != 3
        or any(value < 4 for value in values)
        or values[1] != 2 * values[0]
        or values[2] != 2 * values[1]
    ):
        raise ValueError(
            "constrained resolutions need a factor-two nested triplet"
        )
    if final_time <= 0.0 or elliptic_tolerance <= 0.0:
        raise ValueError("time and elliptic tolerance must be positive")

    def pack_geometry(state: CCZ4State, points: int) -> Array:
        packed = np.zeros((points, 31), dtype=np.float64)
        packed[:, :9] = state.conformal_metric.reshape(9, points).T
        packed[:, 9:18] = state.conformal_A.reshape(9, points).T
        packed[:, 18] = state.conformal_factor
        packed[:, 19] = state.trace_K
        packed[:, 20] = state.theta
        packed[:, 21:24] = state.gamma_hat.T
        packed[:, 24] = state.lapse
        packed[:, 25:28] = state.shift.T
        packed[:, 28:31] = state.shift_driver.T
        return packed

    def pack_matter(state: Theory3ProductionState) -> Array:
        return np.column_stack(
            (
                state.matter.D,
                state.target_charge,
                state.matter.momentum[0],
                state.matter.momentum[1],
                state.matter.momentum[2],
                state.matter.energy,
                state.target_current[0],
                state.target_current[1],
                state.target_current[2],
                state.matter.entropy,
                state.matter.tracer,
            )
        )

    def pack_vector(state: Theory3ProductionState) -> Array:
        points = state.matter.D.size
        packed = np.zeros((points, 16), dtype=np.float64)
        packed[:, 0:3] = state.a.T
        packed[:, 3:6] = state.pi_A.T
        packed[:, 6:9] = state.b.T
        packed[:, 9:12] = state.pi_B.T
        packed[:, 12] = state.longitudinal_A
        packed[:, 13] = state.longitudinal_B
        packed[:, 14] = state.cleaning_A
        packed[:, 15] = state.cleaning_B
        return packed

    def state_from_snapshot(
        solver: Theory33ProductionSolver,
        snapshot: object,
    ) -> Theory3ProductionState:
        matter_values = np.asarray(
            snapshot.coupled.matter.state, dtype=float
        )
        primitive_values = np.asarray(
            snapshot.coupled.matter.primitive, dtype=float
        )
        geometry_values = np.asarray(
            snapshot.coupled.ccz4_state, dtype=float
        )
        vector_values = np.asarray(snapshot.vector_state, dtype=float)
        points = matter_values.shape[0]
        solver._primitive_guess = np.stack(
            [
                primitive_values[:, component].copy()
                for component in (0, 2, 3, 4, 5, 6, 7, 8)
            ]
        )
        geometry = CCZ4State(
            geometry_values[:, :9].T.reshape(3, 3, points),
            geometry_values[:, 9:18].T.reshape(3, 3, points),
            geometry_values[:, 18].copy(),
            geometry_values[:, 19].copy(),
            geometry_values[:, 20].copy(),
            geometry_values[:, 21:24].T.copy(),
            geometry_values[:, 24].copy(),
            geometry_values[:, 25:28].T.copy(),
            geometry_values[:, 28:31].T.copy(),
            float(snapshot.coupled.time),
        )
        matter = Theory3MatterState(
            matter_values[:, 0].copy(),
            matter_values[:, 2:5].T.copy(),
            matter_values[:, 5].copy(),
            matter_values[:, 9].copy(),
            matter_values[:, 10].copy(),
        )
        return Theory3ProductionState(
            geometry,
            matter,
            vector_values[:, 0:3].T.copy(),
            vector_values[:, 3:6].T.copy(),
            vector_values[:, 6:9].T.copy(),
            vector_values[:, 9:12].T.copy(),
            vector_values[:, 12].copy(),
            vector_values[:, 13].copy(),
            vector_values[:, 14].copy(),
            vector_values[:, 15].copy(),
            (
                solver.master_parameters.carrier_charge
                * matter_values[:, 1]
            ),
            matter_values[:, 6:9].T.copy(),
            float(snapshot.coupled.time),
        )

    def packed_fields(snapshot: object) -> tuple[Array, ...]:
        matrices = (
            snapshot.coupled.matter.state,
            snapshot.coupled.ccz4_state,
            snapshot.vector_state,
        )
        return tuple(
            np.asarray(matrix[:, component], dtype=float)
            for matrix in matrices
            for component in range(matrix.shape[1])
        )

    def conserved_integrals(snapshot: object) -> dict[str, float]:
        state = np.asarray(snapshot.coupled.matter.state, dtype=float)
        dx = 1.0 / state.shape[0]
        return {
            "baryon_mass": float(np.sum(state[:, 0]) * dx),
            "carrier_number": float(np.sum(state[:, 1]) * dx),
            "combined_energy": float(np.sum(state[:, 5]) * dx),
            "entropy_integral": float(np.sum(state[:, 9]) * dx),
        }

    def relative_change(
        before: dict[str, float],
        after: dict[str, float],
        name: str,
    ) -> float:
        return abs(after[name] - before[name]) / max(
            abs(before[name]), 1.0e-300
        )

    def run_model(model: str) -> dict[str, object]:
        levels: list[dict[str, object]] = []
        for points in values:
            closure = (
                QualifiedFrozenClosure()
                if model == "qualified_frozen"
                else None
            )
            solver = Theory33ProductionSolver(
                PeriodicGrid((points,), (1.0,), method="fd2"),
                constitutive_closure=closure,
            )
            oracle_state, oracle_elliptic = (
                build_constraint_converged_state(
                    solver,
                    picard_iterations=picard_iterations,
                    elliptic_tolerance=elliptic_tolerance,
                )
            )
            gpu = VulkanTesseractEvolvedBackend(
                (points,),
                constitutive_model=model,
                reconstruction="piecewise_constant",
                relaxation_time=solver.master_parameters.relaxation_time,
                kappa_gravity=solver.ccz4.parameters.kappa_gravity,
                kappa1=solver.ccz4.parameters.kappa1,
                kappa2=solver.ccz4.parameters.kappa2,
                kappa3=solver.ccz4.parameters.kappa3,
                cosmological_constant=(
                    solver.ccz4.parameters.cosmological_constant
                ),
                shift_speed=solver.ccz4.parameters.shift_speed,
                eta_shift=solver.ccz4.parameters.eta_shift,
                slicing=solver.ccz4.parameters.slicing,
                evolve_shift=solver.ccz4.parameters.evolve_shift,
                vector_cleaning_damping=(
                    solver.production_parameters.cleaning_damping
                ),
            )
            gpu_state, gpu_elliptic = gpu.initialize_constrained(
                picard_iterations=picard_iterations,
                elliptic_tolerance=elliptic_tolerance,
            )
            initial_gpu = gpu.snapshot(gpu_state)
            (
                initial_resident,
                initial_diagnostic_state,
                initial_recomposition_error,
            ) = _diagnostic_state_from_snapshot(solver, initial_gpu)
            initial_diagnostics = solver.diagnostics(
                initial_diagnostic_state
            )
            initial_diagnostics.update(
                _characteristic_diagnostics(
                    solver, initial_diagnostic_state
                )
            )
            initial_integrals = conserved_integrals(initial_gpu)
            initial_fields = packed_fields(initial_gpu)
            initial_state_error = float(
                max(
                    np.max(
                        np.abs(
                            initial_gpu.coupled.matter.state
                            - pack_matter(oracle_state)
                        )
                    ),
                    np.max(
                        np.abs(
                            initial_gpu.coupled.ccz4_state
                            - pack_geometry(
                                oracle_state.geometry, points
                            )
                        )
                    ),
                    np.max(
                        np.abs(
                            initial_gpu.vector_state
                            - pack_vector(oracle_state)
                        )
                    ),
                )
            )
            gpu._rhs_dispatch(
                gpu_state.matter.tensor, gpu_state.matter.primitive
            )
            resident_rhs = gpu._rhs.to_numpy().astype(float)
            maximum_dt = 1.0e-7 * 8.0 / points
            steps = max(1, int(math.ceil(final_time / maximum_dt)))
            dt = final_time / steps
            reads_before = gpu_state.host_reads
            for _ in range(steps):
                oracle_state = solver.step(oracle_state, dt)
                gpu.step(gpu_state, dt, apply_drag=True)
                gpu.device.wait_for_idle()
            reads_after = gpu_state.host_reads
            final_gpu = gpu.snapshot(gpu_state)
            (
                final_resident,
                final_diagnostic_state,
                final_recomposition_error,
            ) = _diagnostic_state_from_snapshot(solver, final_gpu)
            final_diagnostics = solver.diagnostics(
                final_diagnostic_state
            )
            final_diagnostics.update(
                _characteristic_diagnostics(
                    solver, final_diagnostic_state
                )
            )
            final_integrals = conserved_integrals(final_gpu)
            matter_error = float(
                np.max(
                    np.abs(
                        final_gpu.coupled.matter.state
                        - pack_matter(oracle_state)
                    )
                )
            )
            vector_error = float(
                np.max(
                    np.abs(
                        final_gpu.vector_state
                        - pack_vector(oracle_state)
                    )
                )
            )
            geometry_error = float(
                np.max(
                    np.abs(
                        final_gpu.coupled.ccz4_state
                        - pack_geometry(oracle_state.geometry, points)
                    )
                )
            )
            levels.append(
                {
                    "points": points,
                    "steps": steps,
                    "dt": dt,
                    "resident_elliptic_report": (
                        gpu_elliptic.__dict__.copy()
                    ),
                    "oracle_elliptic_report": oracle_elliptic,
                    "initial_state_absolute_error": initial_state_error,
                    "initial_conservative_recomposition_l2_error": (
                        initial_recomposition_error
                    ),
                    "final_conservative_recomposition_l2_error": (
                        final_recomposition_error
                    ),
                    "matter_absolute_error": matter_error,
                    "vector_absolute_error": vector_error,
                    "geometry_absolute_error": geometry_error,
                    "initial_diagnostics": initial_diagnostics,
                    "final_diagnostics": final_diagnostics,
                    "relative_balance_errors": {
                        "baryon_mass": relative_change(
                            initial_integrals,
                            final_integrals,
                            "baryon_mass",
                        ),
                        "carrier_number": relative_change(
                            initial_integrals,
                            final_integrals,
                            "carrier_number",
                        ),
                        "combined_energy": relative_change(
                            initial_integrals,
                            final_integrals,
                            "combined_energy",
                        ),
                    },
                    "entropy_change": (
                        final_integrals["entropy_integral"]
                        - initial_integrals["entropy_integral"]
                    ),
                    "energy_change_rate": (
                        float(final_diagnostics["combined_energy"])
                        - float(initial_diagnostics["combined_energy"])
                    )
                    / final_time,
                    "maximum_recovery_residual": float(
                        np.max(
                            final_gpu.coupled.matter.diagnostics[:, 0]
                        )
                    ),
                    "recovery_failures": int(
                        np.count_nonzero(
                            final_gpu.coupled.matter.diagnostics[:, 1]
                            < 0.5
                        )
                    ),
                    "phase_two_fraction": float(
                        np.mean(
                            final_gpu.coupled.matter.diagnostics[:, 3]
                            >= 0.5
                        )
                    ),
                    "minimum_switching_margin": float(
                        np.min(
                            np.abs(
                                final_gpu.coupled.matter.diagnostics[:, 4]
                            )
                        )
                    ),
                    "host_reads_during_evolution": int(
                        reads_after - reads_before
                    ),
                    "_initial_fields": initial_fields,
                    "_final_fields": packed_fields(final_gpu),
                    "_resident_rhs": resident_rhs,
                }
            )

        initial_h = [
            float(level["initial_diagnostics"]["hamiltonian_l2"])
            for level in levels
        ]
        initial_m = [
            float(level["initial_diagnostics"]["momentum_l2"])
            for level in levels
        ]
        final_h = [
            float(level["final_diagnostics"]["hamiltonian_l2"])
            for level in levels
        ]
        final_m = [
            float(level["final_diagnostics"]["momentum_l2"])
            for level in levels
        ]
        evolved_cm = _tuple_difference_norm(
            levels[0]["_final_fields"],
            levels[1]["_final_fields"],
            2,
        )
        evolved_mf = _tuple_difference_norm(
            levels[1]["_final_fields"],
            levels[2]["_final_fields"],
            2,
        )
        energy_cm = abs(
            levels[0]["energy_change_rate"]
            - levels[1]["energy_change_rate"]
        )
        energy_mf = abs(
            levels[1]["energy_change_rate"]
            - levels[2]["energy_change_rate"]
        )
        convergence = {
            "initial_constraints": {
                "hamiltonian": {
                    "values": initial_h,
                    "medium_fine_order": _observed_order(
                        initial_h[1], initial_h[2]
                    ),
                },
                "momentum": {
                    "values": initial_m,
                    "medium_fine_order": _observed_order(
                        initial_m[1], initial_m[2]
                    ),
                },
            },
            "final_constraints": {
                "hamiltonian": {
                    "values": final_h,
                    "medium_fine_order": _observed_order(
                        final_h[1], final_h[2]
                    ),
                },
                "momentum": {
                    "values": final_m,
                    "medium_fine_order": _observed_order(
                        final_m[1], final_m[2]
                    ),
                },
            },
            "evolved_state": {
                "coarse_medium_error": evolved_cm,
                "medium_fine_error": evolved_mf,
                "observed_order": _observed_order(
                    evolved_cm, evolved_mf
                ),
            },
            "geometric_work_energy_rate": {
                "coarse_medium_difference": energy_cm,
                "medium_fine_difference": energy_mf,
                "observed_order": _observed_order(
                    energy_cm, energy_mf
                ),
            },
        }
        qualification = {
            "resident_elliptic_constraints_converged": all(
                bool(
                    level["resident_elliptic_report"]["converged"]
                )
                for level in levels
            ),
            "pointwise_momentum_compatible": all(
                float(
                    level["resident_elliptic_report"][
                        "pointwise_momentum_balance"
                    ]
                )
                < 1.0e-10
                for level in levels
            ),
            "resident_initial_data_matches_oracle": all(
                level["initial_state_absolute_error"] < 3.0e-6
                for level in levels
            ),
            "resident_conserved_state_has_diagnostic_manifold_match": all(
                level[
                    "initial_conservative_recomposition_l2_error"
                ]
                < 5.0e-6
                and level[
                    "final_conservative_recomposition_l2_error"
                ]
                < 5.0e-6
                for level in levels
            ),
            "coupled_evolution_matches_oracle": all(
                level["matter_absolute_error"] < 8.0e-5
                and level["vector_absolute_error"] < 8.0e-5
                and level["geometry_absolute_error"] < 8.0e-5
                for level in levels
            ),
            "all_recoveries_valid": all(
                level["recovery_failures"] == 0 for level in levels
            ),
            "currents_and_entropy_controlled": all(
                level["relative_balance_errors"]["baryon_mass"]
                < 2.0e-6
                and level["relative_balance_errors"]["carrier_number"]
                < 2.0e-6
                and level["entropy_change"] >= -2.0e-7
                for level in levels
            ),
            "device_resident_during_evolution": all(
                level["host_reads_during_evolution"] == 0
                for level in levels
            ),
            "convexity_hyperbolicity_and_causality": all(
                float(
                    level["final_diagnostics"][
                        "minimum_legendre_eigenvalue"
                    ]
                )
                > 0.0
                and float(
                    level["final_diagnostics"][
                        "minimum_thermodynamic_eigenvalue"
                    ]
                )
                > 0.0
                and bool(
                    level["final_diagnostics"][
                        "characteristic_all_strongly_hyperbolic"
                    ]
                )
                and bool(
                    level["final_diagnostics"][
                        "characteristic_all_causal"
                    ]
                )
                for level in levels
            ),
            "initial_ccz4_constraints_second_order": (
                convergence["initial_constraints"]["hamiltonian"][
                    "medium_fine_order"
                ]
                > 1.8
                and convergence["initial_constraints"]["momentum"][
                    "medium_fine_order"
                ]
                > 1.8
            ),
            "evolved_ccz4_constraints_second_order": (
                convergence["final_constraints"]["hamiltonian"][
                    "medium_fine_order"
                ]
                > 1.8
                and convergence["final_constraints"]["momentum"][
                    "medium_fine_order"
                ]
                > 1.8
            ),
            "micro_horizon_energy_balance_is_controlled": all(
                level["relative_balance_errors"]["combined_energy"]
                < 2.0e-6
                for level in levels
            ),
            "positive_evolved_state_convergence": (
                np.isfinite(
                    convergence["evolved_state"]["observed_order"]
                )
                and convergence["evolved_state"]["observed_order"] > 0.0
            ),
        }
        if model == "qualified_frozen":
            qualification["frozen_phase_and_switching_active"] = all(
                level["phase_two_fraction"] > 0.99
                and level["minimum_switching_margin"] > 1.0e-3
                for level in levels
            )
        serial_levels = []
        for level in levels:
            serial_levels.append(
                {
                    key: value
                    for key, value in level.items()
                    if not key.startswith("_")
                }
            )
        return {
            "model": model,
            "levels": serial_levels,
            "convergence": convergence,
            "geometric_work_precision_note": (
                "The 1e-6 float32 micro-horizon energy increment is below "
                "a reliable self-convergence signal. Absolute balance is "
                "gated here; resolved geometric-work convergence is gated "
                "by vulkan_frontier_results.json."
            ),
            "qualification": qualification,
            "qualified": all(qualification.values()),
            "_levels": levels,
        }

    analytic = run_model("analytic_m1")
    frozen = run_model("qualified_frozen")
    finest_analytic = analytic["_levels"][-1]
    finest_frozen = frozen["_levels"][-1]
    rhs_difference = float(
        np.sqrt(
            np.mean(
                (
                    finest_analytic["_resident_rhs"]
                    - finest_frozen["_resident_rhs"]
                )
                ** 2
            )
        )
    )
    evolved_difference = _tuple_difference_norm(
        finest_analytic["_final_fields"],
        finest_frozen["_final_fields"],
        1,
    )
    qualification = {
        **{
            f"analytic_{name}": bool(value)
            for name, value in analytic["qualification"].items()
        },
        **{
            f"frozen_{name}": bool(value)
            for name, value in frozen["qualification"].items()
        },
        "neural_spacetime_response_is_active": rhs_difference > 1.0e-14,
    }
    analytic.pop("_levels")
    frozen.pop("_levels")
    return {
        "schema": "vulkan-tesseract.constrained-convergence.v2",
        "backend": "Slang/Vulkan",
        "scope": (
            "resident active-counterflow construction, fp64 periodic "
            "CMC/Picard solve, float32 Carter/Proca/CCZ4 evolution, and "
            "independent CPU diagnostics"
        ),
        "resolutions": list(values),
        "final_time": final_time,
        "picard_iterations": picard_iterations,
        "elliptic_tolerance": elliptic_tolerance,
        "analytic_control": analytic,
        "qualified_frozen": frozen,
        "finest_model_difference": {
            "rhs_l2": rhs_difference,
            "evolved_state_l2": evolved_difference,
        },
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def qualify_vulkan_amr(
    *,
    seed: int = 7,
) -> dict[str, object]:
    """Qualify resident conservative AMR transfer, reflux, and Gauss repair."""
    from .adm import flat_metric
    from .amr import conservative_prolong
    from .theory3 import Theory3System
    from .vulkan_amr import VulkanAMRBackend

    rng = np.random.default_rng(seed)
    backend = VulkanAMRBackend()
    transfer: list[dict[str, object]] = []
    for ndim, shape in (
        (1, (8,)),
        (2, (4, 5)),
        (3, (4, 5, 6)),
    ):
        coarse = rng.normal(size=(2, *shape)).astype(np.float32)
        gpu_fine = backend.prolong(coarse, ndim)
        gpu_coarse = backend.restrict(gpu_fine, ndim)
        cpu_fine = conservative_prolong(coarse, ndim)
        transfer.append(
            {
                "dimensions": ndim,
                "coarse_shape": list(shape),
                "fine_shape": list(gpu_fine.shape[-ndim:]),
                "prolong_cpu_maximum_absolute_error": float(
                    np.max(np.abs(gpu_fine - cpu_fine))
                ),
                "parent_conservation_maximum_absolute_error": float(
                    np.max(np.abs(gpu_coarse - coarse))
                ),
            }
        )

    grid = PeriodicGrid((16,), (1.0,), method="fd2")
    system = Theory3System(grid)
    (coordinate,) = grid.coordinates()
    pi_a = grid.zeros((3,))
    pi_b = grid.zeros((3,))
    pi_a[0] = 0.1 * np.sin(2.0 * np.pi * coordinate)
    pi_b[0] = -0.05 * np.cos(4.0 * np.pi * coordinate)
    long_a, long_b = backend.project_gauss(
        pi_a, pi_b, shape=grid.shape, lengths=grid.lengths
    )
    gauss = system.gauss_constraints(
        flat_metric(grid), pi_a, pi_b, long_a, long_b
    )
    cpu_long = system.initial_longitudinal_charges(
        flat_metric(grid), pi_a, pi_b
    )
    gauss_report = {
        "longitudinal_cpu_maximum_absolute_error": float(
            max(
                np.max(np.abs(long_a - cpu_long[0])),
                np.max(np.abs(long_b - cpu_long[1])),
            )
        ),
        "gauss_A_linf": float(np.max(np.abs(gauss[0]))),
        "gauss_B_linf": float(np.max(np.abs(gauss[1]))),
    }

    components = 10
    coarse = np.zeros((8, components), dtype=np.float32)
    coarse_lower = np.arange(components, dtype=np.float32) - 2.0
    fine_lower = coarse_lower + np.arange(
        1, components + 1, dtype=np.float32
    )
    coarse_upper = -0.5 * coarse_lower
    fine_upper = coarse_upper + np.arange(
        1, components + 1, dtype=np.float32
    )
    refluxed = backend.reflux_1d(
        coarse,
        lower=2,
        upper=6,
        coarse_lower_flux=coarse_lower,
        fine_lower_flux=fine_lower,
        coarse_upper_flux=coarse_upper,
        fine_upper_flux=fine_upper,
        dt=0.1,
        dx=0.5,
    )
    expected_lower = -0.2 * (fine_lower - coarse_lower)
    expected_upper = 0.2 * (fine_upper - coarse_upper)
    reflux_report = {
        "global_component_balance_linf": float(
            np.max(np.abs(np.sum(refluxed, axis=0)))
        ),
        "lower_face_maximum_absolute_error": float(
            np.max(np.abs(refluxed[1] - expected_lower))
        ),
        "upper_face_maximum_absolute_error": float(
            np.max(np.abs(refluxed[6] - expected_upper))
        ),
    }
    qualification = {
        "one_to_three_dimensional_prolongation_matches_cpu": all(
            item["prolong_cpu_maximum_absolute_error"] < 1.0e-6
            for item in transfer
        ),
        "restriction_restores_parent_average": all(
            item["parent_conservation_maximum_absolute_error"] < 1.0e-6
            for item in transfer
        ),
        "theory3_gauss_projection_matches_cpu": (
            gauss_report["longitudinal_cpu_maximum_absolute_error"]
            < 1.0e-6
            and gauss_report["gauss_A_linf"] < 1.0e-6
            and gauss_report["gauss_B_linf"] < 1.0e-6
        ),
        "berger_colella_reflux_is_componentwise_conservative": (
            reflux_report["global_component_balance_linf"] < 1.0e-6
            and reflux_report["lower_face_maximum_absolute_error"]
            < 1.0e-6
            and reflux_report["upper_face_maximum_absolute_error"]
            < 1.0e-6
        ),
    }
    return {
        "schema": "vulkan-tesseract.amr-primitives.v1",
        "backend": "Slang/Vulkan",
        "refinement_ratio": 2,
        "transfer": transfer,
        "gauss_projection": gauss_report,
        "reflux": reflux_report,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def qualify_vulkan_boundaries(
    *,
    points: int = 48,
    seed: int = 33,
) -> dict[str, object]:
    """Qualify resident nonperiodic characteristic boundary execution."""
    from .boundary import (
        RadiativeBoundary,
        zero_incoming_linear_flux,
    )
    from .grid import CartesianGrid
    from .vulkan_boundary import VulkanBoundaryBackend

    if points < 8:
        raise ValueError("boundary qualification needs at least eight cells")
    rng = np.random.default_rng(seed)
    grid = CartesianGrid(
        (points,), (1.0,), origin=(-0.5,), method="fd2"
    )
    backend = VulkanBoundaryBackend(
        (points,), lengths=(1.0,), origins=(-0.5,)
    )

    # Exercise non-axis-aligned real eigensystems, not only a diagonal toy.
    angles = rng.uniform(-np.pi, np.pi, size=points)
    eigenvalues = np.column_stack(
        (
            -0.3 - 0.7 * rng.random(points),
            0.2 + 0.8 * rng.random(points),
        )
    )
    matrices = np.empty((points, 2, 2), dtype=np.float32)
    for point, angle in enumerate(angles):
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.array(
            [[cosine, -sine], [sine, cosine]], dtype=float
        )
        matrices[point] = (
            rotation
            @ np.diag(eigenvalues[point])
            @ rotation.T
        )
    linear_state = rng.normal(size=(points, 2)).astype(np.float32)
    linear_errors: dict[str, float] = {}
    for sign in (-1, 1):
        gpu_flux = backend.linear_flux(matrices, linear_state, sign)
        cpu_flux = zero_incoming_linear_flux(
            np.moveaxis(matrices, (1, 2), (0, 1)),
            linear_state.T,
            sign,
        ).T
        linear_errors[str(sign)] = float(
            np.max(np.abs(gpu_flux - cpu_flux))
        )

    # Flat CCZ4 with a nontrivial Theta/GammaHat profile exercises both the
    # Sommerfeld and incoming-Z4 characteristic projections.
    (x,) = grid.coordinates()
    ccz4_state = np.zeros((points, 31), dtype=np.float32)
    ccz4_state[:, 0] = 1.0
    ccz4_state[:, 4] = 1.0
    ccz4_state[:, 8] = 1.0
    ccz4_state[:, 18] = 1.0
    ccz4_state[:, 20] = 0.02 + 0.01 * np.cos(2.0 * np.pi * x)
    ccz4_state[:, 21] = 0.015 * np.sin(2.0 * np.pi * x)
    ccz4_state[:, 24] = 1.0
    ccz4_raw = (
        1.0e-3 * rng.normal(size=(points, 31))
    ).astype(np.float32)
    gpu_ccz4 = backend.ccz4_rhs(ccz4_state, ccz4_raw)
    cpu_ccz4_state = CCZ4State(
        np.moveaxis(
            ccz4_state[:, :9].reshape(points, 3, 3),
            (1, 2, 0),
            (0, 1, 2),
        ),
        np.moveaxis(
            ccz4_state[:, 9:18].reshape(points, 3, 3),
            (1, 2, 0),
            (0, 1, 2),
        ),
        ccz4_state[:, 18],
        ccz4_state[:, 19],
        ccz4_state[:, 20],
        ccz4_state[:, 21:24].T,
        ccz4_state[:, 24],
        ccz4_state[:, 25:28].T,
        ccz4_state[:, 28:31].T,
    )
    raw_metric = np.moveaxis(
        ccz4_raw[:, :9].reshape(points, 3, 3),
        (1, 2, 0),
        (0, 1, 2),
    )
    raw_A = np.moveaxis(
        ccz4_raw[:, 9:18].reshape(points, 3, 3),
        (1, 2, 0),
        (0, 1, 2),
    )
    cpu_tuple = RadiativeBoundary(grid).ccz4_rhs(
        cpu_ccz4_state,
        (
            raw_metric,
            raw_A,
            ccz4_raw[:, 18],
            ccz4_raw[:, 19],
            ccz4_raw[:, 20],
            ccz4_raw[:, 21:24].T,
            ccz4_raw[:, 24],
            ccz4_raw[:, 25:28].T,
            ccz4_raw[:, 28:31].T,
        ),
    )
    cpu_ccz4 = np.zeros_like(ccz4_raw, dtype=float)
    cpu_ccz4[:, :9] = np.moveaxis(
        cpu_tuple[0], (0, 1, 2), (1, 2, 0)
    ).reshape(points, 9)
    cpu_ccz4[:, 9:18] = np.moveaxis(
        cpu_tuple[1], (0, 1, 2), (1, 2, 0)
    ).reshape(points, 9)
    cpu_ccz4[:, 18] = cpu_tuple[2]
    cpu_ccz4[:, 19] = cpu_tuple[3]
    cpu_ccz4[:, 20] = cpu_tuple[4]
    cpu_ccz4[:, 21:24] = cpu_tuple[5].T
    cpu_ccz4[:, 24] = cpu_tuple[6]
    cpu_ccz4[:, 25:28] = cpu_tuple[7].T
    cpu_ccz4[:, 28:31] = cpu_tuple[8].T
    boundary_mask = np.zeros(points, dtype=bool)
    boundary_mask[[0, -1]] = True
    ccz4_boundary_error = float(
        np.max(np.abs(gpu_ccz4[boundary_mask] - cpu_ccz4[boundary_mask]))
    )
    ccz4_interior_error = float(
        np.max(
            np.abs(
                gpu_ccz4[~boundary_mask] - ccz4_raw[~boundary_mask]
            )
        )
    )
    # Repeat the same independent-oracle comparison with the original
    # fourth-order Cartesian boundary stencils.  This closes the otherwise
    # easy-to-miss gap between periodic FD4 CCZ4 and isolated-domain FD4
    # evolution.
    fd4_grid = CartesianGrid(
        (points,), (1.0,), origin=(-0.5,), method="fd4"
    )
    fd4_backend = VulkanBoundaryBackend(
        (points,),
        lengths=(1.0,),
        origins=(-0.5,),
        derivative_order=4,
    )
    gpu_ccz4_fd4 = fd4_backend.ccz4_rhs(ccz4_state, ccz4_raw)
    cpu_tuple_fd4 = RadiativeBoundary(fd4_grid).ccz4_rhs(
        cpu_ccz4_state,
        (
            raw_metric,
            raw_A,
            ccz4_raw[:, 18],
            ccz4_raw[:, 19],
            ccz4_raw[:, 20],
            ccz4_raw[:, 21:24].T,
            ccz4_raw[:, 24],
            ccz4_raw[:, 25:28].T,
            ccz4_raw[:, 28:31].T,
        ),
    )
    cpu_ccz4_fd4 = np.zeros_like(ccz4_raw, dtype=float)
    cpu_ccz4_fd4[:, :9] = np.moveaxis(
        cpu_tuple_fd4[0], (0, 1, 2), (1, 2, 0)
    ).reshape(points, 9)
    cpu_ccz4_fd4[:, 9:18] = np.moveaxis(
        cpu_tuple_fd4[1], (0, 1, 2), (1, 2, 0)
    ).reshape(points, 9)
    cpu_ccz4_fd4[:, 18] = cpu_tuple_fd4[2]
    cpu_ccz4_fd4[:, 19] = cpu_tuple_fd4[3]
    cpu_ccz4_fd4[:, 20] = cpu_tuple_fd4[4]
    cpu_ccz4_fd4[:, 21:24] = cpu_tuple_fd4[5].T
    cpu_ccz4_fd4[:, 24] = cpu_tuple_fd4[6]
    cpu_ccz4_fd4[:, 25:28] = cpu_tuple_fd4[7].T
    cpu_ccz4_fd4[:, 28:31] = cpu_tuple_fd4[8].T
    ccz4_fd4_boundary_error = float(
        np.max(
            np.abs(
                gpu_ccz4_fd4[boundary_mask]
                - cpu_ccz4_fd4[boundary_mask]
            )
        )
    )
    ccz4_fd4_interior_error = float(
        np.max(
            np.abs(
                gpu_ccz4_fd4[~boundary_mask]
                - ccz4_raw[~boundary_mask]
            )
        )
    )

    # The mixed-vector correction must leave the semidiscrete time
    # derivative of both flat-space Gauss constraints identically zero.
    vector_state = np.zeros((points, 16), dtype=np.float32)
    vector_state[:, 1] = 0.02 * np.exp(-((x - 0.25) / 0.1) ** 2)
    vector_state[:, 8] = -0.01 * np.exp(-((x - 0.2) / 0.12) ** 2)
    vector_state[:, 3] = 0.03 * np.sin(2.0 * np.pi * x)
    vector_state[:, 9] = -0.02 * np.cos(2.0 * np.pi * x)
    vector_raw = (
        2.0e-3 * rng.normal(size=(points, 16))
    ).astype(np.float32)
    vector_raw[:, 12] = grid.derivative(vector_raw[:, 3], 0)
    vector_raw[:, 13] = grid.derivative(vector_raw[:, 9], 0)
    gpu_vector = backend.mixed_vector_rhs(vector_state, vector_raw)
    dconstraint_A = grid.derivative(
        -gpu_vector[:, 3], 0
    ) + gpu_vector[:, 12]
    dconstraint_B = grid.derivative(
        -gpu_vector[:, 9], 0
    ) + gpu_vector[:, 13]
    vector_raw_fd4 = vector_raw.copy()
    vector_raw_fd4[:, 12] = fd4_grid.derivative(
        vector_raw_fd4[:, 3], 0
    )
    vector_raw_fd4[:, 13] = fd4_grid.derivative(
        vector_raw_fd4[:, 9], 0
    )
    gpu_vector_fd4 = fd4_backend.mixed_vector_rhs(
        vector_state, vector_raw_fd4
    )
    dconstraint_A_fd4 = fd4_grid.derivative(
        -gpu_vector_fd4[:, 3], 0
    ) + gpu_vector_fd4[:, 12]
    dconstraint_B_fd4 = fd4_grid.derivative(
        -gpu_vector_fd4[:, 9], 0
    ) + gpu_vector_fd4[:, 13]

    def audit_multidimensional_boundary(
        shape: tuple[int, ...],
        derivative_order: int,
    ) -> dict[str, object]:
        dimensions = len(shape)
        method = f"fd{derivative_order}"
        lengths = (1.0,) * dimensions
        origins = (-0.5,) * dimensions
        local_grid = CartesianGrid(
            shape, lengths, origin=origins, method=method
        )
        local_backend = VulkanBoundaryBackend(
            shape,
            lengths=lengths,
            origins=origins,
            derivative_order=derivative_order,
        )
        local_points = int(np.prod(shape))
        coordinates = local_grid.coordinates()

        def flatten(values: Array) -> Array:
            return np.asarray(values).reshape(-1, order="F")

        def field(values: Array, column: int) -> Array:
            return np.asarray(values[:, column], dtype=float).reshape(
                shape, order="F"
            )

        phase = np.zeros(shape, dtype=float)
        for axis, coordinate in enumerate(coordinates):
            phase += (axis + 1.0) * coordinate
        local_state = np.zeros((local_points, 31), dtype=np.float32)
        metric_wave = 0.006 * np.sin(2.0 * np.pi * phase)
        local_state[:, 0] = flatten(np.exp(metric_wave))
        local_state[:, 4] = flatten(np.exp(-metric_wave))
        local_state[:, 8] = 1.0
        local_state[:, 18] = 1.0
        local_state[:, 20] = flatten(
            0.01 + 0.003 * np.cos(2.0 * np.pi * phase)
        )
        for axis, coordinate in enumerate(coordinates):
            local_state[:, 21 + axis] = flatten(
                0.004
                * (axis + 1.0)
                * np.sin(2.0 * np.pi * coordinate)
            )
        local_state[:, 24] = flatten(
            1.0 + 0.002 * np.sin(2.0 * np.pi * phase)
        )
        local_raw = (
            1.0e-3 * rng.normal(size=(local_points, 31))
        ).astype(np.float32)
        gpu_local = local_backend.ccz4_rhs(
            local_state, local_raw
        )

        metric = local_grid.zeros((3, 3))
        conformal_a = local_grid.zeros((3, 3))
        gamma_hat = local_grid.zeros((3,))
        shift = local_grid.zeros((3,))
        shift_driver = local_grid.zeros((3,))
        for row in range(3):
            for column in range(3):
                metric[row, column] = field(
                    local_state, row * 3 + column
                )
                conformal_a[row, column] = field(
                    local_state, 9 + row * 3 + column
                )
        for axis in range(3):
            gamma_hat[axis] = field(local_state, 21 + axis)
            shift[axis] = field(local_state, 25 + axis)
            shift_driver[axis] = field(local_state, 28 + axis)
        local_cpu_state = CCZ4State(
            metric,
            conformal_a,
            field(local_state, 18),
            field(local_state, 19),
            field(local_state, 20),
            gamma_hat,
            field(local_state, 24),
            shift,
            shift_driver,
        )
        raw_metric_local = local_grid.zeros((3, 3))
        raw_a_local = local_grid.zeros((3, 3))
        raw_gamma_local = local_grid.zeros((3,))
        raw_shift_local = local_grid.zeros((3,))
        raw_driver_local = local_grid.zeros((3,))
        for row in range(3):
            for column in range(3):
                raw_metric_local[row, column] = field(
                    local_raw, row * 3 + column
                )
                raw_a_local[row, column] = field(
                    local_raw, 9 + row * 3 + column
                )
        for axis in range(3):
            raw_gamma_local[axis] = field(local_raw, 21 + axis)
            raw_shift_local[axis] = field(local_raw, 25 + axis)
            raw_driver_local[axis] = field(local_raw, 28 + axis)
        local_cpu_tuple = RadiativeBoundary(local_grid).ccz4_rhs(
            local_cpu_state,
            (
                raw_metric_local,
                raw_a_local,
                field(local_raw, 18),
                field(local_raw, 19),
                field(local_raw, 20),
                raw_gamma_local,
                field(local_raw, 24),
                raw_shift_local,
                raw_driver_local,
            ),
        )
        local_cpu = np.zeros_like(local_raw, dtype=float)
        for row in range(3):
            for column in range(3):
                local_cpu[:, row * 3 + column] = flatten(
                    local_cpu_tuple[0][row, column]
                )
                local_cpu[:, 9 + row * 3 + column] = flatten(
                    local_cpu_tuple[1][row, column]
                )
        local_cpu[:, 18] = flatten(local_cpu_tuple[2])
        local_cpu[:, 19] = flatten(local_cpu_tuple[3])
        local_cpu[:, 20] = flatten(local_cpu_tuple[4])
        local_cpu[:, 24] = flatten(local_cpu_tuple[6])
        for axis in range(3):
            local_cpu[:, 21 + axis] = flatten(
                local_cpu_tuple[5][axis]
            )
            local_cpu[:, 25 + axis] = flatten(
                local_cpu_tuple[7][axis]
            )
            local_cpu[:, 28 + axis] = flatten(
                local_cpu_tuple[8][axis]
            )
        local_mask = flatten(
            RadiativeBoundary(local_grid).mask
        ).astype(bool)
        local_ccz4_boundary_error = float(
            np.max(
                np.abs(
                    gpu_local[local_mask] - local_cpu[local_mask]
                )
            )
        )
        local_ccz4_interior_error = float(
            np.max(
                np.abs(
                    gpu_local[~local_mask]
                    - local_raw[~local_mask]
                )
            )
        )

        local_vector_state = np.zeros(
            (local_points, 16), dtype=np.float32
        )
        local_vector_raw = (
            1.0e-3 * rng.normal(size=(local_points, 16))
        ).astype(np.float32)
        divergence_a = np.zeros(shape, dtype=float)
        divergence_b = np.zeros(shape, dtype=float)
        for axis in range(dimensions):
            divergence_a += local_grid.derivative(
                field(local_vector_raw, 3 + axis), axis
            )
            divergence_b += local_grid.derivative(
                field(local_vector_raw, 9 + axis), axis
            )
        local_vector_raw[:, 12] = flatten(divergence_a)
        local_vector_raw[:, 13] = flatten(divergence_b)
        local_vector_gpu = local_backend.mixed_vector_rhs(
            local_vector_state, local_vector_raw
        )
        local_vector_cpu = np.array(
            local_vector_raw, dtype=float, copy=True
        )
        local_radiative = RadiativeBoundary(local_grid)
        for component in range(16):
            if component in {12, 13}:
                continue
            local_vector_cpu[:, component] = flatten(
                local_radiative.radiative_rhs(
                    field(local_vector_state, component),
                    field(local_vector_raw, component),
                    0.0,
                )
            )
        longitudinal_a = field(local_vector_raw, 12)
        longitudinal_b = field(local_vector_raw, 13)
        for axis in range(dimensions):
            longitudinal_a += local_grid.derivative(
                field(local_vector_cpu, 3 + axis)
                - field(local_vector_raw, 3 + axis),
                axis,
            )
            longitudinal_b += local_grid.derivative(
                field(local_vector_cpu, 9 + axis)
                - field(local_vector_raw, 9 + axis),
                axis,
            )
        local_vector_cpu[:, 12] = flatten(longitudinal_a)
        local_vector_cpu[:, 13] = flatten(longitudinal_b)
        local_vector_error = float(
            np.max(np.abs(local_vector_gpu - local_vector_cpu))
        )
        gauss_a = field(local_vector_gpu, 12)
        gauss_b = field(local_vector_gpu, 13)
        for axis in range(dimensions):
            gauss_a += local_grid.derivative(
                -field(local_vector_gpu, 3 + axis), axis
            )
            gauss_b += local_grid.derivative(
                -field(local_vector_gpu, 9 + axis), axis
            )
        return {
            "dimensions": dimensions,
            "shape": list(shape),
            "derivative_order": derivative_order,
            "ccz4_boundary_maximum_absolute_error": (
                local_ccz4_boundary_error
            ),
            "ccz4_interior_maximum_absolute_error": (
                local_ccz4_interior_error
            ),
            "gauss_A_rhs_linf": float(np.max(np.abs(gauss_a))),
            "gauss_B_rhs_linf": float(np.max(np.abs(gauss_b))),
            "proca_maximum_absolute_error": local_vector_error,
            "qualified": (
                local_ccz4_boundary_error < 1.0e-5
                and local_ccz4_interior_error == 0.0
                and local_vector_error < 5.0e-6
                and float(np.max(np.abs(gauss_a))) < 5.0e-6
                and float(np.max(np.abs(gauss_b))) < 5.0e-6
            ),
        }

    multidimensional_boundary_audits = [
        audit_multidimensional_boundary(
            (8,) * dimensions, derivative_order
        )
        for dimensions in (2, 3)
        for derivative_order in (2, 4)
    ]

    # Finally run the physical boundary through the resident Carter RHS.
    primitive = np.zeros((points, 9), dtype=np.float32)
    primitive[:, 0] = np.log(0.1)
    primitive[:, 1] = 0.15
    primitive[:, 5] = np.log(
        0.002
        + 0.008 * np.exp(-((x - 0.42) / 0.045) ** 2)
    )
    primitive[:, 6] = 0.16
    vector_input = np.zeros((points, 20), dtype=np.float32)
    geometry = np.zeros((points, 13), dtype=np.float32)
    geometry[:, 0] = 1.0
    geometry[:, 4] = 1.0
    geometry[:, 8] = 1.0
    geometry[:, 9] = 1.0
    matter_backend = VulkanTheory33MultidimensionalBackend(
        (points,),
        lengths=(1.0,),
        origins=(-0.5,),
        boundary_mode="radiative_outflow",
        reconstruction="piecewise_constant",
        relaxation_time=1.0,
    )
    matter_state = matter_backend.initialize(
        primitive,
        vector_input,
        geometry,
        np.zeros(points, dtype=np.float32),
    )
    initial = matter_backend.snapshot(matter_state)
    reads_before = matter_state.host_reads
    dt = 2.0e-5
    for _ in range(3):
        matter_backend.step(matter_state, dt, apply_drag=False)
    reads_after = matter_state.host_reads
    final = matter_backend.snapshot(matter_state)
    initial_carrier = float(np.sum(initial.state[:, 1]) / points)
    final_carrier = float(np.sum(final.state[:, 1]) / points)
    lower_change = float(
        abs(final.state[0, 1] - initial.state[0, 1])
    )

    # Dispatch the complete Carter + evolved-Proca + CCZ4 Strang step through
    # the fourth-order physical-boundary route, with no stage readback.
    evolved_fd4 = VulkanTesseractEvolvedBackend(
        (points,),
        lengths=(1.0,),
        origins=(-0.5,),
        boundary_mode="radiative_outflow",
        reconstruction="piecewise_constant",
        relaxation_time=1.0,
        derivative_order=4,
    )
    ccz4_initial_fd4 = fd4_backend.complete_ccz4_gamma_hat(ccz4_state)
    evolved_state_fd4 = evolved_fd4.initialize(
        primitive,
        vector_state,
        ccz4_initial_fd4,
        np.zeros(points, dtype=np.float32),
    )
    evolved_reads_before = evolved_state_fd4.host_reads
    evolved_fd4.step(evolved_state_fd4, 1.0e-6, apply_drag=False)
    evolved_reads_after = evolved_state_fd4.host_reads
    evolved_snapshot_fd4 = evolved_fd4.snapshot(evolved_state_fd4)
    evolved_fd4_invalid = int(
        np.count_nonzero(
            evolved_snapshot_fd4.coupled.matter.diagnostics[:, 1] < 0.5
        )
    )
    evolved_fd4_finite = bool(
        np.all(np.isfinite(evolved_snapshot_fd4.coupled.ccz4_state))
        and np.all(
            np.isfinite(evolved_snapshot_fd4.coupled.matter.state)
        )
        and np.all(
            np.isfinite(evolved_snapshot_fd4.vector_state)
        )
    )

    qualification = {
        "linear_characteristic_flux_matches_cpu": (
            max(linear_errors.values()) < 2.0e-6
        ),
        "ccz4_radiative_constraint_projection_matches_cpu": (
            ccz4_boundary_error < 3.0e-6
            and ccz4_interior_error == 0.0
        ),
        "mixed_proca_boundary_is_gauss_compatible": (
            bool(np.max(np.abs(dconstraint_A)) < 3.0e-6)
            and bool(np.max(np.abs(dconstraint_B)) < 3.0e-6)
        ),
        "fourth_order_ccz4_boundary_matches_cpu": (
            ccz4_fd4_boundary_error < 3.0e-6
            and ccz4_fd4_interior_error == 0.0
        ),
        "fourth_order_proca_boundary_is_gauss_compatible": (
            bool(np.max(np.abs(dconstraint_A_fd4)) < 3.0e-6)
            and bool(np.max(np.abs(dconstraint_B_fd4)) < 3.0e-6)
        ),
        "outgoing_carter_pulse_loses_carrier": (
            final_carrier < initial_carrier
        ),
        "lower_boundary_has_no_detectable_inflow": (
            lower_change < 5.0e-8
        ),
        "physical_boundary_evolution_is_resident": (
            reads_after - reads_before == 0
        ),
        "physical_boundary_recovery_remains_valid": (
            bool(np.count_nonzero(final.diagnostics[:, 1] < 0.5) == 0)
        ),
        "fourth_order_full_stack_boundary_step_is_resident": (
            evolved_reads_after - evolved_reads_before == 0
            and evolved_fd4_invalid == 0
            and evolved_fd4_finite
        ),
        "two_and_three_dimensional_fd2_fd4_boundaries_match": (
            all(
                bool(item["qualified"])
                for item in multidimensional_boundary_audits
            )
        ),
    }
    return {
        "schema": "vulkan-tesseract.physical-boundaries.v1",
        "backend": "Slang/Vulkan",
        "points": points,
        "linear_flux_maximum_absolute_error": linear_errors,
        "ccz4_boundary_maximum_absolute_error": ccz4_boundary_error,
        "ccz4_interior_maximum_absolute_error": ccz4_interior_error,
        "ccz4_fd4_boundary_maximum_absolute_error": (
            ccz4_fd4_boundary_error
        ),
        "ccz4_fd4_interior_maximum_absolute_error": (
            ccz4_fd4_interior_error
        ),
        "gauss_A_rhs_linf": float(np.max(np.abs(dconstraint_A))),
        "gauss_B_rhs_linf": float(np.max(np.abs(dconstraint_B))),
        "gauss_A_fd4_rhs_linf": float(
            np.max(np.abs(dconstraint_A_fd4))
        ),
        "gauss_B_fd4_rhs_linf": float(
            np.max(np.abs(dconstraint_B_fd4))
        ),
        "initial_integrated_carrier": initial_carrier,
        "final_integrated_carrier": final_carrier,
        "lower_boundary_carrier_change": lower_change,
        "host_reads_during_evolution": int(reads_after - reads_before),
        "invalid_recoveries": int(
            np.count_nonzero(final.diagnostics[:, 1] < 0.5)
        ),
        "fourth_order_full_stack_host_reads": int(
            evolved_reads_after - evolved_reads_before
        ),
        "fourth_order_full_stack_invalid_recoveries": (
            evolved_fd4_invalid
        ),
        "fourth_order_full_stack_finite": evolved_fd4_finite,
        "multidimensional_boundary_audits": (
            multidimensional_boundary_audits
        ),
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def qualify_vulkan_tov_strong_field(
    *,
    central_density: float = 1.28e-3,
    radial_spacing: float = 0.01,
    points: int = 64,
    steps: int = 8,
) -> dict[str, object]:
    """Qualify fp64 resident TOV construction and a curved active step."""
    from .grid import CartesianGrid
    from .tov import solve_tov
    from .vulkan_boundary import VulkanBoundaryBackend
    from .vulkan_tov import VulkanTOVBackend

    if points < 16 or steps < 1:
        raise ValueError("strong-field qualification is undersampled")
    cpu = solve_tov(
        central_density,
        dr=radial_spacing,
        maximum_radius=20.0,
    )
    tov_backend = VulkanTOVBackend()
    gpu = tov_backend.solve(
        central_density,
        dr=radial_spacing,
        maximum_radius=20.0,
    )
    profile_errors = {
        name: float(
            np.max(np.abs(getattr(gpu, name) - getattr(cpu, name)))
        )
        for name in (
            "radius",
            "density",
            "pressure",
            "energy_density",
            "enclosed_mass",
            "lapse",
            "radial_metric",
        )
    }
    scalar_errors = {
        "surface_radius": abs(gpu.surface_radius - cpu.surface_radius),
        "gravitational_mass": abs(
            gpu.gravitational_mass - cpu.gravitational_mass
        ),
        "baryonic_mass": abs(gpu.baryonic_mass - cpu.baryonic_mass),
    }
    radius = gpu.radius
    tov_rhs = -(
        (gpu.energy_density + gpu.pressure)
        * (
            gpu.enclosed_mass
            + 4.0 * np.pi * radius**3 * gpu.pressure
        )
        / (radius * (radius - 2.0 * gpu.enclosed_mass))
    )
    numerical = np.gradient(gpu.pressure, radius)
    hydro_mask = (np.arange(radius.size) > 5) & (
        gpu.pressure > 1.0e-6 * gpu.pressure[0]
    )
    hydrostatic_relative_l2 = float(
        np.sqrt(np.mean((numerical[hydro_mask] - tov_rhs[hydro_mask]) ** 2))
        / np.sqrt(np.mean(tov_rhs[hydro_mask] ** 2))
    )

    lengths = (24.0,)
    origins = (-12.0,)
    atmosphere = 1.0e-6
    primitive, ccz4, vector = tov_backend.sample_cartesian(
        (points,),
        lengths=lengths,
        origins=origins,
        density_floor=atmosphere,
    )
    boundary_backend = VulkanBoundaryBackend(
        (points,), lengths=lengths, origins=origins
    )
    ccz4 = boundary_backend.complete_ccz4_gamma_hat(ccz4)
    grid = CartesianGrid(
        (points,), lengths, origin=origins, method="fd2"
    )
    (coordinate,) = grid.coordinates()
    sample_radius = np.abs(coordinate)
    expected_radial = np.empty(points)
    expected_lapse = np.empty(points)
    inside = sample_radius <= gpu.surface_radius
    expected_radial[inside] = np.interp(
        sample_radius[inside], gpu.radius, gpu.radial_metric
    )
    expected_lapse[inside] = np.interp(
        sample_radius[inside], gpu.radius, gpu.lapse
    )
    compact = np.maximum(
        1.0 - 2.0 * gpu.gravitational_mass / sample_radius[~inside],
        1.0e-8,
    )
    expected_radial[~inside] = 1.0 / compact
    expected_lapse[~inside] = np.sqrt(compact)
    phi = ccz4[:, 18]
    physical_xx = ccz4[:, 0] / phi**2
    sampled_metric_error = float(
        np.max(np.abs(physical_xx - expected_radial))
    )
    sampled_lapse_error = float(
        np.max(np.abs(ccz4[:, 24] - expected_lapse))
    )
    conformal = ccz4[:, :9].reshape(points, 3, 3)
    conformal_inverse = np.linalg.inv(conformal)
    contracted = np.zeros((points, 3))
    for upper in range(3):
        contracted[:, upper] = -grid.derivative(
            conformal_inverse[:, upper, 0], 0
        )
    gamma_hat_error = float(
        np.max(np.abs(ccz4[:, 21:24] - contracted))
    )
    conformal_determinant_error = float(
        np.max(np.abs(np.linalg.det(conformal) - 1.0))
    )

    evolution = VulkanTesseractEvolvedBackend(
        (points,),
        lengths=lengths,
        origins=origins,
        boundary_mode="radiative_outflow",
        reconstruction="piecewise_constant",
        relaxation_time=1.0,
    )
    state = evolution.initialize(
        primitive,
        vector,
        ccz4,
        np.zeros(points, dtype=np.float32),
    )
    initial = evolution.snapshot(state)
    reads_before = state.host_reads
    for _ in range(steps):
        evolution.step(state, 1.0e-6, apply_drag=False)
    reads_after = state.host_reads
    final = evolution.snapshot(state)
    maximum_matter_change = float(
        np.max(
            np.abs(
                final.coupled.matter.state
                - initial.coupled.matter.state
            )
        )
    )
    maximum_geometry_change = float(
        np.max(
            np.abs(
                final.coupled.ccz4_state
                - initial.coupled.ccz4_state
            )
        )
    )
    invalid_recoveries = int(
        np.count_nonzero(
            final.coupled.matter.diagnostics[:, 1] < 0.5
        )
    )
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint_path = Path(temporary) / "strong_field_restart.npz"
        evolution.save_checkpoint(state, checkpoint_path)
        restarted_backend = VulkanTesseractEvolvedBackend(
            (points,),
            lengths=lengths,
            origins=origins,
            boundary_mode="radiative_outflow",
            reconstruction="piecewise_constant",
            relaxation_time=1.0,
        )
        restarted = restarted_backend.load_checkpoint(checkpoint_path)
        restored_snapshot = restarted_backend.snapshot(restarted)
        checkpoint_state_error = float(
            np.max(
                np.abs(
                    restored_snapshot.coupled.matter.state
                    - final.coupled.matter.state
                )
            )
        )
        checkpoint_geometry_error = float(
            np.max(
                np.abs(
                    restored_snapshot.coupled.ccz4_state
                    - final.coupled.ccz4_state
                )
            )
        )
        checkpoint_vector_error = float(
            np.max(
                np.abs(
                    restored_snapshot.vector_state - final.vector_state
                )
            )
        )
        evolution.step(state, 1.0e-6, apply_drag=False)
        restarted_backend.step(
            restarted, 1.0e-6, apply_drag=False
        )
        continued = evolution.snapshot(state)
        restarted_continued = restarted_backend.snapshot(restarted)
        checkpoint_continuation_error = float(
            max(
                np.max(
                    np.abs(
                        continued.coupled.matter.state
                        - restarted_continued.coupled.matter.state
                    )
                ),
                np.max(
                    np.abs(
                        continued.coupled.ccz4_state
                        - restarted_continued.coupled.ccz4_state
                    )
                ),
                np.max(
                    np.abs(
                        continued.vector_state
                        - restarted_continued.vector_state
                    )
                ),
            )
        )
    qualification = {
        "fp64_tov_profile_matches_original": (
            max(profile_errors.values()) < 3.0e-7
            and max(scalar_errors.values()) < 3.0e-7
        ),
        "standard_gamma_two_star_is_reproduced": (
            abs(gpu.gravitational_mass - 1.400) < 0.01
            and abs(gpu.surface_radius - 9.575) < 0.03
            and gpu.baryonic_mass > gpu.gravitational_mass
        ),
        "hydrostatic_residual_is_converged": (
            hydrostatic_relative_l2 < 1.0e-4
        ),
        "cartesian_strong_field_sampling_matches_profile": (
            sampled_metric_error < 2.0e-6
            and sampled_lapse_error < 2.0e-6
            and conformal_determinant_error < 2.0e-6
        ),
        "sampled_z4_constraint_is_algebraically_zero": (
            gamma_hat_error < 2.0e-6
        ),
        "strong_field_step_is_finite_and_active": (
            bool(np.all(np.isfinite(final.coupled.ccz4_state)))
            and bool(np.all(np.isfinite(final.coupled.matter.state)))
            and maximum_geometry_change > 0.0
            and maximum_matter_change > 0.0
        ),
        "strong_field_recovery_remains_valid": invalid_recoveries == 0,
        "strong_field_evolution_is_device_resident": (
            reads_after - reads_before == 0
        ),
        "complete_evolved_checkpoint_is_deterministic": (
            checkpoint_state_error == 0.0
            and checkpoint_geometry_error < 1.0e-7
            and checkpoint_vector_error == 0.0
            and checkpoint_continuation_error < 2.0e-6
        ),
    }
    return {
        "schema": "vulkan-tesseract.tov-strong-field.v1",
        "backend": "Slang/Vulkan",
        "central_density": central_density,
        "radial_spacing": radial_spacing,
        "profile_points": int(gpu.radius.size),
        "surface_radius": gpu.surface_radius,
        "gravitational_mass": gpu.gravitational_mass,
        "baryonic_mass": gpu.baryonic_mass,
        "profile_maximum_absolute_errors": profile_errors,
        "scalar_absolute_errors": scalar_errors,
        "hydrostatic_relative_l2": hydrostatic_relative_l2,
        "sampled_metric_maximum_absolute_error": sampled_metric_error,
        "sampled_lapse_maximum_absolute_error": sampled_lapse_error,
        "conformal_determinant_linf": conformal_determinant_error,
        "gamma_hat_constraint_linf": gamma_hat_error,
        "evolution_points": points,
        "evolution_steps": steps,
        "maximum_matter_change": maximum_matter_change,
        "maximum_geometry_change": maximum_geometry_change,
        "maximum_recovery_residual": float(
            np.max(final.coupled.matter.diagnostics[:, 0])
        ),
        "invalid_recoveries": invalid_recoveries,
        "host_reads_during_evolution": int(
            reads_after - reads_before
        ),
        "checkpoint_state_roundtrip_error": checkpoint_state_error,
        "checkpoint_geometry_roundtrip_error": checkpoint_geometry_error,
        "checkpoint_vector_roundtrip_error": checkpoint_vector_error,
        "checkpoint_continuation_error": checkpoint_continuation_error,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def qualify_vulkan_isolated_ctt(
    *,
    shape: tuple[int, int, int] = (10, 10, 10),
) -> dict[str, object]:
    """Qualify the resident fp64 isolated CTT constraint solve."""
    from .adm import ADMParameters, ADMSolver, ADMState
    from .grid import CartesianGrid
    from .initial_constraints import (
        CTTInitialDataSolver,
        CTTParameters,
    )
    from .vulkan_ctt import VulkanCTTBackend

    if len(shape) != 3 or min(shape) < 6:
        raise ValueError("isolated CTT qualification requires a 3D patch")
    lengths = (8.0, 8.0, 8.0)
    grid = CartesianGrid(shape, lengths, method="fd2")
    gpu_backend = VulkanCTTBackend(shape, lengths)
    zero_rho = grid.zeros()
    zero_momentum = grid.zeros((3,))
    vacuum = gpu_backend.solve(zero_rho, zero_momentum)

    x, y, z = grid.coordinates()
    radius_squared = x * x + y * y + z * z
    envelope = np.exp(-radius_squared / 2.0)
    rho = 2.0e-4 * envelope
    momentum = grid.zeros((3,))
    momentum[0] = 1.0e-6 * y * envelope
    momentum[1] = -1.0e-6 * x * envelope
    parameters = CTTParameters(
        maximum_iterations=5000,
        tolerance=8.0e-7,
        report_interval=20,
    )
    cpu = CTTInitialDataSolver(grid, parameters).solve(
        rho, momentum
    )
    gpu = gpu_backend.solve(rho, momentum, parameters)
    psi_error = float(
        np.max(np.abs(gpu.conformal_factor - cpu.conformal_factor))
    )
    vector_error = float(
        np.max(np.abs(gpu.vector_potential - cpu.vector_potential))
    )
    ccz4 = gpu.ccz4_state
    phi = ccz4[:, 18].reshape(shape, order="F")
    metric = grid.zeros((3, 3))
    extrinsic = grid.zeros((3, 3))
    for row in range(3):
        for column in range(3):
            component = 3 * row + column
            gt = ccz4[:, component].reshape(shape, order="F")
            At = ccz4[:, 9 + component].reshape(
                shape, order="F"
            )
            metric[row, column] = gt / phi**2
            extrinsic[row, column] = At / phi**2
    hamiltonian, momentum_constraint = ADMSolver(
        grid, ADMParameters(parameters.kappa)
    ).constraints(
        ADMState(metric, extrinsic),
        (rho, momentum, grid.zeros((3, 3))),
    )
    core = (slice(2, -2),) * 3
    adm_hamiltonian_core_l2 = float(
        np.sqrt(np.mean(hamiltonian[core] ** 2))
    )
    adm_momentum_linf = float(
        np.max(np.abs(momentum_constraint[(slice(None),) + core]))
    )
    qualification = {
        "vacuum_is_exact_flat_fixed_point": (
            vacuum.report.converged
            and vacuum.report.iterations == 1
            and bool(
                np.max(np.abs(vacuum.conformal_factor - 1.0)) == 0.0
            )
            and bool(np.max(np.abs(vacuum.vector_potential)) == 0.0)
        ),
        "localized_matter_ctt_converges": (
            gpu.report.converged
            and gpu.report.hamiltonian_residual_l2
            < parameters.tolerance
            and gpu.report.momentum_residual_l2
            < parameters.tolerance
            and bool(np.max(gpu.conformal_factor) > 1.0)
        ),
        "fp64_ctt_fields_match_original": (
            psi_error < 1.0e-9 and vector_error < 1.0e-9
        ),
        "packed_geometry_reduces_adm_constraints": (
            adm_hamiltonian_core_l2 < 2.0e-4
            and adm_momentum_linf < 2.0e-6
        ),
        "isolated_outer_data_are_exact": (
            bool(
                np.all(gpu.conformal_factor[0, :, :] == 1.0)
                and np.all(gpu.conformal_factor[-1, :, :] == 1.0)
                and np.all(gpu.conformal_factor[:, 0, :] == 1.0)
                and np.all(gpu.conformal_factor[:, -1, :] == 1.0)
                and np.all(gpu.conformal_factor[:, :, 0] == 1.0)
                and np.all(gpu.conformal_factor[:, :, -1] == 1.0)
            )
        ),
    }
    return {
        "schema": "vulkan-tesseract.isolated-ctt.v1",
        "backend": "Slang/Vulkan fp64",
        "shape": list(shape),
        "vacuum": {
            "converged": vacuum.report.converged,
            "iterations": vacuum.report.iterations,
            "hamiltonian_residual_l2": (
                vacuum.report.hamiltonian_residual_l2
            ),
            "momentum_residual_l2": (
                vacuum.report.momentum_residual_l2
            ),
        },
        "localized_matter": {
            "iterations": gpu.report.iterations,
            "hamiltonian_residual_l2": (
                gpu.report.hamiltonian_residual_l2
            ),
            "momentum_residual_l2": (
                gpu.report.momentum_residual_l2
            ),
            "minimum_conformal_factor": (
                gpu.report.minimum_conformal_factor
            ),
            "maximum_conformal_factor": float(
                np.max(gpu.conformal_factor)
            ),
            "scalar_readbacks": gpu.scalar_readbacks,
        },
        "cpu_parity": {
            "conformal_factor_maximum_absolute_error": psi_error,
            "vector_potential_maximum_absolute_error": vector_error,
            "iteration_difference": (
                gpu.report.iterations - cpu.report.iterations
            ),
        },
        "adm_audit": {
            "hamiltonian_core_l2": adm_hamiltonian_core_l2,
            "momentum_core_linf": adm_momentum_linf,
        },
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }
