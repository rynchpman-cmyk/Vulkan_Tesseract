"""Independent CPU oracle for the first GPU-resident Theory 3.3 slice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .frozen_closure import QualifiedFrozenClosure
from .ccz4 import CCZ4Parameters, CCZ4Solver, CCZ4State
from .grid import PeriodicGrid
from .matter import FluidPrimitive
from .production3 import Theory3ProductionParameters
from .production33 import Theory33ProductionSolver
from .theory33 import CarrierPrimitive
from .vulkan_backend import (
    DIAGNOSTIC_FIELDS,
    STATE_FIELDS,
    VulkanCCZ4GeometryBackend,
    VulkanTesseractCoupledBackend,
    VulkanTheory33Backend,
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


def qualify_vulkan_multidimensional_full_stack(
    *,
    shape: tuple[int, int] = (8, 8),
    steps: int = 2,
    seed: int = 3344,
) -> dict[str, object]:
    """Qualify resident curved Carter evolution and the full CCZ4 operator."""
    if min(shape) < 4 or steps < 1:
        raise ValueError("qualification needs at least 4x4 cells and one step")
    nx, ny = shape
    points = nx * ny
    grid = PeriodicGrid(shape, (1.0, 1.0), method="fd2")
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
    }
    qualification = {
        "derivative_ccz4_matches_fd2_cpu": (
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
        "full_stack_outputs_are_finite": metrics["all_outputs_finite"],
    }
    return {
        "schema": "vulkan-tesseract.multidimensional-full-stack.v1",
        "backend": "Slang/Vulkan",
        "shape": list(shape),
        "steps": steps,
        "dt": dt,
        "constitutive_model": "qualified_frozen",
        "spatial_operator": (
            "multidirectional Rusanov plus midpoint Carter DLM path"
        ),
        "time_integrator": (
            "CCZ4 half-step / Carter SSP-RK3 with implicit drag / "
            "CCZ4 half-step"
        ),
        "ccz4_derivatives": "periodic centered FD2",
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
    """Return the qualified two-branch contact used by the CPU frontier."""
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
    module_path: str | Path | None = None,
) -> dict[str, object]:
    """Qualify GPU-resident frozen-neural phase evolution against the CPU."""
    values = tuple(int(value) for value in resolutions)
    if len(values) < 2 or any(value < 8 for value in values):
        raise ValueError("Vulkan ladder needs at least two resolutions")
    closure = QualifiedFrozenClosure()
    diagnostics: list[dict[str, object]] = []
    for points in values:
        primitives = neural_phase_contact_primitives(points)
        cpu = CpuMaterialReference.initialize(
            primitives, constitutive_model="qualified_frozen"
        )
        initial_baryon = float(np.sum(cpu.state[:, 0]))
        initial_carrier = float(np.sum(cpu.state[:, 1]))
        gpu = VulkanTheory33Backend(
            points,
            constitutive_model="qualified_frozen",
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
        "constitutive_model": "qualified_frozen",
        "variant_digest": closure.variant_digest,
        "phase_semantics": (
            "hard branch frozen within Newton/Jacobian; reclassified "
            "between recovery transition iterations"
        ),
        "reconstruction": "piecewise_constant_rusanov",
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
