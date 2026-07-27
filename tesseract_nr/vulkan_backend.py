"""Slang/Vulkan control plane for GPU-resident Theory 3.3 evolution.

The initial production slice advances the flat, periodic, one-dimensional
Carter principal subsystem. Python performs initialization, dispatch, and
optional readback only; primitive recovery, constitutive evaluation, fluxes,
the canonical-momentum path product, SSP-RK3 stages, and entropy projection
all execute on Vulkan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


STATE_WIDTH = 7
RECOVERY_WIDTH = 5
DIAGNOSTIC_WIDTH = 14
PATH_WIDTH = 4
EIGEN_AUDIT_WIDTH = 8
PRIMITIVE_3D_WIDTH = 9
VECTOR_INPUT_WIDTH = 20
VECTOR_STRESS_WIDTH = 15
FULL_CARTER_STATE_WIDTH = 11
DRAG_DIAGNOSTIC_WIDTH = 8
GEOMETRY_3D_WIDTH = 13
PRINCIPAL_PATH_WIDTH = 4
FULL_EIGEN_AUDIT_WIDTH = 6
CCZ4_STATE_WIDTH = 31
CCZ4_PROJECTION_AUDIT_WIDTH = 3
FULL_RECOVERY_DIAGNOSTIC_WIDTH = 8
FULL_DRAG_DIAGNOSTIC_WIDTH = 6

STATE_FIELDS = (
    "baryon_density_conserved",
    "carrier_density_conserved",
    "momentum_x",
    "energy",
    "carrier_canonical_momentum_x",
    "entropy",
    "tracer",
)

DIAGNOSTIC_FIELDS = (
    "baryon_density",
    "fluid_velocity_x",
    "carrier_density",
    "carrier_velocity_x",
    "specific_entropy",
    "recovery_residual",
    "recovery_valid",
    "relative_lorentz_factor",
    "minimum_legendre_eigenvalue",
    "minimum_thermodynamic_eigenvalue",
    "constitutive_phase_two",
    "constitutive_phase_threshold",
    "constitutive_switching_margin",
    "recovery_phase_crossings",
)

CONSTITUTIVE_MODES = {
    "analytic_m1": 0,
    "qualified_frozen": 1,
}

RECONSTRUCTION_MODES = {
    "piecewise_constant": 0,
    "weno5_z": 1,
    "carter_characteristic_weno5_z": 2,
}

EIGEN_AUDIT_FIELDS = (
    "maximum_characteristic_speed",
    "maximum_eigenpair_residual",
    "inverse_basis_residual",
    "condition_estimate",
    "minimum_eigenvalue_separation",
    "eigensystem_valid",
    "phase_boundary_face",
    "causal_margin",
)

PRIMITIVE_3D_FIELDS = (
    "log_baryon_density",
    "specific_entropy",
    "baryon_spatial_four_velocity_x",
    "baryon_spatial_four_velocity_y",
    "baryon_spatial_four_velocity_z",
    "log_carrier_density",
    "carrier_spatial_four_velocity_x",
    "carrier_spatial_four_velocity_y",
    "carrier_spatial_four_velocity_z",
)

VECTOR_STRESS_FIELDS = (
    "energy_density",
    "momentum_x",
    "momentum_y",
    "momentum_z",
    "stress_xx",
    "stress_xy",
    "stress_xz",
    "stress_yx",
    "stress_yy",
    "stress_yz",
    "stress_zx",
    "stress_zy",
    "stress_zz",
    "phi_a",
    "phi_b",
)

FULL_CARTER_STATE_FIELDS = (
    "baryon_density_conserved",
    "carrier_density_conserved",
    "total_momentum_x",
    "total_momentum_y",
    "total_momentum_z",
    "total_energy",
    "carrier_canonical_momentum_x",
    "carrier_canonical_momentum_y",
    "carrier_canonical_momentum_z",
    "entropy_density",
    "tracer_density",
)

DRAG_DIAGNOSTIC_FIELDS = (
    "relative_lorentz_factor_before",
    "relative_lorentz_factor_after",
    "specific_entropy_increase",
    "scaled_conservation_residual",
    "valid",
    "phase_crossings",
    "relative_rapidity_decay",
    "thermalized_energy_estimate",
)


def _slangpy() -> Any:
    try:
        import slangpy as spy
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "The Vulkan backend needs slangpy. Run through .venv-slang or "
            "install the 'shader' optional dependencies."
        ) from exc
    return spy


@dataclass
class VulkanTheory33State:
    """Opaque GPU-resident state handle.

    ``host_reads`` changes only when ``snapshot`` is requested. A complete
    ``step`` performs no tensor readback.
    """

    tensor: Any
    recovery_guess: Any
    diagnostics: Any
    positivity_theta: Any
    eigenvalues: Any
    eigensystem_audit: Any
    points: int
    length: float
    constitutive_model: str
    time: float = 0.0
    steps: int = 0
    host_reads: int = 0


@dataclass(frozen=True)
class VulkanTheory33Snapshot:
    state: np.ndarray
    recovery: np.ndarray
    diagnostics: np.ndarray
    positivity_theta: np.ndarray
    eigenvalues: np.ndarray
    eigensystem_audit: np.ndarray
    time: float
    steps: int
    constitutive_model: str

    def state_field(self, name: str) -> np.ndarray:
        return self.state[:, STATE_FIELDS.index(name)]

    def diagnostic_field(self, name: str) -> np.ndarray:
        return self.diagnostics[:, DIAGNOSTIC_FIELDS.index(name)]

    def eigen_audit_field(self, name: str) -> np.ndarray:
        return self.eigensystem_audit[:, EIGEN_AUDIT_FIELDS.index(name)]


@dataclass(frozen=True)
class VulkanTheory33InterfaceSnapshot:
    low: np.ndarray
    high: np.ndarray


class VulkanTheory33Backend:
    """Dispatch a GPU-resident flat 1D Theory 3.3 SSP-RK3 evolution."""

    def __init__(
        self,
        points: int,
        *,
        length: float = 1.0,
        constitutive_model: str = "analytic_m1",
        reconstruction: str = "piecewise_constant",
        positivity_preserving: bool = False,
        positivity_floor_factor: float = 1.001,
        module_path: str | Path | None = None,
    ) -> None:
        if points < 8:
            raise ValueError("Vulkan Theory 3.3 needs at least eight cells")
        if length <= 0.0:
            raise ValueError("domain length must be positive")
        if constitutive_model not in CONSTITUTIVE_MODES:
            raise ValueError(
                "constitutive_model must be 'analytic_m1' or "
                "'qualified_frozen'"
            )
        if reconstruction not in RECONSTRUCTION_MODES:
            raise ValueError(
                "reconstruction must be 'piecewise_constant', 'weno5_z', "
                "or 'carter_characteristic_weno5_z'"
            )
        if positivity_floor_factor < 1.0:
            raise ValueError("positivity_floor_factor must be at least one")
        self.points = int(points)
        self.length = float(length)
        self.constitutive_model = constitutive_model
        self.model_mode = CONSTITUTIVE_MODES[constitutive_model]
        self.reconstruction = reconstruction
        self.reconstruction_mode = RECONSTRUCTION_MODES[reconstruction]
        self.positivity_preserving = bool(positivity_preserving)
        self.positivity_floor_factor = float(positivity_floor_factor)
        self.dx = self.length / self.points
        self.spy = _slangpy()
        root = Path(__file__).resolve().parents[1]
        self.module_path = (
            root / "tesseract_nr_gpu.slang"
            if module_path is None
            else Path(module_path).resolve()
        )
        self.device = self.spy.create_device(
            type=self.spy.DeviceType.vulkan,
            include_paths=[self.module_path.parent],
        )
        self.module = self.spy.Module.load_from_file(
            self.device, str(self.module_path)
        )
        self.grid = self.spy.grid(shape=(self.points,))
        self._state_buffers = [
            self.spy.Tensor.empty(
                self.device,
                shape=(self.points, STATE_WIDTH),
                dtype=self.module.float,
            )
            for _ in range(4)
        ]
        self._guess_buffers = [
            self.spy.Tensor.empty(
                self.device,
                shape=(self.points, RECOVERY_WIDTH),
                dtype=self.module.float,
            )
            for _ in range(3)
        ]
        self._diagnostic_buffers = [
            self.spy.Tensor.empty(
                self.device,
                shape=(self.points, DIAGNOSTIC_WIDTH),
                dtype=self.module.float,
            )
            for _ in range(2)
        ]
        self._flux = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, STATE_WIDTH),
            dtype=self.module.float,
        )
        self._path = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, PATH_WIDTH),
            dtype=self.module.float,
        )
        self._low_interface = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, STATE_WIDTH),
            dtype=self.module.float,
        )
        self._high_interface = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, STATE_WIDTH),
            dtype=self.module.float,
        )
        self._low_rhs = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, STATE_WIDTH),
            dtype=self.module.float,
        )
        self._high_rhs = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, STATE_WIDTH),
            dtype=self.module.float,
        )
        self._rhs_buffer = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, STATE_WIDTH),
            dtype=self.module.float,
        )
        self._stage_theta = self.spy.Tensor.empty(
            self.device,
            shape=(self.points,),
            dtype=self.module.float,
        )
        self._positivity_theta = self.spy.Tensor.empty(
            self.device,
            shape=(self.points,),
            dtype=self.module.float,
        )
        self._right_basis = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, 25),
            dtype=self.module.float,
        )
        self._left_basis = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, 25),
            dtype=self.module.float,
        )
        self._eigenvalues = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, 5),
            dtype=self.module.float,
        )
        self._eigensystem_audit = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, EIGEN_AUDIT_WIDTH),
            dtype=self.module.float,
        )
        self._current_state_index = 0
        self._current_guess_index = 0
        self._initialized = False

    def initialize(self, primitives: np.ndarray) -> VulkanTheory33State:
        """Upload primitives once and construct conserved state on the GPU.

        Input columns are ``(n, specific_internal_energy, u_x, d, v_x,
        tracer)``.
        """
        values = np.asarray(primitives, dtype=np.float32)
        if values.shape != (self.points, 6):
            raise ValueError(
                f"primitives must have shape {(self.points, 6)}, "
                f"received {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("initial primitives must be finite")
        primitive_tensor = self.spy.Tensor.from_numpy(self.device, values)
        self._current_state_index = 0
        self._current_guess_index = 0
        state = self._state_buffers[self._current_state_index]
        guess = self._guess_buffers[self._current_guess_index]
        diagnostics = self._diagnostic_buffers[0]
        self.module.initializeTheory33Kernel(
            point=self.grid,
            modelMode=self.model_mode,
            primitiveInput=primitive_tensor,
            state=state,
            recoveryGuess=guess,
            diagnostics=diagnostics,
        )
        self.module.theory33UnitThetaKernel(
            point=self.grid,
            theta=self._positivity_theta,
        )
        self.module.theory33UnitThetaKernel(
            point=self.grid,
            theta=self._stage_theta,
        )
        self.module.theory33EigensystemKernel(
            point=self.grid,
            modelMode=self.model_mode,
            recoveryGuess=guess,
            rightBasis=self._right_basis,
            leftBasis=self._left_basis,
            eigenvalues=self._eigenvalues,
            audit=self._eigensystem_audit,
        )
        self._initialized = True
        return VulkanTheory33State(
            state,
            guess,
            diagnostics,
            self._positivity_theta,
            self._eigenvalues,
            self._eigensystem_audit,
            self.points,
            self.length,
            self.constitutive_model,
        )

    def _recover_and_rhs(
        self,
        state: Any,
        guess_input: Any,
        guess_output: Any,
        diagnostics: Any,
        dt: float,
    ) -> None:
        self.module.recoverTheory33Kernel(
            point=self.grid,
            modelMode=self.model_mode,
            state=state,
            guessInput=guess_input,
            guessOutput=guess_output,
            diagnostics=diagnostics,
        )
        self.module.theory33FluxKernel(
            point=self.grid,
            modelMode=self.model_mode,
            state=state,
            recoveryGuess=guess_output,
            flux=self._flux,
            pathData=self._path,
        )
        self.module.theory33InterfaceKernel(
            point=self.grid,
            reconstructionMode=self.reconstruction_mode,
            state=state,
            flux=self._flux,
            lowInterface=self._low_interface,
            highInterface=self._high_interface,
        )
        if self.reconstruction_mode == 2:
            self.module.theory33EigensystemKernel(
                point=self.grid,
                modelMode=self.model_mode,
                recoveryGuess=guess_output,
                rightBasis=self._right_basis,
                leftBasis=self._left_basis,
                eigenvalues=self._eigenvalues,
                audit=self._eigensystem_audit,
            )
            self.module.theory33CharacteristicInterfaceKernel(
                point=self.grid,
                state=state,
                flux=self._flux,
                rightBasis=self._right_basis,
                leftBasis=self._left_basis,
                eigenvalues=self._eigenvalues,
                highInterface=self._high_interface,
            )
        if self.positivity_preserving:
            self.module.theory33InterfaceRhsKernel(
                point=self.grid,
                inverseDx=float(1.0 / self.dx),
                interfaceFlux=self._low_interface,
                pathData=self._path,
                rhs=self._low_rhs,
            )
            self.module.theory33InterfaceRhsKernel(
                point=self.grid,
                inverseDx=float(1.0 / self.dx),
                interfaceFlux=self._high_interface,
                pathData=self._path,
                rhs=self._high_rhs,
            )
            self.module.theory33PositivityThetaKernel(
                point=self.grid,
                dt=float(dt),
                densityFloor=(
                    1.0e-8 * self.positivity_floor_factor
                ),
                carrierFloor=(
                    1.0e-8 * self.positivity_floor_factor
                ),
                energyFloor=(
                    1.0e-12 * self.positivity_floor_factor
                ),
                state=state,
                lowRhs=self._low_rhs,
                highRhs=self._high_rhs,
                theta=self._stage_theta,
            )
            self.module.theory33AccumulateThetaKernel(
                point=self.grid,
                stageTheta=self._stage_theta,
                minimumTheta=self._positivity_theta,
            )
            self.module.theory33LimitedRhsKernel(
                point=self.grid,
                inverseDx=float(1.0 / self.dx),
                lowInterface=self._low_interface,
                highInterface=self._high_interface,
                theta=self._stage_theta,
                pathData=self._path,
                rhs=self._rhs_buffer,
            )
        else:
            self.module.theory33UnitThetaKernel(
                point=self.grid,
                theta=self._stage_theta,
            )
            self.module.theory33InterfaceRhsKernel(
                point=self.grid,
                inverseDx=float(1.0 / self.dx),
                interfaceFlux=self._high_interface,
                pathData=self._path,
                rhs=self._rhs_buffer,
            )

    def _stage(
        self,
        *,
        dt: float,
        base_weight: float,
        evolved_weight: float,
        base_state: Any,
        stage_state: Any,
        output: Any,
    ) -> None:
        self.module.theory33StageKernel(
            point=self.grid,
            dt=float(dt),
            baseWeight=float(base_weight),
            evolvedWeight=float(evolved_weight),
            baseState=base_state,
            stageState=stage_state,
            rhs=self._rhs_buffer,
            output=output,
        )

    def step(
        self, state: VulkanTheory33State, dt: float
    ) -> VulkanTheory33State:
        """Advance one SSP-RK3 step without reading state back to the host."""
        if not self._initialized:
            raise RuntimeError("initialize the Vulkan backend before stepping")
        if state.points != self.points or state.length != self.length:
            raise ValueError("state belongs to a different Vulkan backend")
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be positive and finite")

        base_index = self._current_state_index
        first_index = (base_index + 1) % 4
        second_index = (base_index + 2) % 4
        final_index = (base_index + 3) % 4
        base = self._state_buffers[base_index]
        first = self._state_buffers[first_index]
        second = self._state_buffers[second_index]
        final = self._state_buffers[final_index]

        guess0_index = self._current_guess_index
        guess1_index = (guess0_index + 1) % 3
        guess2_index = (guess0_index + 2) % 3
        guess0 = self._guess_buffers[guess0_index]
        guess1 = self._guess_buffers[guess1_index]
        guess2 = self._guess_buffers[guess2_index]
        scratch_diagnostics = self._diagnostic_buffers[1]
        final_diagnostics = self._diagnostic_buffers[0]
        self.module.theory33UnitThetaKernel(
            point=self.grid,
            theta=self._positivity_theta,
        )

        self._recover_and_rhs(
            base, guess0, guess1, scratch_diagnostics, dt
        )
        self._stage(
            dt=dt,
            base_weight=0.0,
            evolved_weight=1.0,
            base_state=base,
            stage_state=base,
            output=first,
        )

        self._recover_and_rhs(
            first, guess1, guess2, scratch_diagnostics, dt
        )
        self._stage(
            dt=dt,
            base_weight=0.75,
            evolved_weight=0.25,
            base_state=base,
            stage_state=first,
            output=second,
        )

        self._recover_and_rhs(
            second, guess2, guess1, scratch_diagnostics, dt
        )
        self._stage(
            dt=dt,
            base_weight=1.0 / 3.0,
            evolved_weight=2.0 / 3.0,
            base_state=base,
            stage_state=second,
            output=final,
        )

        self.module.recoverTheory33Kernel(
            point=self.grid,
            modelMode=self.model_mode,
            state=final,
            guessInput=guess1,
            guessOutput=guess0,
            diagnostics=final_diagnostics,
        )
        self.module.projectTheory33Kernel(
            point=self.grid,
            recoveryGuess=guess0,
            state=final,
        )

        self._current_state_index = final_index
        state.tensor = final
        state.recovery_guess = guess0
        state.diagnostics = final_diagnostics
        state.positivity_theta = self._positivity_theta
        state.eigenvalues = self._eigenvalues
        state.eigensystem_audit = self._eigensystem_audit
        state.time += float(dt)
        state.steps += 1
        return state

    def evolve(
        self,
        state: VulkanTheory33State,
        *,
        final_time: float,
        cfl: float = 0.2,
    ) -> VulkanTheory33State:
        if final_time < state.time:
            raise ValueError("final_time precedes the current state")
        if not 0.0 < cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1]")
        while state.time < final_time - 1.0e-15:
            dt = min(cfl * self.dx, final_time - state.time)
            self.step(state, dt)
        return state

    def interface_snapshot(
        self, state: VulkanTheory33State
    ) -> VulkanTheory33InterfaceSnapshot:
        """Read reconstructed interfaces for an explicit accuracy audit."""
        scratch_guess = self._guess_buffers[
            (self._current_guess_index + 1) % 3
        ]
        self.module.recoverTheory33Kernel(
            point=self.grid,
            modelMode=self.model_mode,
            state=state.tensor,
            guessInput=state.recovery_guess,
            guessOutput=scratch_guess,
            diagnostics=self._diagnostic_buffers[1],
        )
        self.module.theory33FluxKernel(
            point=self.grid,
            modelMode=self.model_mode,
            state=state.tensor,
            recoveryGuess=scratch_guess,
            flux=self._flux,
            pathData=self._path,
        )
        self.module.theory33InterfaceKernel(
            point=self.grid,
            reconstructionMode=self.reconstruction_mode,
            state=state.tensor,
            flux=self._flux,
            lowInterface=self._low_interface,
            highInterface=self._high_interface,
        )
        state.host_reads += 1
        return VulkanTheory33InterfaceSnapshot(
            self._low_interface.to_numpy(),
            self._high_interface.to_numpy(),
        )

    @staticmethod
    def snapshot(state: VulkanTheory33State) -> VulkanTheory33Snapshot:
        state.host_reads += 1
        return VulkanTheory33Snapshot(
            state.tensor.to_numpy(),
            state.recovery_guess.to_numpy(),
            state.diagnostics.to_numpy(),
            state.positivity_theta.to_numpy(),
            state.eigenvalues.to_numpy(),
            state.eigensystem_audit.to_numpy(),
            state.time,
            state.steps,
            state.constitutive_model,
        )


@dataclass(frozen=True)
class VulkanTheory33SourceSnapshot:
    """Explicit qualification readback for the local source operators."""

    primitive_before: np.ndarray
    primitive_after_drag: np.ndarray
    drag_diagnostics: np.ndarray
    vector_stress: np.ndarray
    conserved_before: np.ndarray
    conserved_after: np.ndarray
    constitutive_model: str

    def drag_field(self, name: str) -> np.ndarray:
        return self.drag_diagnostics[
            :, DRAG_DIAGNOSTIC_FIELDS.index(name)
        ]

    def vector_field(self, name: str) -> np.ndarray:
        return self.vector_stress[:, VECTOR_STRESS_FIELDS.index(name)]

    def conserved_field(self, name: str, *, after: bool = False) -> np.ndarray:
        values = self.conserved_after if after else self.conserved_before
        return values[:, FULL_CARTER_STATE_FIELDS.index(name)]


@dataclass(frozen=True)
class VulkanTheory33PrincipalSnapshot:
    """Readback of the curved, arbitrary-axis nine-field principal map."""

    conserved: np.ndarray
    flux: np.ndarray
    path: np.ndarray
    valid: np.ndarray
    axis: int
    constitutive_model: str


@dataclass(frozen=True)
class VulkanTheory33FullEigensystemSnapshot:
    """Complete numerical eigensystem of the curved nine-field block."""

    symbol: np.ndarray
    right_basis: np.ndarray
    left_basis: np.ndarray
    eigenvalues: np.ndarray
    audit: np.ndarray
    axis: int
    constitutive_model: str


@dataclass(frozen=True)
class VulkanCCZ4GeometrySnapshot:
    projected_state: np.ndarray
    physical_geometry: np.ndarray
    extrinsic_curvature: np.ndarray
    matter_source: np.ndarray
    projection_audit: np.ndarray


@dataclass(frozen=True)
class VulkanCCZ4VacuumSnapshot:
    projected_state: np.ndarray
    vacuum_rhs: np.ndarray
    projection_audit: np.ndarray
    shape: tuple[int, int, int]


@dataclass
class VulkanTheory33MultidimensionalState:
    tensor: Any
    primitive: Any
    diagnostics: Any
    vector_stress: Any
    geometry: Any
    shape: tuple[int, int, int]
    time: float = 0.0
    steps: int = 0
    host_reads: int = 0


@dataclass(frozen=True)
class VulkanTheory33MultidimensionalSnapshot:
    state: np.ndarray
    primitive: np.ndarray
    diagnostics: np.ndarray
    vector_stress: np.ndarray
    geometry: np.ndarray
    shape: tuple[int, int, int]
    time: float
    steps: int


@dataclass
class VulkanTesseractCoupledState:
    matter: VulkanTheory33MultidimensionalState
    ccz4: Any
    time: float = 0.0
    steps: int = 0
    host_reads: int = 0


@dataclass(frozen=True)
class VulkanTesseractCoupledSnapshot:
    matter: VulkanTheory33MultidimensionalSnapshot
    ccz4_state: np.ndarray
    ccz4_rhs: np.ndarray
    ccz4_projection_audit: np.ndarray
    matter_stress: np.ndarray
    time: float
    steps: int


class VulkanTheory33SourceBackend:
    """Vulkan mixed-vector stress and implicit carrier-drag control plane.

    This is the local source half of the full multidimensional ABI. The
    primitive ordering is documented by ``PRIMITIVE_3D_FIELDS``. Vector input
    columns are ``a_i, pi_A^i, b_i, pi_B^i, longitudinal_A,
    longitudinal_B, magnetic_A^i, magnetic_B^i``.
    """

    def __init__(
        self,
        points: int,
        *,
        constitutive_model: str = "analytic_m1",
        module_path: str | Path | None = None,
    ) -> None:
        if points < 1:
            raise ValueError("at least one source point is required")
        if constitutive_model not in CONSTITUTIVE_MODES:
            raise ValueError(
                "constitutive_model must be 'analytic_m1' or "
                "'qualified_frozen'"
            )
        self.points = int(points)
        self.constitutive_model = constitutive_model
        self.model_mode = CONSTITUTIVE_MODES[constitutive_model]
        self.spy = _slangpy()
        root = Path(__file__).resolve().parents[1]
        self.module_path = (
            root / "tesseract_nr_gpu.slang"
            if module_path is None
            else Path(module_path).resolve()
        )
        self.device = self.spy.create_device(
            type=self.spy.DeviceType.vulkan,
            include_paths=[self.module_path.parent],
        )
        self.module = self.spy.Module.load_from_file(
            self.device, str(self.module_path)
        )
        self.grid = self.spy.grid(shape=(self.points,))
        self._primitive_after = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, PRIMITIVE_3D_WIDTH),
            dtype=self.module.float,
        )
        self._drag_diagnostics = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, DRAG_DIAGNOSTIC_WIDTH),
            dtype=self.module.float,
        )
        self._vector_stress = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, VECTOR_STRESS_WIDTH),
            dtype=self.module.float,
        )
        self._single_vector_stress_a = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, 13),
            dtype=self.module.float,
        )
        self._single_vector_stress_b = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, 13),
            dtype=self.module.float,
        )
        self._conserved_before = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, FULL_CARTER_STATE_WIDTH),
            dtype=self.module.float,
        )
        self._conserved_after = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, FULL_CARTER_STATE_WIDTH),
            dtype=self.module.float,
        )
        self._principal_conserved = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, PRIMITIVE_3D_WIDTH),
            dtype=self.module.float,
        )
        self._principal_flux = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, PRIMITIVE_3D_WIDTH),
            dtype=self.module.float,
        )
        self._principal_path = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, PRINCIPAL_PATH_WIDTH),
            dtype=self.module.float,
        )
        self._principal_valid = self.spy.Tensor.empty(
            self.device,
            shape=(self.points,),
            dtype=self.module.float,
        )
        self._full_symbol = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, 81),
            dtype=self.module.float,
        )
        self._full_right_basis = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, 81),
            dtype=self.module.float,
        )
        self._full_left_basis = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, 81),
            dtype=self.module.float,
        )
        self._full_eigenvalues = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, 9),
            dtype=self.module.float,
        )
        self._full_eigen_audit = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, FULL_EIGEN_AUDIT_WIDTH),
            dtype=self.module.float,
        )

    @staticmethod
    def _array(
        values: np.ndarray,
        *,
        shape: tuple[int, ...],
        name: str,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, received {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array

    def evaluate(
        self,
        primitive: np.ndarray,
        vector_input: np.ndarray,
        tracer: np.ndarray,
        *,
        dt: float,
        relaxation_time: float,
        lapse: float = 1.0,
    ) -> VulkanTheory33SourceSnapshot:
        """Apply vector stress, compose the state, and solve implicit drag."""
        if dt < 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and nonnegative")
        if relaxation_time <= 0.0 or not np.isfinite(relaxation_time):
            raise ValueError("relaxation_time must be positive and finite")
        if lapse < 0.0 or not np.isfinite(lapse):
            raise ValueError("lapse must be finite and nonnegative")
        primitive_values = self._array(
            primitive,
            shape=(self.points, PRIMITIVE_3D_WIDTH),
            name="primitive",
        )
        vector_values = self._array(
            vector_input,
            shape=(self.points, VECTOR_INPUT_WIDTH),
            name="vector_input",
        )
        tracer_values = self._array(
            tracer,
            shape=(self.points,),
            name="tracer",
        )
        primitive_tensor = self.spy.Tensor.from_numpy(
            self.device, primitive_values
        )
        vector_tensor = self.spy.Tensor.from_numpy(
            self.device, vector_values
        )
        tracer_tensor = self.spy.Tensor.from_numpy(
            self.device, tracer_values
        )
        self.module.theory33VectorStressKernel(
            point=self.grid,
            primitive=primitive_tensor,
            vectorInput=vector_tensor,
            output=self._vector_stress,
        )
        self.module.theory33ComposeNineFieldKernel(
            point=self.grid,
            modelMode=self.model_mode,
            primitive=primitive_tensor,
            tracer=tracer_tensor,
            vectorStress=self._vector_stress,
            conserved=self._conserved_before,
        )
        self.module.theory33DragKernel(
            point=self.grid,
            modelMode=self.model_mode,
            dt=float(dt),
            lapse=float(lapse),
            relaxationTime=float(relaxation_time),
            primitiveInput=primitive_tensor,
            primitiveOutput=self._primitive_after,
            diagnostics=self._drag_diagnostics,
        )
        self.module.theory33ComposeNineFieldKernel(
            point=self.grid,
            modelMode=self.model_mode,
            primitive=self._primitive_after,
            tracer=tracer_tensor,
            vectorStress=self._vector_stress,
            conserved=self._conserved_after,
        )
        return VulkanTheory33SourceSnapshot(
            primitive_before=primitive_values.copy(),
            primitive_after_drag=self._primitive_after.to_numpy(),
            drag_diagnostics=self._drag_diagnostics.to_numpy(),
            vector_stress=self._vector_stress.to_numpy(),
            conserved_before=self._conserved_before.to_numpy(),
            conserved_after=self._conserved_after.to_numpy(),
            constitutive_model=self.constitutive_model,
        )

    def principal(
        self,
        primitive: np.ndarray,
        geometry: np.ndarray,
        *,
        axis: int,
    ) -> VulkanTheory33PrincipalSnapshot:
        """Evaluate curved-space U, F^axis, and Carter path data on Vulkan.

        Geometry columns are a row-major spatial metric, lapse, and the three
        shift components.
        """
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        primitive_values = self._array(
            primitive,
            shape=(self.points, PRIMITIVE_3D_WIDTH),
            name="primitive",
        )
        geometry_values = self._array(
            geometry,
            shape=(self.points, GEOMETRY_3D_WIDTH),
            name="geometry",
        )
        primitive_tensor = self.spy.Tensor.from_numpy(
            self.device, primitive_values
        )
        geometry_tensor = self.spy.Tensor.from_numpy(
            self.device, geometry_values
        )
        self.module.theory33FullPrincipalKernel(
            point=self.grid,
            axis=axis,
            modelMode=self.model_mode,
            primitive=primitive_tensor,
            geometryInput=geometry_tensor,
            conserved=self._principal_conserved,
            flux=self._principal_flux,
            path=self._principal_path,
            valid=self._principal_valid,
        )
        return VulkanTheory33PrincipalSnapshot(
            conserved=self._principal_conserved.to_numpy(),
            flux=self._principal_flux.to_numpy(),
            path=self._principal_path.to_numpy(),
            valid=self._principal_valid.to_numpy(),
            axis=axis,
            constitutive_model=self.constitutive_model,
        )

    def curved_vector_stress(
        self,
        primitive: np.ndarray,
        vector_input: np.ndarray,
        geometry: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the mixed-Proca stress on a general spatial metric."""
        primitive_values = self._array(
            primitive,
            shape=(self.points, PRIMITIVE_3D_WIDTH),
            name="primitive",
        )
        vector_values = self._array(
            vector_input,
            shape=(self.points, VECTOR_INPUT_WIDTH),
            name="vector_input",
        )
        geometry_values = self._array(
            geometry,
            shape=(self.points, GEOMETRY_3D_WIDTH),
            name="geometry",
        )
        primitive_tensor = self.spy.Tensor.from_numpy(
            self.device, primitive_values
        )
        vector_tensor = self.spy.Tensor.from_numpy(
            self.device, vector_values
        )
        geometry_tensor = self.spy.Tensor.from_numpy(
            self.device, geometry_values
        )
        self.module.theory33SingleCurvedVectorStressKernel(
            point=self.grid,
            fieldIndex=0,
            vectorInput=vector_tensor,
            geometryInput=geometry_tensor,
            output=self._single_vector_stress_a,
        )
        self.module.theory33SingleCurvedVectorStressKernel(
            point=self.grid,
            fieldIndex=1,
            vectorInput=vector_tensor,
            geometryInput=geometry_tensor,
            output=self._single_vector_stress_b,
        )
        self.module.theory33CombineCurvedVectorStressKernel(
            point=self.grid,
            primitive=primitive_tensor,
            vectorInput=vector_tensor,
            geometryInput=geometry_tensor,
            stressA=self._single_vector_stress_a,
            stressB=self._single_vector_stress_b,
            output=self._vector_stress,
        )
        return self._vector_stress.to_numpy()

    def full_eigensystem(
        self,
        primitive: np.ndarray,
        geometry: np.ndarray,
        *,
        axis: int,
    ) -> VulkanTheory33FullEigensystemSnapshot:
        """Form the complete numerical Carter flux/path eigensystem."""
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        primitive_values = self._array(
            primitive,
            shape=(self.points, PRIMITIVE_3D_WIDTH),
            name="primitive",
        )
        geometry_values = self._array(
            geometry,
            shape=(self.points, GEOMETRY_3D_WIDTH),
            name="geometry",
        )
        primitive_tensor = self.spy.Tensor.from_numpy(
            self.device, primitive_values
        )
        geometry_tensor = self.spy.Tensor.from_numpy(
            self.device, geometry_values
        )
        self.module.theory33FullEigensystemKernel(
            point=self.grid,
            axis=axis,
            modelMode=self.model_mode,
            primitive=primitive_tensor,
            geometryInput=geometry_tensor,
            symbol=self._full_symbol,
            rightBasis=self._full_right_basis,
            leftBasis=self._full_left_basis,
            eigenvalues=self._full_eigenvalues,
            audit=self._full_eigen_audit,
        )
        return VulkanTheory33FullEigensystemSnapshot(
            symbol=self._full_symbol.to_numpy().reshape(
                self.points, 9, 9
            ),
            right_basis=self._full_right_basis.to_numpy().reshape(
                self.points, 9, 9
            ),
            left_basis=self._full_left_basis.to_numpy().reshape(
                self.points, 9, 9
            ),
            eigenvalues=self._full_eigenvalues.to_numpy(),
            audit=self._full_eigen_audit.to_numpy(),
            axis=axis,
            constitutive_model=self.constitutive_model,
        )


class VulkanTheory33MultidimensionalBackend:
    """Resident periodic multidimensional eleven-component Carter solver.

    Grid indexing is x-fastest. Host arrays are flattened as ``(points,
    fields)``; ``shape`` is supplied as ``(nx, ny, nz)`` with inactive
    dimensions equal to one.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        lengths: tuple[float, ...] | None = None,
        constitutive_model: str = "analytic_m1",
        relaxation_time: float = 0.01,
        module_path: str | Path | None = None,
    ) -> None:
        if not 1 <= len(shape) <= 3:
            raise ValueError("shape must contain one to three dimensions")
        expanded = tuple(int(value) for value in shape) + (1,) * (
            3 - len(shape)
        )
        if any(value < 1 for value in expanded):
            raise ValueError("grid dimensions must be positive")
        if any(
            value < 4 for value in expanded[: len(shape)]
        ):
            raise ValueError("each active dimension needs at least four cells")
        if constitutive_model not in CONSTITUTIVE_MODES:
            raise ValueError("unknown constitutive model")
        if relaxation_time <= 0.0:
            raise ValueError("relaxation_time must be positive")
        self.shape = expanded
        self.ndim = len(shape)
        self.points = int(np.prod(expanded))
        if lengths is None:
            lengths = (1.0,) * self.ndim
        if len(lengths) != self.ndim or any(v <= 0.0 for v in lengths):
            raise ValueError("lengths must match active positive dimensions")
        self.lengths = tuple(float(v) for v in lengths) + (1.0,) * (
            3 - self.ndim
        )
        self.spacing = tuple(
            self.lengths[axis] / self.shape[axis] for axis in range(3)
        )
        self.constitutive_model = constitutive_model
        self.model_mode = CONSTITUTIVE_MODES[constitutive_model]
        self.relaxation_time = float(relaxation_time)
        self.spy = _slangpy()
        root = Path(__file__).resolve().parents[1]
        self.module_path = (
            root / "tesseract_nr_gpu.slang"
            if module_path is None
            else Path(module_path).resolve()
        )
        self.device = self.spy.create_device(
            type=self.spy.DeviceType.vulkan,
            include_paths=[self.module_path.parent],
        )
        self.module = self.spy.Module.load_from_file(
            self.device, str(self.module_path)
        )
        self.grid = self.spy.grid(shape=(self.points,))
        tensor = self.spy.Tensor.empty
        scalar = lambda: tensor(
            self.device, shape=(self.points,), dtype=self.module.float
        )
        matrix = lambda width: tensor(
            self.device,
            shape=(self.points, width),
            dtype=self.module.float,
        )
        self._state_buffers = [matrix(11) for _ in range(5)]
        self._primitive_buffers = [matrix(9) for _ in range(4)]
        self._diagnostic_buffers = [
            matrix(FULL_RECOVERY_DIAGNOSTIC_WIDTH) for _ in range(2)
        ]
        self._drag_diagnostics = matrix(FULL_DRAG_DIAGNOSTIC_WIDTH)
        self._vector_stress = matrix(VECTOR_STRESS_WIDTH)
        self._single_vector_stress_a = matrix(13)
        self._single_vector_stress_b = matrix(13)
        self._flux = matrix(11)
        self._path = matrix(PRINCIPAL_PATH_WIDTH)
        self._speed = scalar()
        self._interface = matrix(11)
        self._specific_tracer = scalar()
        self._directional_rhs = [matrix(11) for _ in range(3)]
        self._rhs = matrix(11)
        self._geometry: Any | None = None
        self._vector_input: Any | None = None
        self._tracer: Any | None = None
        self._current_state = 0
        self._current_primitive = 0
        self._initialized = False

    def _host_array(
        self, values: np.ndarray, width: int, name: str
    ) -> np.ndarray:
        return VulkanTheory33SourceBackend._array(
            values,
            shape=(self.points, width),
            name=name,
        )

    def _vector_stress_dispatch(self, primitive: Any) -> None:
        assert self._vector_input is not None
        assert self._geometry is not None
        self.module.theory33SingleCurvedVectorStressKernel(
            point=self.grid,
            fieldIndex=0,
            vectorInput=self._vector_input,
            geometryInput=self._geometry,
            output=self._single_vector_stress_a,
        )
        self.module.theory33SingleCurvedVectorStressKernel(
            point=self.grid,
            fieldIndex=1,
            vectorInput=self._vector_input,
            geometryInput=self._geometry,
            output=self._single_vector_stress_b,
        )
        self.module.theory33CombineCurvedVectorStressKernel(
            point=self.grid,
            primitive=primitive,
            vectorInput=self._vector_input,
            geometryInput=self._geometry,
            stressA=self._single_vector_stress_a,
            stressB=self._single_vector_stress_b,
            output=self._vector_stress,
        )

    def initialize(
        self,
        primitive: np.ndarray,
        vector_input: np.ndarray,
        geometry: np.ndarray,
        tracer: np.ndarray,
    ) -> VulkanTheory33MultidimensionalState:
        primitive_values = self._host_array(primitive, 9, "primitive")
        vector_values = self._host_array(
            vector_input, VECTOR_INPUT_WIDTH, "vector_input"
        )
        geometry_values = self._host_array(
            geometry, GEOMETRY_3D_WIDTH, "geometry"
        )
        tracer_values = np.asarray(tracer, dtype=np.float32)
        if tracer_values.shape != (self.points,):
            raise ValueError("tracer has the wrong shape")
        self._geometry = self.spy.Tensor.from_numpy(
            self.device, geometry_values
        )
        self._vector_input = self.spy.Tensor.from_numpy(
            self.device, vector_values
        )
        self._tracer = self.spy.Tensor.from_numpy(
            self.device, tracer_values
        )
        primitive_tensor = self.spy.Tensor.from_numpy(
            self.device, primitive_values
        )
        self._primitive_buffers[0] = primitive_tensor
        self._current_primitive = 0
        self._vector_stress_dispatch(primitive_tensor)
        state = self._state_buffers[0]
        self.module.theory33FullComposeCurvedKernel(
            point=self.grid,
            modelMode=self.model_mode,
            primitive=primitive_tensor,
            tracer=self._tracer,
            geometryInput=self._geometry,
            vectorStress=self._vector_stress,
            state=state,
        )
        self._current_state = 0
        self._initialized = True
        diagnostics = self._diagnostic_buffers[0]
        self.module.theory33FullRecoverCurvedKernel(
            point=self.grid,
            modelMode=self.model_mode,
            state=state,
            geometryInput=self._geometry,
            vectorStress=self._vector_stress,
            guessInput=primitive_tensor,
            guessOutput=self._primitive_buffers[1],
            diagnostics=diagnostics,
        )
        self._current_primitive = 1
        return VulkanTheory33MultidimensionalState(
            tensor=state,
            primitive=self._primitive_buffers[1],
            diagnostics=diagnostics,
            vector_stress=self._vector_stress,
            geometry=self._geometry,
            shape=self.shape,
        )

    def _recover(
        self, state: Any, guess_input: Any, guess_output: Any
    ) -> None:
        self.module.theory33FullRecoverCurvedKernel(
            point=self.grid,
            modelMode=self.model_mode,
            state=state,
            geometryInput=self._geometry,
            vectorStress=self._vector_stress,
            guessInput=guess_input,
            guessOutput=guess_output,
            diagnostics=self._diagnostic_buffers[1],
        )

    def _rhs_dispatch(self, state: Any, primitive: Any) -> None:
        nx, ny, nz = self.shape
        for axis in range(3):
            target = self._directional_rhs[axis]
            if axis >= self.ndim:
                self.module.theory33FullZeroRhsKernel(
                    point=self.grid, rhs=target
                )
                continue
            self.module.theory33FullFluxCurvedKernel(
                point=self.grid,
                axis=axis,
                modelMode=self.model_mode,
                state=state,
                primitive=primitive,
                geometryInput=self._geometry,
                vectorStress=self._vector_stress,
                flux=self._flux,
                pathData=self._path,
                speed=self._speed,
            )
            self.module.theory33FullInterfaceKernel(
                point=self.grid,
                axis=axis,
                nx=nx,
                ny=ny,
                nz=nz,
                state=state,
                flux=self._flux,
                speed=self._speed,
                interfaceFlux=self._interface,
            )
            self.module.theory33FullDirectionalRhsKernel(
                point=self.grid,
                axis=axis,
                nx=nx,
                ny=ny,
                nz=nz,
                inverseSpacing=float(1.0 / self.spacing[axis]),
                interfaceFlux=self._interface,
                pathData=self._path,
                rhs=target,
            )
        self.module.theory33FullSumRhsKernel(
            point=self.grid,
            rhsX=self._directional_rhs[0],
            rhsY=self._directional_rhs[1],
            rhsZ=self._directional_rhs[2],
            rhs=self._rhs,
        )

    def _stage(
        self,
        *,
        base: Any,
        stage: Any,
        output: Any,
        dt: float,
        base_weight: float,
        evolved_weight: float,
    ) -> None:
        self.module.theory33FullStageKernel(
            point=self.grid,
            dt=float(dt),
            baseWeight=float(base_weight),
            evolvedWeight=float(evolved_weight),
            baseState=base,
            stageState=stage,
            rhs=self._rhs,
            output=output,
        )

    def _drag_half_step(
        self,
        state: Any,
        primitive: Any,
        output_state: Any,
        output_primitive: Any,
        dt: float,
    ) -> None:
        self.module.theory33FullDragCurvedKernel(
            point=self.grid,
            modelMode=self.model_mode,
            dt=float(0.5 * dt),
            relaxationTime=self.relaxation_time,
            geometryInput=self._geometry,
            primitiveInput=primitive,
            primitiveOutput=output_primitive,
            diagnostics=self._drag_diagnostics,
        )
        self.module.theory33FullExtractTracerKernel(
            point=self.grid,
            state=state,
            tracer=self._specific_tracer,
        )
        self._vector_stress_dispatch(output_primitive)
        self.module.theory33FullComposeCurvedKernel(
            point=self.grid,
            modelMode=self.model_mode,
            primitive=output_primitive,
            tracer=self._specific_tracer,
            geometryInput=self._geometry,
            vectorStress=self._vector_stress,
            state=output_state,
        )

    def step(
        self,
        state: VulkanTheory33MultidimensionalState,
        dt: float,
        *,
        apply_drag: bool = True,
    ) -> VulkanTheory33MultidimensionalState:
        if not self._initialized:
            raise RuntimeError("initialize before stepping")
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be positive and finite")
        state_index = self._current_state
        primitive_index = self._current_primitive
        base = self._state_buffers[state_index]
        primitive0 = self._primitive_buffers[primitive_index]
        if apply_drag:
            split_state = self._state_buffers[
                (state_index + 1) % 5
            ]
            primitive1 = self._primitive_buffers[
                (primitive_index + 1) % 4
            ]
            self._drag_half_step(
                base, primitive0, split_state, primitive1, dt
            )
            base = split_state
            primitive0 = primitive1
        first = self._state_buffers[(state_index + 2) % 5]
        second = self._state_buffers[(state_index + 3) % 5]
        final = self._state_buffers[(state_index + 4) % 5]
        primitive1 = self._primitive_buffers[
            (primitive_index + 1) % 4
        ]
        primitive2 = self._primitive_buffers[
            (primitive_index + 2) % 4
        ]
        primitive3 = self._primitive_buffers[
            (primitive_index + 3) % 4
        ]
        self._recover(base, primitive0, primitive1)
        self._vector_stress_dispatch(primitive1)
        self._recover(base, primitive1, primitive2)
        self._rhs_dispatch(base, primitive2)
        self._stage(
            base=base,
            stage=base,
            output=first,
            dt=dt,
            base_weight=0.0,
            evolved_weight=1.0,
        )
        self._recover(first, primitive2, primitive3)
        self._vector_stress_dispatch(primitive3)
        self._rhs_dispatch(first, primitive3)
        self._stage(
            base=base,
            stage=first,
            output=second,
            dt=dt,
            base_weight=0.75,
            evolved_weight=0.25,
        )
        self._recover(second, primitive3, primitive1)
        self._vector_stress_dispatch(primitive1)
        self._rhs_dispatch(second, primitive1)
        self._stage(
            base=base,
            stage=second,
            output=final,
            dt=dt,
            base_weight=1.0 / 3.0,
            evolved_weight=2.0 / 3.0,
        )
        self._recover(final, primitive1, primitive2)
        result_state = final
        result_primitive = primitive2
        result_state_index = (state_index + 4) % 5
        result_primitive_index = (primitive_index + 2) % 4
        if apply_drag:
            dragged = self._state_buffers[state_index]
            primitive3 = self._primitive_buffers[
                (primitive_index + 3) % 4
            ]
            self._drag_half_step(
                final, primitive2, dragged, primitive3, dt
            )
            result_state = dragged
            result_primitive = primitive3
            result_state_index = state_index
            result_primitive_index = (primitive_index + 3) % 4
        self._vector_stress_dispatch(result_primitive)
        self._current_state = result_state_index
        self._current_primitive = result_primitive_index
        state.tensor = result_state
        state.primitive = result_primitive
        state.diagnostics = self._diagnostic_buffers[1]
        state.vector_stress = self._vector_stress
        state.time += float(dt)
        state.steps += 1
        return state

    @staticmethod
    def snapshot(
        state: VulkanTheory33MultidimensionalState,
    ) -> VulkanTheory33MultidimensionalSnapshot:
        state.host_reads += 1
        return VulkanTheory33MultidimensionalSnapshot(
            state=state.tensor.to_numpy(),
            primitive=state.primitive.to_numpy(),
            diagnostics=state.diagnostics.to_numpy(),
            vector_stress=state.vector_stress.to_numpy(),
            geometry=state.geometry.to_numpy(),
            shape=state.shape,
            time=state.time,
            steps=state.steps,
        )


class VulkanTesseractCoupledBackend(
    VulkanTheory33MultidimensionalBackend
):
    """Resident Strang-coupled Carter/CCZ4 multidimensional evolution.

    Geometry takes half steps around one full Carter transport/implicit-drag
    step. Each CCZ4 half step uses projected SSP-RK3 stages and refreshes the
    curved primitive recovery and full Eulerian matter stress at every stage.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        lengths: tuple[float, ...] | None = None,
        constitutive_model: str = "analytic_m1",
        relaxation_time: float = 0.01,
        kappa_gravity: float = 8.0 * np.pi,
        kappa1: float = 0.1,
        kappa2: float = 0.0,
        kappa3: float = 1.0,
        cosmological_constant: float = 0.0,
        shift_speed: float = 1.0,
        eta_shift: float = 1.0,
        slicing: str = "1+log",
        evolve_shift: bool = True,
        module_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            shape,
            lengths=lengths,
            constitutive_model=constitutive_model,
            relaxation_time=relaxation_time,
            module_path=module_path,
        )
        if kappa_gravity <= 0.0 or kappa1 < 0.0:
            raise ValueError(
                "kappa_gravity must be positive and kappa1 nonnegative"
            )
        if kappa2 <= -1.0:
            raise ValueError("kappa2 must be greater than -1")
        if shift_speed <= 0.0 or eta_shift < 0.0:
            raise ValueError("invalid shift parameters")
        if slicing not in {"1+log", "harmonic"}:
            raise ValueError("slicing must be '1+log' or 'harmonic'")
        self.kappa_gravity = float(kappa_gravity)
        self.kappa1 = float(kappa1)
        self.kappa2 = float(kappa2)
        self.kappa3 = float(kappa3)
        self.cosmological_constant = float(cosmological_constant)
        self.shift_speed = float(shift_speed)
        self.eta_shift = float(eta_shift)
        self.slicing_mode = 0 if slicing == "1+log" else 1
        self.evolve_shift = int(evolve_shift)
        tensor = self.spy.Tensor.empty
        matrix = lambda width: tensor(
            self.device,
            shape=(self.points, width),
            dtype=self.module.float,
        )
        self._ccz4_buffers = [matrix(CCZ4_STATE_WIDTH) for _ in range(8)]
        self._ccz4_vacuum_rhs = matrix(CCZ4_STATE_WIDTH)
        self._ccz4_matter_rhs = matrix(CCZ4_STATE_WIDTH)
        self._ccz4_rhs = matrix(CCZ4_STATE_WIDTH)
        self._ccz4_projection_audit = matrix(
            CCZ4_PROJECTION_AUDIT_WIDTH
        )
        self._ccz4_extrinsic = matrix(9)
        self._matter_stress = matrix(13)
        self._current_ccz4 = 0

    def _project_ccz4(self, source: Any, target: Any) -> None:
        self.module.ccz4ProjectGeometryKernel(
            point=self.grid,
            stateInput=source,
            stateOutput=target,
            audit=self._ccz4_projection_audit,
        )

    def _set_geometry(self, ccz4_state: Any) -> None:
        self.module.ccz4PhysicalGeometryKernel(
            point=self.grid,
            state=ccz4_state,
            geometryOutput=self._geometry,
            extrinsicCurvature=self._ccz4_extrinsic,
        )

    def initialize(
        self,
        primitive: np.ndarray,
        vector_input: np.ndarray,
        ccz4_state: np.ndarray,
        tracer: np.ndarray,
    ) -> VulkanTesseractCoupledState:
        ccz4_values = self._host_array(
            ccz4_state, CCZ4_STATE_WIDTH, "ccz4_state"
        )
        dummy_geometry = np.zeros(
            (self.points, GEOMETRY_3D_WIDTH), dtype=np.float32
        )
        dummy_geometry[:, :9] = np.eye(3, dtype=np.float32).reshape(
            1, 9
        )
        dummy_geometry[:, 9] = 1.0
        matter = super().initialize(
            primitive, vector_input, dummy_geometry, tracer
        )
        uploaded = self.spy.Tensor.from_numpy(self.device, ccz4_values)
        self._ccz4_buffers[0] = uploaded
        self._project_ccz4(uploaded, self._ccz4_buffers[1])
        self._current_ccz4 = 1
        self._set_geometry(self._ccz4_buffers[1])
        self._vector_stress_dispatch(matter.primitive)
        self.module.theory33FullComposeCurvedKernel(
            point=self.grid,
            modelMode=self.model_mode,
            primitive=matter.primitive,
            tracer=self._tracer,
            geometryInput=self._geometry,
            vectorStress=self._vector_stress,
            state=matter.tensor,
        )
        recovered_index = (self._current_primitive + 1) % 4
        self._recover(
            matter.tensor,
            matter.primitive,
            self._primitive_buffers[recovered_index],
        )
        self._current_primitive = recovered_index
        matter.primitive = self._primitive_buffers[recovered_index]
        self._vector_stress_dispatch(matter.primitive)
        matter.geometry = self._geometry
        return VulkanTesseractCoupledState(
            matter=matter,
            ccz4=self._ccz4_buffers[1],
        )

    def _ccz4_stage_rhs(
        self,
        ccz4_state: Any,
        matter: VulkanTheory33MultidimensionalState,
        guess_index: int,
    ) -> int:
        self._set_geometry(ccz4_state)
        first_index = (guess_index + 1) % 4
        second_index = (guess_index + 2) % 4
        self._vector_stress_dispatch(
            self._primitive_buffers[guess_index]
        )
        self._recover(
            matter.tensor,
            self._primitive_buffers[guess_index],
            self._primitive_buffers[first_index],
        )
        self._vector_stress_dispatch(
            self._primitive_buffers[first_index]
        )
        self._recover(
            matter.tensor,
            self._primitive_buffers[first_index],
            self._primitive_buffers[second_index],
        )
        primitive = self._primitive_buffers[second_index]
        self._vector_stress_dispatch(primitive)
        self.module.theory33FullMatterStressKernel(
            point=self.grid,
            modelMode=self.model_mode,
            primitive=primitive,
            geometryInput=self._geometry,
            vectorStress=self._vector_stress,
            matterStress=self._matter_stress,
        )
        nx, ny, nz = self.shape
        self.module.ccz4VacuumRhsKernel(
            point=self.grid,
            nx=nx,
            ny=ny,
            nz=nz,
            inverseDx=float(1.0 / self.spacing[0]),
            inverseDy=float(1.0 / self.spacing[1]),
            inverseDz=float(1.0 / self.spacing[2]),
            kappa1=self.kappa1,
            kappa2=self.kappa2,
            kappa3=self.kappa3,
            cosmologicalConstant=self.cosmological_constant,
            shiftSpeed=self.shift_speed,
            etaShift=self.eta_shift,
            slicingMode=self.slicing_mode,
            evolveShift=self.evolve_shift,
            state=ccz4_state,
            rhs=self._ccz4_vacuum_rhs,
        )
        self.module.ccz4MatterSourceKernel(
            point=self.grid,
            kappaGravity=self.kappa_gravity,
            projectedState=ccz4_state,
            matterStress=self._matter_stress,
            source=self._ccz4_matter_rhs,
        )
        self.module.ccz4SumRhsKernel(
            point=self.grid,
            vacuumRhs=self._ccz4_vacuum_rhs,
            matterRhs=self._ccz4_matter_rhs,
            rhs=self._ccz4_rhs,
        )
        return second_index

    def _advance_ccz4(
        self,
        state: VulkanTesseractCoupledState,
        dt: float,
    ) -> None:
        base_index = self._current_ccz4
        indices = [
            (base_index + offset) % 8 for offset in range(1, 7)
        ]
        base = self._ccz4_buffers[base_index]
        guess_index = self._current_primitive
        guess_index = self._ccz4_stage_rhs(
            base, state.matter, guess_index
        )
        self.module.ccz4StageKernel(
            point=self.grid,
            dt=float(dt),
            baseWeight=0.0,
            evolvedWeight=1.0,
            baseState=base,
            stageState=base,
            rhs=self._ccz4_rhs,
            output=self._ccz4_buffers[indices[0]],
        )
        self._project_ccz4(
            self._ccz4_buffers[indices[0]],
            self._ccz4_buffers[indices[1]],
        )
        guess_index = self._ccz4_stage_rhs(
            self._ccz4_buffers[indices[1]], state.matter, guess_index
        )
        self.module.ccz4StageKernel(
            point=self.grid,
            dt=float(dt),
            baseWeight=0.75,
            evolvedWeight=0.25,
            baseState=base,
            stageState=self._ccz4_buffers[indices[1]],
            rhs=self._ccz4_rhs,
            output=self._ccz4_buffers[indices[2]],
        )
        self._project_ccz4(
            self._ccz4_buffers[indices[2]],
            self._ccz4_buffers[indices[3]],
        )
        guess_index = self._ccz4_stage_rhs(
            self._ccz4_buffers[indices[3]], state.matter, guess_index
        )
        self.module.ccz4StageKernel(
            point=self.grid,
            dt=float(dt),
            baseWeight=1.0 / 3.0,
            evolvedWeight=2.0 / 3.0,
            baseState=base,
            stageState=self._ccz4_buffers[indices[3]],
            rhs=self._ccz4_rhs,
            output=self._ccz4_buffers[indices[4]],
        )
        self._project_ccz4(
            self._ccz4_buffers[indices[4]],
            self._ccz4_buffers[indices[5]],
        )
        self._current_ccz4 = indices[5]
        state.ccz4 = self._ccz4_buffers[self._current_ccz4]
        self._set_geometry(state.ccz4)
        first_index = (guess_index + 1) % 4
        second_index = (guess_index + 2) % 4
        self._vector_stress_dispatch(
            self._primitive_buffers[guess_index]
        )
        self._recover(
            state.matter.tensor,
            self._primitive_buffers[guess_index],
            self._primitive_buffers[first_index],
        )
        self._vector_stress_dispatch(
            self._primitive_buffers[first_index]
        )
        self._recover(
            state.matter.tensor,
            self._primitive_buffers[first_index],
            self._primitive_buffers[second_index],
        )
        self._current_primitive = second_index
        state.matter.primitive = self._primitive_buffers[second_index]
        self._vector_stress_dispatch(state.matter.primitive)
        state.matter.geometry = self._geometry

    def step(
        self,
        state: VulkanTesseractCoupledState,
        dt: float,
        *,
        apply_drag: bool = True,
    ) -> VulkanTesseractCoupledState:
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be positive and finite")
        self._advance_ccz4(state, 0.5 * dt)
        super().step(state.matter, dt, apply_drag=apply_drag)
        self._advance_ccz4(state, 0.5 * dt)
        state.time += float(dt)
        state.steps += 1
        return state

    def snapshot(
        self,
        state: VulkanTesseractCoupledState,
    ) -> VulkanTesseractCoupledSnapshot:
        state.host_reads += 1
        matter = VulkanTheory33MultidimensionalBackend.snapshot(
            state.matter
        )
        return VulkanTesseractCoupledSnapshot(
            matter=matter,
            ccz4_state=state.ccz4.to_numpy(),
            ccz4_rhs=self._ccz4_rhs.to_numpy(),
            ccz4_projection_audit=(
                self._ccz4_projection_audit.to_numpy()
            ),
            matter_stress=self._matter_stress.to_numpy(),
            time=state.time,
            steps=state.steps,
        )


class VulkanCCZ4GeometryBackend:
    """Packed GPU CCZ4 projection, geometry, and matter-source layer.

    State order is ``gt_ij, At_ij, phi, K, theta, GammaHat_i, alpha,
    beta_i, B_i`` with both tensors stored row-major.
    """

    def __init__(
        self,
        points: int,
        *,
        module_path: str | Path | None = None,
    ) -> None:
        if points < 1:
            raise ValueError("at least one CCZ4 point is required")
        self.points = int(points)
        self.spy = _slangpy()
        root = Path(__file__).resolve().parents[1]
        self.module_path = (
            root / "tesseract_nr_gpu.slang"
            if module_path is None
            else Path(module_path).resolve()
        )
        self.device = self.spy.create_device(
            type=self.spy.DeviceType.vulkan,
            include_paths=[self.module_path.parent],
        )
        self.module = self.spy.Module.load_from_file(
            self.device, str(self.module_path)
        )
        self.grid = self.spy.grid(shape=(self.points,))
        self._projected = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, CCZ4_STATE_WIDTH),
            dtype=self.module.float,
        )
        self._geometry = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, GEOMETRY_3D_WIDTH),
            dtype=self.module.float,
        )
        self._extrinsic = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, 9),
            dtype=self.module.float,
        )
        self._source = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, CCZ4_STATE_WIDTH),
            dtype=self.module.float,
        )
        self._vacuum_rhs = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, CCZ4_STATE_WIDTH),
            dtype=self.module.float,
        )
        self._audit = self.spy.Tensor.empty(
            self.device,
            shape=(self.points, CCZ4_PROJECTION_AUDIT_WIDTH),
            dtype=self.module.float,
        )

    def evaluate(
        self,
        state: np.ndarray,
        matter_stress: np.ndarray,
        *,
        kappa_gravity: float = 8.0 * np.pi,
    ) -> VulkanCCZ4GeometrySnapshot:
        values = VulkanTheory33SourceBackend._array(
            state,
            shape=(self.points, CCZ4_STATE_WIDTH),
            name="state",
        )
        matter = VulkanTheory33SourceBackend._array(
            matter_stress,
            shape=(self.points, 13),
            name="matter_stress",
        )
        if kappa_gravity <= 0.0 or not np.isfinite(kappa_gravity):
            raise ValueError("kappa_gravity must be positive and finite")
        state_tensor = self.spy.Tensor.from_numpy(self.device, values)
        matter_tensor = self.spy.Tensor.from_numpy(self.device, matter)
        self.module.ccz4ProjectGeometryKernel(
            point=self.grid,
            stateInput=state_tensor,
            stateOutput=self._projected,
            audit=self._audit,
        )
        self.module.ccz4PhysicalGeometryKernel(
            point=self.grid,
            state=self._projected,
            geometryOutput=self._geometry,
            extrinsicCurvature=self._extrinsic,
        )
        self.module.ccz4MatterSourceKernel(
            point=self.grid,
            kappaGravity=float(kappa_gravity),
            projectedState=self._projected,
            matterStress=matter_tensor,
            source=self._source,
        )
        return VulkanCCZ4GeometrySnapshot(
            projected_state=self._projected.to_numpy(),
            physical_geometry=self._geometry.to_numpy(),
            extrinsic_curvature=self._extrinsic.to_numpy().reshape(
                self.points, 3, 3
            ),
            matter_source=self._source.to_numpy(),
            projection_audit=self._audit.to_numpy(),
        )

    def vacuum_rhs(
        self,
        state: np.ndarray,
        *,
        shape: tuple[int, ...],
        lengths: tuple[float, ...] | None = None,
        kappa1: float = 0.1,
        kappa2: float = 0.0,
        kappa3: float = 1.0,
        cosmological_constant: float = 0.0,
        shift_speed: float = 1.0,
        eta_shift: float = 1.0,
        slicing: str = "1+log",
        evolve_shift: bool = True,
    ) -> VulkanCCZ4VacuumSnapshot:
        """Evaluate the derivative-containing periodic vacuum operator."""
        if not 1 <= len(shape) <= 3:
            raise ValueError("shape must contain one to three dimensions")
        expanded = tuple(int(value) for value in shape) + (1,) * (
            3 - len(shape)
        )
        if int(np.prod(expanded)) != self.points:
            raise ValueError("shape does not match backend point count")
        if lengths is None:
            lengths = (1.0,) * len(shape)
        if len(lengths) != len(shape) or any(v <= 0 for v in lengths):
            raise ValueError("lengths must match shape")
        spacing = tuple(
            float(lengths[axis]) / expanded[axis]
            for axis in range(len(shape))
        ) + (1.0,) * (3 - len(shape))
        if kappa1 < 0.0 or kappa2 <= -1.0:
            raise ValueError("invalid CCZ4 damping parameters")
        if shift_speed <= 0.0 or eta_shift < 0.0:
            raise ValueError("invalid shift parameters")
        if slicing not in {"1+log", "harmonic"}:
            raise ValueError("unknown slicing condition")
        values = VulkanTheory33SourceBackend._array(
            state,
            shape=(self.points, CCZ4_STATE_WIDTH),
            name="state",
        )
        state_tensor = self.spy.Tensor.from_numpy(self.device, values)
        self.module.ccz4ProjectGeometryKernel(
            point=self.grid,
            stateInput=state_tensor,
            stateOutput=self._projected,
            audit=self._audit,
        )
        self.module.ccz4VacuumRhsKernel(
            point=self.grid,
            nx=expanded[0],
            ny=expanded[1],
            nz=expanded[2],
            inverseDx=float(1.0 / spacing[0]),
            inverseDy=float(1.0 / spacing[1]),
            inverseDz=float(1.0 / spacing[2]),
            kappa1=float(kappa1),
            kappa2=float(kappa2),
            kappa3=float(kappa3),
            cosmologicalConstant=float(cosmological_constant),
            shiftSpeed=float(shift_speed),
            etaShift=float(eta_shift),
            slicingMode=0 if slicing == "1+log" else 1,
            evolveShift=int(evolve_shift),
            state=self._projected,
            rhs=self._vacuum_rhs,
        )
        return VulkanCCZ4VacuumSnapshot(
            projected_state=self._projected.to_numpy(),
            vacuum_rhs=self._vacuum_rhs.to_numpy(),
            projection_audit=self._audit.to_numpy(),
            shape=expanded,
        )
