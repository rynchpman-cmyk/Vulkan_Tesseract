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
FULL_CARTER_FLUX_PATH_WIDTH = 8
FULL_EIGEN_AUDIT_WIDTH = 6
CCZ4_STATE_WIDTH = 31
CCZ4_PROJECTION_AUDIT_WIDTH = 3
FULL_RECOVERY_DIAGNOSTIC_WIDTH = 8
FULL_DRAG_DIAGNOSTIC_WIDTH = 6
MIXED_VECTOR_DYNAMIC_WIDTH = 16

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
    "carter_characteristic_roe": 3,
}

FULL_RECONSTRUCTION_MODES = {
    "piecewise_constant": 0,
    "full_carter_roe": 1,
    "full_carter_weno5_z": 2,
}

BOUNDARY_MODES = {
    "periodic": 0,
    "radiative_outflow": 1,
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
                "'carter_characteristic_weno5_z', or "
                "'carter_characteristic_roe'"
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
        if self.reconstruction_mode in {2, 3}:
            self.module.theory33EigensystemKernel(
                point=self.grid,
                modelMode=self.model_mode,
                recoveryGuess=guess_output,
                rightBasis=self._right_basis,
                leftBasis=self._left_basis,
                eigenvalues=self._eigenvalues,
                audit=self._eigensystem_audit,
            )
            if self.reconstruction_mode == 2:
                self.module.theory33CharacteristicInterfaceKernel(
                    point=self.grid,
                    state=state,
                    flux=self._flux,
                    rightBasis=self._right_basis,
                    leftBasis=self._left_basis,
                    eigenvalues=self._eigenvalues,
                    highInterface=self._high_interface,
                )
            else:
                self.module.theory33CharacteristicRoeInterfaceKernel(
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


@dataclass(frozen=True)
class VulkanTesseractEvolvedSnapshot:
    coupled: VulkanTesseractCoupledSnapshot
    vector_state: np.ndarray
    vector_input: np.ndarray


@dataclass(frozen=True)
class VulkanPeriodicCMCReport:
    converged: bool
    iterations: int
    picard_iterations: int
    hamiltonian_residual_l2: float
    momentum_residual_l2: float
    trace_K: float
    minimum_conformal_factor: float
    maximum_conformal_factor: float
    pointwise_momentum_balance: float
    final_source_relative_change: float
    scalar_readbacks: int


@dataclass
class VulkanMixedProcaState:
    tensor: Any
    vector_input: Any
    geometry: Any
    matter_state: Any
    primitive: Any
    shape: tuple[int, int, int]
    time: float = 0.0
    steps: int = 0
    host_reads: int = 0


@dataclass(frozen=True)
class VulkanMixedProcaSnapshot:
    state: np.ndarray
    rhs: np.ndarray
    vector_input: np.ndarray
    geometry: np.ndarray
    shape: tuple[int, int, int]
    time: float
    steps: int


class VulkanMixedProcaBackend:
    """Resident mixed-Proca hyperbolic evolution and Gauss cleaning."""

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        lengths: tuple[float, ...] | None = None,
        cleaning_damping: float = 1.0,
        module_path: str | Path | None = None,
    ) -> None:
        if not 1 <= len(shape) <= 3 or any(value < 4 for value in shape):
            raise ValueError("mixed-Proca grid needs 1-3 dimensions >= 4")
        if cleaning_damping < 0.0:
            raise ValueError("cleaning_damping must be nonnegative")
        self.shape = tuple(shape) + (1,) * (3 - len(shape))
        self.ndim = len(shape)
        self.points = int(np.prod(self.shape))
        if lengths is None:
            lengths = (1.0,) * self.ndim
        if len(lengths) != self.ndim or any(value <= 0 for value in lengths):
            raise ValueError("lengths must match active dimensions")
        self.lengths = tuple(float(value) for value in lengths) + (
            1.0,
        ) * (3 - self.ndim)
        self.spacing = tuple(
            self.lengths[axis] / self.shape[axis] for axis in range(3)
        )
        self.cleaning_damping = float(cleaning_damping)
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
        matrix = lambda width: tensor(
            self.device,
            shape=(self.points, width),
            dtype=self.module.float,
        )
        self._state_buffers = [
            matrix(MIXED_VECTOR_DYNAMIC_WIDTH) for _ in range(4)
        ]
        self._rhs = matrix(MIXED_VECTOR_DYNAMIC_WIDTH)
        self._vector_input = matrix(VECTOR_INPUT_WIDTH)
        self._geometry: Any | None = None
        self._matter_state: Any | None = None
        self._primitive: Any | None = None
        self._current_state = 0

    @staticmethod
    def _array(values: np.ndarray, shape: tuple[int, int], name: str):
        array = np.asarray(values, dtype=np.float32)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape {shape}")
        return np.ascontiguousarray(array)

    def _refresh_vector_input(self, dynamic: Any) -> None:
        assert self._geometry is not None
        nx, ny, nz = self.shape
        self.module.theory33MixedVectorInputKernel(
            point=self.grid,
            nx=nx,
            ny=ny,
            nz=nz,
            inverseDx=float(1.0 / self.spacing[0]),
            inverseDy=float(1.0 / self.spacing[1]),
            inverseDz=float(1.0 / self.spacing[2]),
            dynamicState=dynamic,
            geometryInput=self._geometry,
            vectorInput=self._vector_input,
        )

    def initialize(
        self,
        dynamic_state: np.ndarray,
        matter_state: np.ndarray,
        primitive: np.ndarray,
        geometry: np.ndarray,
    ) -> VulkanMixedProcaState:
        dynamic = self.spy.Tensor.from_numpy(
            self.device,
            self._array(
                dynamic_state,
                (self.points, MIXED_VECTOR_DYNAMIC_WIDTH),
                "dynamic_state",
            ),
        )
        self._state_buffers[0] = dynamic
        self._matter_state = self.spy.Tensor.from_numpy(
            self.device,
            self._array(
                matter_state,
                (self.points, FULL_CARTER_STATE_WIDTH),
                "matter_state",
            ),
        )
        self._primitive = self.spy.Tensor.from_numpy(
            self.device,
            self._array(
                primitive,
                (self.points, PRIMITIVE_3D_WIDTH),
                "primitive",
            ),
        )
        self._geometry = self.spy.Tensor.from_numpy(
            self.device,
            self._array(
                geometry,
                (self.points, GEOMETRY_3D_WIDTH),
                "geometry",
            ),
        )
        self._current_state = 0
        self._refresh_vector_input(dynamic)
        return VulkanMixedProcaState(
            tensor=dynamic,
            vector_input=self._vector_input,
            geometry=self._geometry,
            matter_state=self._matter_state,
            primitive=self._primitive,
            shape=self.shape,
        )

    def _rhs_dispatch(self, dynamic: Any) -> None:
        assert self._geometry is not None
        assert self._matter_state is not None
        assert self._primitive is not None
        nx, ny, nz = self.shape
        self.module.theory33MixedVectorRhsKernel(
            point=self.grid,
            nx=nx,
            ny=ny,
            nz=nz,
            inverseDx=float(1.0 / self.spacing[0]),
            inverseDy=float(1.0 / self.spacing[1]),
            inverseDz=float(1.0 / self.spacing[2]),
            cleaningDamping=self.cleaning_damping,
            dynamicState=dynamic,
            matterState=self._matter_state,
            primitive=self._primitive,
            geometryInput=self._geometry,
            rhs=self._rhs,
        )

    def _stage(
        self,
        base: Any,
        stage: Any,
        output: Any,
        dt: float,
        base_weight: float,
        evolved_weight: float,
    ) -> None:
        self.module.theory33MixedVectorStageKernel(
            point=self.grid,
            dt=float(dt),
            baseWeight=float(base_weight),
            evolvedWeight=float(evolved_weight),
            baseState=base,
            stageState=stage,
            rhs=self._rhs,
            output=output,
        )

    def step(
        self, state: VulkanMixedProcaState, dt: float
    ) -> VulkanMixedProcaState:
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be positive and finite")
        base_index = self._current_state
        first_index = (base_index + 1) % 4
        second_index = (base_index + 2) % 4
        final_index = (base_index + 3) % 4
        base = self._state_buffers[base_index]
        first = self._state_buffers[first_index]
        second = self._state_buffers[second_index]
        final = self._state_buffers[final_index]
        self._rhs_dispatch(base)
        self._stage(base, base, first, dt, 0.0, 1.0)
        self._rhs_dispatch(first)
        self._stage(base, first, second, dt, 0.75, 0.25)
        self._rhs_dispatch(second)
        self._stage(
            base, second, final, dt, 1.0 / 3.0, 2.0 / 3.0
        )
        self._current_state = final_index
        self._refresh_vector_input(final)
        state.tensor = final
        state.vector_input = self._vector_input
        state.time += float(dt)
        state.steps += 1
        return state

    def rhs(self, state: VulkanMixedProcaState) -> np.ndarray:
        self._rhs_dispatch(state.tensor)
        state.host_reads += 1
        return self._rhs.to_numpy()

    def snapshot(
        self, state: VulkanMixedProcaState
    ) -> VulkanMixedProcaSnapshot:
        self._rhs_dispatch(state.tensor)
        self._refresh_vector_input(state.tensor)
        state.host_reads += 1
        return VulkanMixedProcaSnapshot(
            state=state.tensor.to_numpy(),
            rhs=self._rhs.to_numpy(),
            vector_input=self._vector_input.to_numpy(),
            geometry=state.geometry.to_numpy(),
            shape=state.shape,
            time=state.time,
            steps=state.steps,
        )


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
    """Resident multidimensional eleven-component Carter solver.

    Grid indexing is x-fastest. Host arrays are flattened as ``(points,
    fields)``; ``shape`` is supplied as ``(nx, ny, nz)`` with inactive
    dimensions equal to one.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        lengths: tuple[float, ...] | None = None,
        origins: tuple[float, ...] | None = None,
        constitutive_model: str = "analytic_m1",
        reconstruction: str = "piecewise_constant",
        boundary_mode: str = "periodic",
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
        if boundary_mode not in BOUNDARY_MODES:
            raise ValueError(
                "boundary_mode must be 'periodic' or "
                "'radiative_outflow'"
            )
        if boundary_mode == "radiative_outflow" and any(
            value < 6 for value in expanded[: len(shape)]
        ):
            raise ValueError(
                "radiative_outflow needs six cells per active axis"
            )
        if reconstruction not in FULL_RECONSTRUCTION_MODES:
            raise ValueError(
                "reconstruction must be 'piecewise_constant', "
                "'full_carter_roe', or 'full_carter_weno5_z'"
            )
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
        if origins is None:
            origins = tuple(-0.5 * value for value in lengths)
        if len(origins) != self.ndim or any(
            not np.isfinite(value) for value in origins
        ):
            raise ValueError("origins must match the active dimensions")
        self.origins = tuple(float(v) for v in origins) + (0.0,) * (
            3 - self.ndim
        )
        self.spacing = tuple(
            self.lengths[axis] / self.shape[axis] for axis in range(3)
        )
        self.constitutive_model = constitutive_model
        self.model_mode = CONSTITUTIVE_MODES[constitutive_model]
        self.reconstruction = reconstruction
        self.reconstruction_mode = FULL_RECONSTRUCTION_MODES[
            reconstruction
        ]
        self.boundary_mode = boundary_mode
        self.boundary_mode_id = BOUNDARY_MODES[boundary_mode]
        if (
            boundary_mode == "radiative_outflow"
            and reconstruction == "full_carter_weno5_z"
        ):
            raise ValueError(
                "radiative_outflow currently supports piecewise_constant "
                "and full_carter_roe reconstruction"
            )
        characteristic = reconstruction in {
            "full_carter_roe",
            "full_carter_weno5_z",
        }
        self.recovery_tolerance = 2.0e-6 if characteristic else 3.0e-5
        self.recovery_step = 1.0e-3 if characteristic else 1.5e-3
        self.phase_switch_tolerance = 4.0e-6 if characteristic else 0.0
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
        self._path = matrix(FULL_CARTER_FLUX_PATH_WIDTH)
        self._speed = scalar()
        self._interface = matrix(11)
        self._boundary_flux = matrix(11)
        self._full_face_right = matrix(81)
        self._full_face_left = matrix(81)
        self._full_face_eigenvalues = matrix(9)
        self._full_face_path = matrix(81)
        self._full_face_entropy_modes = matrix(9)
        self._full_face_transport_speed = scalar()
        self._full_face_audit = matrix(FULL_EIGEN_AUDIT_WIDTH)
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
            recoveryTolerance=self.recovery_tolerance,
            recoveryStep=self.recovery_step,
            phaseSwitchTolerance=self.phase_switch_tolerance,
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
            recoveryTolerance=self.recovery_tolerance,
            recoveryStep=self.recovery_step,
            phaseSwitchTolerance=self.phase_switch_tolerance,
            state=state,
            geometryInput=self._geometry,
            vectorStress=self._vector_stress,
            guessInput=guess_input,
            guessOutput=guess_output,
            diagnostics=self._diagnostic_buffers[1],
        )

    def linearize_conserved_corridor(
        self, state: VulkanTheory33MultidimensionalState
    ) -> VulkanTheory33MultidimensionalState:
        """Replace a 1D state by its resident straight conserved path."""
        if self.ndim != 1:
            raise ValueError("the conserved corridor is one-dimensional")
        output = self._state_buffers[(self._current_state + 1) % 5]
        primitive = self._primitive_buffers[
            (self._current_primitive + 1) % 4
        ]
        self.module.theory33FullLinearCorridorKernel(
            point=self.grid,
            inputState=state.tensor,
            outputState=output,
        )
        self._recover(output, state.primitive, primitive)
        self._vector_stress_dispatch(primitive)
        self._current_state = (self._current_state + 1) % 5
        self._current_primitive = (self._current_primitive + 1) % 4
        state.tensor = output
        state.primitive = primitive
        state.diagnostics = self._diagnostic_buffers[1]
        state.vector_stress = self._vector_stress
        return state

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
            if self.boundary_mode_id == 1:
                self.module.theory33FullOutflowBoundaryFluxKernel(
                    point=self.grid,
                    axis=axis,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    state=state,
                    primitive=primitive,
                    flux=self._flux,
                    boundaryFlux=self._boundary_flux,
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
            if self.reconstruction_mode in {1, 2}:
                self.module.theory33FullFaceEigensystemKernel(
                    point=self.grid,
                    axis=axis,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    modelMode=self.model_mode,
                    primitive=primitive,
                    geometryInput=self._geometry,
                    rightBasis=self._full_face_right,
                    leftBasis=self._full_face_left,
                    eigenvalues=self._full_face_eigenvalues,
                    pathSymbol=self._full_face_path,
                    entropyModes=self._full_face_entropy_modes,
                    transportSpeed=self._full_face_transport_speed,
                    audit=self._full_face_audit,
                )
                if self.reconstruction_mode == 1:
                    self.module.theory33FullRoeInterfaceKernel(
                        point=self.grid,
                        axis=axis,
                        nx=nx,
                        ny=ny,
                        nz=nz,
                        state=state,
                        flux=self._flux,
                        speed=self._speed,
                        rightBasis=self._full_face_right,
                        leftBasis=self._full_face_left,
                        eigenvalues=self._full_face_eigenvalues,
                        entropyModes=self._full_face_entropy_modes,
                        transportSpeed=self._full_face_transport_speed,
                        interfaceFlux=self._interface,
                    )
                else:
                    self.module.theory33FullCharacteristicInterfaceKernel(
                        point=self.grid,
                        axis=axis,
                        nx=nx,
                        ny=ny,
                        nz=nz,
                        state=state,
                        flux=self._flux,
                        speed=self._speed,
                        rightBasis=self._full_face_right,
                        leftBasis=self._full_face_left,
                        entropyModes=self._full_face_entropy_modes,
                        interfaceFlux=self._interface,
                    )
            if self.reconstruction_mode in {1, 2}:
                if self.boundary_mode_id == 1:
                    self.module.theory33FullRoeOutflowDirectionalRhsKernel(
                        point=self.grid,
                        axis=axis,
                        nx=nx,
                        ny=ny,
                        nz=nz,
                        inverseSpacing=float(
                            1.0 / self.spacing[axis]
                        ),
                        state=state,
                        interfaceFlux=self._interface,
                        boundaryFlux=self._boundary_flux,
                        pathSymbol=self._full_face_path,
                        rhs=target,
                    )
                else:
                    self.module.theory33FullRoeDirectionalRhsKernel(
                        point=self.grid,
                        axis=axis,
                        nx=nx,
                        ny=ny,
                        nz=nz,
                        inverseSpacing=float(
                            1.0 / self.spacing[axis]
                        ),
                        state=state,
                        interfaceFlux=self._interface,
                        pathSymbol=self._full_face_path,
                        rhs=target,
                    )
            else:
                if self.boundary_mode_id == 1:
                    self.module.theory33FullOutflowDirectionalRhsKernel(
                        point=self.grid,
                        axis=axis,
                        nx=nx,
                        ny=ny,
                        nz=nz,
                        inverseSpacing=float(
                            1.0 / self.spacing[axis]
                        ),
                        interfaceFlux=self._interface,
                        boundaryFlux=self._boundary_flux,
                        pathData=self._path,
                        rhs=target,
                    )
                else:
                    self.module.theory33FullDirectionalRhsKernel(
                        point=self.grid,
                        axis=axis,
                        nx=nx,
                        ny=ny,
                        nz=nz,
                        inverseSpacing=float(
                            1.0 / self.spacing[axis]
                        ),
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

    def _project_entropy(self, state: Any, primitive: Any) -> None:
        self.module.theory33FullProjectEntropyKernel(
            point=self.grid,
            primitive=primitive,
            stateInput=state,
            stateOutput=state,
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
        self.module.theory33FullConservativeDragCorrectKernel(
            point=self.grid,
            stateBefore=state,
            stateAfter=output_state,
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
        self._project_entropy(first, primitive3)
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
        self._project_entropy(second, primitive1)
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
        self._project_entropy(final, primitive2)
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

    def save_checkpoint(
        self,
        state: VulkanTheory33MultidimensionalState,
        path: str | Path,
    ) -> Path:
        """Write a restartable resident matter state."""
        snapshot = self.snapshot(state)
        target = Path(path)
        np.savez_compressed(
            target,
            schema=np.array(
                "vulkan-tesseract.multidimensional-checkpoint.v1"
            ),
            shape=np.asarray(self.shape, dtype=np.int64),
            lengths=np.asarray(self.lengths, dtype=np.float64),
            origins=np.asarray(self.origins, dtype=np.float64),
            constitutive_model=np.array(self.constitutive_model),
            reconstruction=np.array(self.reconstruction),
            boundary_mode=np.array(self.boundary_mode),
            relaxation_time=np.array(self.relaxation_time),
            state=snapshot.state,
            primitive=snapshot.primitive,
            vector_input=self._vector_input.to_numpy(),
            geometry=snapshot.geometry,
            time=np.array(snapshot.time),
            steps=np.array(snapshot.steps, dtype=np.int64),
        )
        return target

    def load_checkpoint(
        self, path: str | Path
    ) -> VulkanTheory33MultidimensionalState:
        """Restore a checkpoint and re-audit primitive recovery on-device."""
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            schema = str(checkpoint["schema"])
            if schema != "vulkan-tesseract.multidimensional-checkpoint.v1":
                raise ValueError(f"unsupported checkpoint schema {schema!r}")
            if tuple(checkpoint["shape"].tolist()) != self.shape:
                raise ValueError("checkpoint grid shape does not match backend")
            if str(checkpoint["constitutive_model"]) != self.constitutive_model:
                raise ValueError("checkpoint constitutive model does not match")
            if str(checkpoint["reconstruction"]) != self.reconstruction:
                raise ValueError("checkpoint reconstruction does not match")
            if (
                "boundary_mode" in checkpoint
                and str(checkpoint["boundary_mode"]) != self.boundary_mode
            ):
                raise ValueError("checkpoint boundary mode does not match")
            conserved = np.asarray(checkpoint["state"], dtype=np.float32)
            primitive = np.asarray(
                checkpoint["primitive"], dtype=np.float32
            )
            vector_input = np.asarray(
                checkpoint["vector_input"], dtype=np.float32
            )
            geometry = np.asarray(
                checkpoint["geometry"], dtype=np.float32
            )
            time = float(checkpoint["time"])
            steps = int(checkpoint["steps"])
        tracer = conserved[:, 10] / np.maximum(
            conserved[:, 0], np.float32(1.0e-12)
        )
        restored = self.initialize(
            primitive, vector_input, geometry, tracer
        )
        uploaded = self.spy.Tensor.from_numpy(self.device, conserved)
        self._state_buffers[0] = uploaded
        self._current_state = 0
        recovered_index = (self._current_primitive + 1) % 4
        self._recover(
            uploaded,
            restored.primitive,
            self._primitive_buffers[recovered_index],
        )
        self._current_primitive = recovered_index
        restored.tensor = uploaded
        restored.primitive = self._primitive_buffers[recovered_index]
        restored.diagnostics = self._diagnostic_buffers[1]
        self._vector_stress_dispatch(restored.primitive)
        restored.vector_stress = self._vector_stress
        restored.time = time
        restored.steps = steps
        return restored


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
        origins: tuple[float, ...] | None = None,
        constitutive_model: str = "analytic_m1",
        reconstruction: str = "piecewise_constant",
        boundary_mode: str = "periodic",
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
        derivative_order: int = 2,
        module_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            shape,
            lengths=lengths,
            origins=origins,
            constitutive_model=constitutive_model,
            reconstruction=reconstruction,
            boundary_mode=boundary_mode,
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
        if derivative_order not in {2, 4}:
            raise ValueError("derivative_order must be 2 or 4")
        if derivative_order == 4 and any(
            value < 8 for value in self.shape[: self.ndim]
        ):
            raise ValueError(
                "fourth-order CCZ4 derivatives need eight cells per axis"
            )
        self.kappa_gravity = float(kappa_gravity)
        self.kappa1 = float(kappa1)
        self.kappa2 = float(kappa2)
        self.kappa3 = float(kappa3)
        self.cosmological_constant = float(cosmological_constant)
        self.shift_speed = float(shift_speed)
        self.eta_shift = float(eta_shift)
        self.slicing_mode = 0 if slicing == "1+log" else 1
        self.evolve_shift = int(evolve_shift)
        self.derivative_order = int(derivative_order)
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
            derivativeOrder=self.derivative_order,
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
        if self.boundary_mode_id == 1:
            self.module.theory33CCZ4RadiativeBoundaryKernel(
                point=self.grid,
                nx=nx,
                ny=ny,
                nz=nz,
                inverseDx=float(1.0 / self.spacing[0]),
                inverseDy=float(1.0 / self.spacing[1]),
                inverseDz=float(1.0 / self.spacing[2]),
                originX=self.origins[0],
                originY=self.origins[1],
                originZ=self.origins[2],
                derivativeOrder=self.derivative_order,
                constraintDamping=1.0,
                state=ccz4_state,
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

    def save_checkpoint(
        self,
        state: VulkanTesseractCoupledState,
        path: str | Path,
    ) -> Path:
        """Write a restartable coupled Carter/CCZ4 state."""
        snapshot = self.snapshot(state)
        target = Path(path)
        np.savez_compressed(
            target,
            schema=np.array("vulkan-tesseract.coupled-checkpoint.v1"),
            shape=np.asarray(self.shape, dtype=np.int64),
            lengths=np.asarray(self.lengths, dtype=np.float64),
            origins=np.asarray(self.origins, dtype=np.float64),
            constitutive_model=np.array(self.constitutive_model),
            reconstruction=np.array(self.reconstruction),
            boundary_mode=np.array(self.boundary_mode),
            derivative_order=np.array(
                self.derivative_order, dtype=np.int64
            ),
            relaxation_time=np.array(self.relaxation_time),
            matter_state=snapshot.matter.state,
            primitive=snapshot.matter.primitive,
            vector_input=self._vector_input.to_numpy(),
            ccz4_state=snapshot.ccz4_state,
            time=np.array(snapshot.time),
            steps=np.array(snapshot.steps, dtype=np.int64),
        )
        return target

    def load_checkpoint(
        self, path: str | Path
    ) -> VulkanTesseractCoupledState:
        """Restore coupled fields and repeat projection/recovery audits."""
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            schema = str(checkpoint["schema"])
            if schema != "vulkan-tesseract.coupled-checkpoint.v1":
                raise ValueError(f"unsupported checkpoint schema {schema!r}")
            if tuple(checkpoint["shape"].tolist()) != self.shape:
                raise ValueError("checkpoint grid shape does not match backend")
            if str(checkpoint["constitutive_model"]) != self.constitutive_model:
                raise ValueError("checkpoint constitutive model does not match")
            if str(checkpoint["reconstruction"]) != self.reconstruction:
                raise ValueError("checkpoint reconstruction does not match")
            if (
                "boundary_mode" in checkpoint
                and str(checkpoint["boundary_mode"]) != self.boundary_mode
            ):
                raise ValueError("checkpoint boundary mode does not match")
            if (
                "derivative_order" in checkpoint
                and int(checkpoint["derivative_order"])
                != self.derivative_order
            ):
                raise ValueError(
                    "checkpoint derivative order does not match"
                )
            conserved = np.asarray(
                checkpoint["matter_state"], dtype=np.float32
            )
            primitive = np.asarray(
                checkpoint["primitive"], dtype=np.float32
            )
            vector_input = np.asarray(
                checkpoint["vector_input"], dtype=np.float32
            )
            ccz4 = np.asarray(
                checkpoint["ccz4_state"], dtype=np.float32
            )
            time = float(checkpoint["time"])
            steps = int(checkpoint["steps"])
        tracer = conserved[:, 10] / np.maximum(
            conserved[:, 0], np.float32(1.0e-12)
        )
        restored = self.initialize(
            primitive, vector_input, ccz4, tracer
        )
        uploaded = self.spy.Tensor.from_numpy(self.device, conserved)
        self._state_buffers[0] = uploaded
        self._current_state = 0
        recovered_index = (self._current_primitive + 1) % 4
        self._recover(
            uploaded,
            restored.matter.primitive,
            self._primitive_buffers[recovered_index],
        )
        self._current_primitive = recovered_index
        restored.matter.tensor = uploaded
        restored.matter.primitive = self._primitive_buffers[recovered_index]
        restored.matter.diagnostics = self._diagnostic_buffers[1]
        self._vector_stress_dispatch(restored.matter.primitive)
        restored.matter.vector_stress = self._vector_stress
        restored.matter.time = time
        restored.matter.steps = steps
        restored.time = time
        restored.steps = steps
        return restored


class VulkanTesseractEvolvedBackend(VulkanTesseractCoupledBackend):
    """Coupled Carter/CCZ4 engine with resident evolved mixed-Proca state."""

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        vector_cleaning_damping: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(shape, **kwargs)
        if vector_cleaning_damping < 0.0:
            raise ValueError("vector cleaning damping must be nonnegative")
        self.vector_cleaning_damping = float(vector_cleaning_damping)
        tensor = self.spy.Tensor.empty
        matrix = lambda width: tensor(
            self.device,
            shape=(self.points, width),
            dtype=self.module.float,
        )
        self._vector_dynamic_buffers = [
            matrix(MIXED_VECTOR_DYNAMIC_WIDTH) for _ in range(4)
        ]
        self._vector_dynamic_rhs = matrix(MIXED_VECTOR_DYNAMIC_WIDTH)
        self._vector_boundary_projected_rhs = matrix(
            MIXED_VECTOR_DYNAMIC_WIDTH
        )
        self._vector_boundary_final_rhs = matrix(
            MIXED_VECTOR_DYNAMIC_WIDTH
        )
        self._previous_vector_stress = matrix(VECTOR_STRESS_WIDTH)
        self._current_vector_dynamic = 0
        cmc_matrix = lambda width: tensor(
            self.device,
            shape=(self.points, width),
            dtype=self.module.float,
        )
        self._cmc_buffers = [cmc_matrix(4), cmc_matrix(4)]
        self._cmc_candidate = cmc_matrix(4)
        self._cmc_residual = cmc_matrix(2)
        self._cmc_control = tensor(
            self.device, shape=(1, 12), dtype=self.module.float
        )
        self._cmc_stored_source = matrix(4)
        self._counterflow_audit = matrix(3)
        self._scalar_grid = self.spy.grid(shape=(1,))
        self._current_cmc = 0
        # The constrained-data solve uses explicit raw-buffer fp64 kernels in
        # every active dimension.  SlangPy's ITensor<double> binder is not
        # reliable on all Windows Vulkan drivers, but raw fp64 storage buffers
        # are; the evolved CCZ4/matter ABI remains float32.
        self._cmc_fp64 = True
        self._cmc64_kernels: dict[str, Any] = {}
        self._cmc64_buffers: list[Any] = []
        self._cmc64_candidate: Any | None = None
        self._cmc64_residual: Any | None = None
        self._cmc64_control: Any | None = None
        if self._cmc_fp64:
            cmc_module_path = (
                Path(__file__).resolve().parents[1]
                / "tesseract_cmc_fp64.slang"
            )
            entry_points = {
                "trace": "theory33CMCTrace64Kernel",
                "update": "theory33CMCUpdate64Kernel",
                "mean": "theory33CMCMean64Kernel",
                "normalize": "theory33CMCNormalize64Kernel",
                "residual": "theory33CMCResidual64Kernel",
                "reduce": "theory33CMCReduce64Kernel",
                "pack": "theory33CMCPack64Kernel",
            }
            for key, entry_point in entry_points.items():
                program = self.device.load_program(
                    module_name=str(cmc_module_path),
                    entry_point_names=[entry_point],
                )
                self._cmc64_kernels[key] = (
                    self.device.create_compute_kernel(program=program)
                )
            usage = (
                self.spy.BufferUsage.shader_resource
                | self.spy.BufferUsage.unordered_access
            )
            self._cmc64_buffers = [
                self.device.create_buffer(
                    element_count=self.points * 4,
                    struct_size=8,
                    usage=usage,
                )
                for _ in range(2)
            ]
            self._cmc64_candidate = self.device.create_buffer(
                element_count=self.points * 4,
                struct_size=8,
                usage=usage,
            )
            self._cmc64_residual = self.device.create_buffer(
                element_count=self.points * 2,
                struct_size=8,
                usage=usage,
            )
            self._cmc64_control = self.device.create_buffer(
                element_count=12,
                struct_size=8,
                usage=usage,
            )

    def _refresh_evolved_vector_input(self, dynamic: Any) -> None:
        nx, ny, nz = self.shape
        self.module.theory33MixedVectorInputKernel(
            point=self.grid,
            nx=nx,
            ny=ny,
            nz=nz,
            inverseDx=float(1.0 / self.spacing[0]),
            inverseDy=float(1.0 / self.spacing[1]),
            inverseDz=float(1.0 / self.spacing[2]),
            dynamicState=dynamic,
            geometryInput=self._geometry,
            vectorInput=self._vector_input,
        )

    def initialize(
        self,
        primitive: np.ndarray,
        vector_dynamic: np.ndarray,
        ccz4_state: np.ndarray,
        tracer: np.ndarray,
    ) -> VulkanTesseractCoupledState:
        dynamic_values = VulkanMixedProcaBackend._array(
            vector_dynamic,
            (self.points, MIXED_VECTOR_DYNAMIC_WIDTH),
            "vector_dynamic",
        )
        vector_input = np.zeros(
            (self.points, VECTOR_INPUT_WIDTH), dtype=np.float32
        )
        vector_input[:, :14] = dynamic_values[:, :14]
        state = super().initialize(
            primitive, vector_input, ccz4_state, tracer
        )
        dynamic = self.spy.Tensor.from_numpy(
            self.device, dynamic_values
        )
        self._vector_dynamic_buffers[0] = dynamic
        self._current_vector_dynamic = 0
        self._refresh_evolved_vector_input(dynamic)
        self._vector_stress_dispatch(state.matter.primitive)
        output_index = (self._current_state + 1) % 5
        self.module.theory33FullExtractTracerKernel(
            point=self.grid,
            state=state.matter.tensor,
            tracer=self._specific_tracer,
        )
        self.module.theory33FullComposeCurvedKernel(
            point=self.grid,
            modelMode=self.model_mode,
            primitive=state.matter.primitive,
            tracer=self._specific_tracer,
            geometryInput=self._geometry,
            vectorStress=self._vector_stress,
            state=self._state_buffers[output_index],
        )
        self._current_state = output_index
        state.matter.tensor = self._state_buffers[output_index]
        return state

    def _balanced_counterflow_dispatch(
        self,
        state: VulkanTesseractCoupledState,
    ) -> None:
        """Generate the original active counterflow and refresh its stress."""
        output_index = (self._current_primitive + 1) % 4
        primitive = self._primitive_buffers[output_index]
        nx, ny, nz = self.shape
        self.module.theory33BalancedCounterflowKernel(
            point=self.grid,
            nx=nx,
            ny=ny,
            nz=nz,
            modelMode=self.model_mode,
            geometryInput=self._geometry,
            primitive=primitive,
            audit=self._counterflow_audit,
        )
        self._current_primitive = output_index
        state.matter.primitive = primitive
        dynamic = self._vector_dynamic_buffers[
            self._current_vector_dynamic
        ]
        self._refresh_evolved_vector_input(dynamic)
        self._vector_stress_dispatch(primitive)
        state_index = (self._current_state + 1) % 5
        self.module.theory33FullComposeCurvedKernel(
            point=self.grid,
            modelMode=self.model_mode,
            primitive=primitive,
            tracer=self._tracer,
            geometryInput=self._geometry,
            vectorStress=self._vector_stress,
            state=self._state_buffers[state_index],
        )
        self._current_state = state_index
        state.matter.tensor = self._state_buffers[state_index]
        recovered_index = (self._current_primitive + 1) % 4
        self._recover(
            state.matter.tensor,
            primitive,
            self._primitive_buffers[recovered_index],
        )
        self._current_primitive = recovered_index
        state.matter.primitive = self._primitive_buffers[recovered_index]
        state.matter.diagnostics = self._diagnostic_buffers[1]
        self._vector_stress_dispatch(state.matter.primitive)
        self.module.theory33FullMatterStressKernel(
            point=self.grid,
            modelMode=self.model_mode,
            primitive=state.matter.primitive,
            geometryInput=self._geometry,
            vectorStress=self._vector_stress,
            matterStress=self._matter_stress,
        )
        state.matter.vector_stress = self._vector_stress
        state.matter.geometry = self._geometry

    def _cmc_trace(self, expanding: bool) -> None:
        if self._cmc_fp64:
            assert self._cmc64_control is not None
            nx, ny, nz = self.shape
            self._cmc64_kernels["trace"].dispatch(
                thread_count=[1, 1, 1],
                numberPoints=self.points,
                nx=nx,
                ny=ny,
                nz=nz,
                inverseDx=float(1.0 / self.spacing[0]),
                inverseDy=float(1.0 / self.spacing[1]),
                inverseDz=float(1.0 / self.spacing[2]),
                kappaGravity=self.kappa_gravity,
                expanding=int(expanding),
                state=self._cmc64_buffers[self._current_cmc],
                matterStress=self._matter_stress.storage,
                control=self._cmc64_control,
            )
            return
        nx, ny, nz = self.shape
        state = self._cmc_buffers[self._current_cmc]
        self.module.theory33CMCTraceKernel(
            point=self._scalar_grid,
            numberPoints=self.points,
            nx=nx,
            ny=ny,
            nz=nz,
            inverseDx=float(1.0 / self.spacing[0]),
            inverseDy=float(1.0 / self.spacing[1]),
            inverseDz=float(1.0 / self.spacing[2]),
            kappaGravity=self.kappa_gravity,
            expanding=int(expanding),
            state=state,
            matterStress=self._matter_stress,
            control=self._cmc_control,
        )

    def _cmc_audit(self, expanding: bool) -> np.ndarray:
        if self._cmc_fp64:
            assert self._cmc64_control is not None
            assert self._cmc64_residual is not None
            nx, ny, nz = self.shape
            self._cmc_trace(expanding)
            self._cmc64_kernels["residual"].dispatch(
                thread_count=[self.points, 1, 1],
                numberPoints=self.points,
                nx=nx,
                ny=ny,
                nz=nz,
                inverseDx=float(1.0 / self.spacing[0]),
                inverseDy=float(1.0 / self.spacing[1]),
                inverseDz=float(1.0 / self.spacing[2]),
                kappaGravity=self.kappa_gravity,
                state=self._cmc64_buffers[self._current_cmc],
                matterStress=self._matter_stress.storage,
                control=self._cmc64_control,
                residual=self._cmc64_residual,
            )
            self._cmc64_kernels["reduce"].dispatch(
                thread_count=[1, 1, 1],
                numberPoints=self.points,
                state=self._cmc64_buffers[self._current_cmc],
                residual=self._cmc64_residual,
                control=self._cmc64_control,
            )
            return self._cmc64_control.to_numpy().view(np.float64)
        nx, ny, nz = self.shape
        self._cmc_trace(expanding)
        state = self._cmc_buffers[self._current_cmc]
        self.module.theory33CMCResidualKernel(
            point=self.grid,
            nx=nx,
            ny=ny,
            nz=nz,
            inverseDx=float(1.0 / self.spacing[0]),
            inverseDy=float(1.0 / self.spacing[1]),
            inverseDz=float(1.0 / self.spacing[2]),
            kappaGravity=self.kappa_gravity,
            state=state,
            matterStress=self._matter_stress,
            control=self._cmc_control,
            residual=self._cmc_residual,
        )
        self.module.theory33CMCReduceResidualKernel(
            point=self._scalar_grid,
            numberPoints=self.points,
            state=state,
            residual=self._cmc_residual,
            control=self._cmc_control,
        )
        return self._cmc_control.to_numpy()[0]

    def _solve_periodic_cmc(
        self,
        *,
        tolerance: float,
        maximum_iterations: int,
        relaxation: float,
        report_interval: int,
        expanding: bool,
    ) -> tuple[np.ndarray, int, int, bool]:
        nx, ny, nz = self.shape
        active_dimensions = self.ndim
        pseudo_dt = (
            relaxation
            * min(self.spacing[:active_dimensions]) ** 2
            / (2.0 * active_dimensions)
        )
        scalar_readbacks = 0
        converged = False
        control = np.full(12, np.nan)
        for iteration in range(1, maximum_iterations + 1):
            self._cmc_trace(expanding)
            if self._cmc_fp64:
                assert self._cmc64_candidate is not None
                assert self._cmc64_control is not None
                source64 = self._cmc64_buffers[self._current_cmc]
                target_index = 1 - self._current_cmc
                target64 = self._cmc64_buffers[target_index]
                self._cmc64_kernels["update"].dispatch(
                    thread_count=[self.points, 1, 1],
                    numberPoints=self.points,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    inverseDx=float(1.0 / self.spacing[0]),
                    inverseDy=float(1.0 / self.spacing[1]),
                    inverseDz=float(1.0 / self.spacing[2]),
                    kappaGravity=self.kappa_gravity,
                    pseudoTimeStep=pseudo_dt,
                    minimumConformalFactor=0.1,
                    maximumConformalFactor=20.0,
                    state=source64,
                    matterStress=self._matter_stress.storage,
                    control=self._cmc64_control,
                    candidate=self._cmc64_candidate,
                )
                self._cmc64_kernels["mean"].dispatch(
                    thread_count=[1, 1, 1],
                    numberPoints=self.points,
                    candidate=self._cmc64_candidate,
                    control=self._cmc64_control,
                )
                self._cmc64_kernels["normalize"].dispatch(
                    thread_count=[self.points, 1, 1],
                    numberPoints=self.points,
                    candidate=self._cmc64_candidate,
                    control=self._cmc64_control,
                    state=target64,
                )
                self._current_cmc = target_index
            else:
                source = self._cmc_buffers[self._current_cmc]
                target_index = 1 - self._current_cmc
                target = self._cmc_buffers[target_index]
                self.module.theory33CMCUpdateKernel(
                    point=self.grid,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    inverseDx=float(1.0 / self.spacing[0]),
                    inverseDy=float(1.0 / self.spacing[1]),
                    inverseDz=float(1.0 / self.spacing[2]),
                    kappaGravity=self.kappa_gravity,
                    pseudoTimeStep=pseudo_dt,
                    minimumConformalFactor=0.1,
                    maximumConformalFactor=20.0,
                    state=source,
                    matterStress=self._matter_stress,
                    control=self._cmc_control,
                    candidate=self._cmc_candidate,
                )
                self.module.theory33CMCMeanKernel(
                    point=self._scalar_grid,
                    numberPoints=self.points,
                    candidate=self._cmc_candidate,
                    control=self._cmc_control,
                )
                self.module.theory33CMCNormalizeKernel(
                    point=self.grid,
                    candidate=self._cmc_candidate,
                    control=self._cmc_control,
                    state=target,
                )
                self._current_cmc = target_index
            if iteration == 1 or iteration % report_interval == 0:
                control = self._cmc_audit(expanding)
                scalar_readbacks += 1
                if (
                    np.isfinite(control[6:10]).all()
                    and max(float(control[6]), float(control[7]))
                    <= tolerance
                ):
                    converged = True
                    break
                if (
                    not np.isfinite(control[8:10]).all()
                    or float(control[9]) > 20.0
                ):
                    break
        if not np.isfinite(control[6:10]).all():
            control = self._cmc_audit(expanding)
            scalar_readbacks += 1
        return control, iteration, scalar_readbacks, converged

    def _pack_cmc_geometry(
        self,
        state: VulkanTesseractCoupledState,
    ) -> None:
        nx, ny, nz = self.shape
        raw_index = (self._current_ccz4 + 1) % 8
        projected_index = (self._current_ccz4 + 2) % 8
        if self._cmc_fp64:
            assert self._cmc64_control is not None
            self._cmc64_kernels["pack"].dispatch(
                thread_count=[self.points, 1, 1],
                numberPoints=self.points,
                nx=nx,
                ny=ny,
                nz=nz,
                inverseDx=float(1.0 / self.spacing[0]),
                inverseDy=float(1.0 / self.spacing[1]),
                inverseDz=float(1.0 / self.spacing[2]),
                state=self._cmc64_buffers[self._current_cmc],
                control=self._cmc64_control,
                ccz4State=self._ccz4_buffers[raw_index].storage,
            )
        else:
            self.module.theory33CMCPackCCZ4Kernel(
                point=self.grid,
                nx=nx,
                ny=ny,
                nz=nz,
                inverseDx=float(1.0 / self.spacing[0]),
                inverseDy=float(1.0 / self.spacing[1]),
                inverseDz=float(1.0 / self.spacing[2]),
                state=self._cmc_buffers[self._current_cmc],
                control=self._cmc_control,
                ccz4State=self._ccz4_buffers[raw_index],
            )
        self._project_ccz4(
            self._ccz4_buffers[raw_index],
            self._ccz4_buffers[projected_index],
        )
        self._current_ccz4 = projected_index
        state.ccz4 = self._ccz4_buffers[projected_index]
        self._set_geometry(state.ccz4)
        state.matter.geometry = self._geometry

    def initialize_constrained(
        self,
        *,
        picard_iterations: int = 3,
        elliptic_tolerance: float = 2.0e-8,
        maximum_iterations: int = 9000,
        relaxation: float = 0.3,
        report_interval: int = 20,
        expanding: bool = False,
    ) -> tuple[VulkanTesseractCoupledState, VulkanPeriodicCMCReport]:
        """Construct the original active periodic CMC state on Vulkan.

        Field construction, the counterflow root, elliptic updates, global
        CMC zero mode, normalization, and CCZ4 packing all execute on the
        device.  Only the two convergence norms are read at report events.
        """
        if picard_iterations < 2:
            raise ValueError(
                "at least two matter/geometry Picard iterations are required"
            )
        if elliptic_tolerance <= 0.0 or maximum_iterations < 1:
            raise ValueError("invalid periodic CMC controls")
        primitive = np.zeros(
            (self.points, PRIMITIVE_3D_WIDTH), dtype=np.float32
        )
        dynamic = np.zeros(
            (self.points, MIXED_VECTOR_DYNAMIC_WIDTH), dtype=np.float32
        )
        ccz4 = np.zeros(
            (self.points, CCZ4_STATE_WIDTH), dtype=np.float32
        )
        for axis in range(3):
            ccz4[:, axis * 3 + axis] = 1.0
        ccz4[:, 18] = 1.0
        ccz4[:, 24] = 1.0
        tracer = np.zeros(self.points, dtype=np.float32)
        state = self.initialize(primitive, dynamic, ccz4, tracer)
        cmc_initial = np.zeros(
            (self.points, 4),
            dtype=np.float64 if self._cmc_fp64 else np.float32,
        )
        cmc_initial[:, 0] = 1.0
        if self._cmc_fp64:
            usage = (
                self.spy.BufferUsage.shader_resource
                | self.spy.BufferUsage.unordered_access
            )
            self._cmc64_buffers[0] = self.device.create_buffer(
                element_count=self.points * 4,
                struct_size=8,
                usage=usage,
                data=cmc_initial.reshape(-1),
            )
        else:
            self._cmc_buffers[0] = self.spy.Tensor.from_numpy(
                self.device, cmc_initial
            )
        self._current_cmc = 0
        scalar_readbacks = 0
        control = np.full(12, np.nan)
        latest_iterations = 0
        latest_converged = False
        source_change = np.inf
        momentum_balance = np.inf
        for picard in range(picard_iterations):
            self._balanced_counterflow_dispatch(state)
            self.module.theory33CounterflowAuditReduceKernel(
                point=self._scalar_grid,
                numberPoints=self.points,
                audit=self._counterflow_audit,
                control=self._cmc_control,
            )
            if picard > 0:
                self.module.theory33CMCSourceChangeKernel(
                    point=self._scalar_grid,
                    numberPoints=self.points,
                    matterStress=self._matter_stress,
                    storedSource=self._cmc_stored_source,
                    control=self._cmc_control,
                )
            self.module.theory33CMCStoreSourceKernel(
                point=self.grid,
                matterStress=self._matter_stress,
                storedSource=self._cmc_stored_source,
            )
            balance_control = self._cmc_control.to_numpy()[0]
            scalar_readbacks += 1
            momentum_balance = float(balance_control[11])
            if picard > 0:
                source_change = float(balance_control[10])
            control, latest_iterations, reads, latest_converged = (
                self._solve_periodic_cmc(
                    tolerance=elliptic_tolerance,
                    maximum_iterations=maximum_iterations,
                    relaxation=relaxation,
                    report_interval=report_interval,
                    expanding=expanding,
                )
            )
            scalar_readbacks += reads
            if not latest_converged:
                break
            self._pack_cmc_geometry(state)

        self._balanced_counterflow_dispatch(state)
        self.module.theory33CounterflowAuditReduceKernel(
            point=self._scalar_grid,
            numberPoints=self.points,
            audit=self._counterflow_audit,
            control=self._cmc_control,
        )
        balance_control = self._cmc_control.to_numpy()[0]
        scalar_readbacks += 1
        momentum_balance = float(balance_control[11])
        control = self._cmc_audit(expanding)
        scalar_readbacks += 1
        report = VulkanPeriodicCMCReport(
            converged=(
                latest_converged
                and float(control[6]) <= 1.1 * elliptic_tolerance
                and float(control[7]) <= 1.1 * elliptic_tolerance
            ),
            iterations=latest_iterations,
            picard_iterations=picard_iterations,
            hamiltonian_residual_l2=float(control[6]),
            momentum_residual_l2=float(control[7]),
            trace_K=float(control[1]),
            minimum_conformal_factor=float(control[8]),
            maximum_conformal_factor=float(control[9]),
            pointwise_momentum_balance=momentum_balance,
            final_source_relative_change=float(source_change),
            scalar_readbacks=scalar_readbacks,
        )
        return state, report

    def _vector_rhs_dispatch(
        self,
        dynamic: Any,
        matter: VulkanTheory33MultidimensionalState,
    ) -> None:
        nx, ny, nz = self.shape
        self.module.theory33MixedVectorRhsKernel(
            point=self.grid,
            nx=nx,
            ny=ny,
            nz=nz,
            inverseDx=float(1.0 / self.spacing[0]),
            inverseDy=float(1.0 / self.spacing[1]),
            inverseDz=float(1.0 / self.spacing[2]),
            cleaningDamping=self.vector_cleaning_damping,
            dynamicState=dynamic,
            matterState=matter.tensor,
            primitive=matter.primitive,
            geometryInput=self._geometry,
            rhs=self._vector_dynamic_rhs,
        )
        if self.boundary_mode_id == 1:
            self.module.theory33VectorRadiativeBoundaryKernel(
                point=self.grid,
                nx=nx,
                ny=ny,
                nz=nz,
                inverseDx=float(1.0 / self.spacing[0]),
                inverseDy=float(1.0 / self.spacing[1]),
                inverseDz=float(1.0 / self.spacing[2]),
                originX=self.origins[0],
                originY=self.origins[1],
                originZ=self.origins[2],
                derivativeOrder=self.derivative_order,
                state=dynamic,
                rawRhs=self._vector_dynamic_rhs,
                projectedRhs=self._vector_boundary_projected_rhs,
            )
            self.module.theory33VectorGaussBoundaryKernel(
                point=self.grid,
                nx=nx,
                ny=ny,
                nz=nz,
                inverseDx=float(1.0 / self.spacing[0]),
                inverseDy=float(1.0 / self.spacing[1]),
                inverseDz=float(1.0 / self.spacing[2]),
                derivativeOrder=self.derivative_order,
                ZA=1.0,
                ZB=1.0,
                rawRhs=self._vector_dynamic_rhs,
                projectedRhs=self._vector_boundary_projected_rhs,
                finalRhs=self._vector_boundary_final_rhs,
            )

    def _vector_stage(
        self,
        base: Any,
        stage: Any,
        output: Any,
        dt: float,
        base_weight: float,
        evolved_weight: float,
    ) -> None:
        self.module.theory33MixedVectorStageKernel(
            point=self.grid,
            dt=float(dt),
            baseWeight=float(base_weight),
            evolvedWeight=float(evolved_weight),
            baseState=base,
            stageState=stage,
            rhs=(
                self._vector_boundary_final_rhs
                if self.boundary_mode_id == 1
                else self._vector_dynamic_rhs
            ),
            output=output,
        )

    def _advance_vectors(
        self,
        state: VulkanTesseractCoupledState,
        dt: float,
    ) -> None:
        base_index = self._current_vector_dynamic
        first_index = (base_index + 1) % 4
        second_index = (base_index + 2) % 4
        final_index = (base_index + 3) % 4
        base = self._vector_dynamic_buffers[base_index]
        first = self._vector_dynamic_buffers[first_index]
        second = self._vector_dynamic_buffers[second_index]
        final = self._vector_dynamic_buffers[final_index]
        self.module.theory33CopyVectorStressKernel(
            point=self.grid,
            source=self._vector_stress,
            target=self._previous_vector_stress,
        )
        self._vector_rhs_dispatch(base, state.matter)
        self._vector_stage(base, base, first, dt, 0.0, 1.0)
        self._vector_rhs_dispatch(first, state.matter)
        self._vector_stage(base, first, second, dt, 0.75, 0.25)
        self._vector_rhs_dispatch(second, state.matter)
        self._vector_stage(
            base, second, final, dt, 1.0 / 3.0, 2.0 / 3.0
        )
        self._current_vector_dynamic = final_index
        self._refresh_evolved_vector_input(final)
        self._vector_stress_dispatch(state.matter.primitive)
        output_index = (self._current_state + 1) % 5
        self.module.theory33FullUpdateVectorStressKernel(
            point=self.grid,
            state=state.matter.tensor,
            geometryInput=self._geometry,
            previousVectorStress=self._previous_vector_stress,
            currentVectorStress=self._vector_stress,
            output=self._state_buffers[output_index],
        )
        self._current_state = output_index
        state.matter.tensor = self._state_buffers[output_index]
        state.matter.vector_stress = self._vector_stress

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
        self._advance_vectors(state, 0.5 * dt)
        VulkanTheory33MultidimensionalBackend.step(
            self, state.matter, dt, apply_drag=apply_drag
        )
        self._advance_vectors(state, 0.5 * dt)
        self._advance_ccz4(state, 0.5 * dt)
        state.time += float(dt)
        state.steps += 1
        return state

    def snapshot(
        self,
        state: VulkanTesseractCoupledState,
    ) -> VulkanTesseractEvolvedSnapshot:
        coupled = super().snapshot(state)
        dynamic = self._vector_dynamic_buffers[
            self._current_vector_dynamic
        ]
        self._refresh_evolved_vector_input(dynamic)
        return VulkanTesseractEvolvedSnapshot(
            coupled=coupled,
            vector_state=dynamic.to_numpy(),
            vector_input=self._vector_input.to_numpy(),
        )

    def save_checkpoint(
        self,
        state: VulkanTesseractCoupledState,
        path: str | Path,
    ) -> Path:
        """Write every evolved Carter, Proca, and CCZ4 resident field."""
        snapshot = self.snapshot(state)
        coupled = snapshot.coupled
        target = Path(path)
        np.savez_compressed(
            target,
            schema=np.array(
                "vulkan-tesseract.evolved-checkpoint.v1"
            ),
            shape=np.asarray(self.shape, dtype=np.int64),
            lengths=np.asarray(self.lengths, dtype=np.float64),
            origins=np.asarray(self.origins, dtype=np.float64),
            constitutive_model=np.array(self.constitutive_model),
            reconstruction=np.array(self.reconstruction),
            boundary_mode=np.array(self.boundary_mode),
            derivative_order=np.array(
                self.derivative_order, dtype=np.int64
            ),
            relaxation_time=np.array(self.relaxation_time),
            vector_cleaning_damping=np.array(
                self.vector_cleaning_damping
            ),
            matter_state=coupled.matter.state,
            primitive=coupled.matter.primitive,
            vector_state=snapshot.vector_state,
            ccz4_state=coupled.ccz4_state,
            time=np.array(coupled.time),
            steps=np.array(coupled.steps, dtype=np.int64),
        )
        return target

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> VulkanTesseractCoupledState:
        """Restore all evolved fields and repeat projection/recovery audits."""
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            schema = str(checkpoint["schema"])
            if schema != "vulkan-tesseract.evolved-checkpoint.v1":
                raise ValueError(f"unsupported checkpoint schema {schema!r}")
            if tuple(checkpoint["shape"].tolist()) != self.shape:
                raise ValueError("checkpoint grid shape does not match backend")
            if str(checkpoint["constitutive_model"]) != self.constitutive_model:
                raise ValueError("checkpoint constitutive model does not match")
            if str(checkpoint["reconstruction"]) != self.reconstruction:
                raise ValueError("checkpoint reconstruction does not match")
            if str(checkpoint["boundary_mode"]) != self.boundary_mode:
                raise ValueError("checkpoint boundary mode does not match")
            if int(checkpoint["derivative_order"]) != self.derivative_order:
                raise ValueError(
                    "checkpoint derivative order does not match"
                )
            conserved = np.asarray(
                checkpoint["matter_state"], dtype=np.float32
            )
            primitive = np.asarray(
                checkpoint["primitive"], dtype=np.float32
            )
            vector_state = np.asarray(
                checkpoint["vector_state"], dtype=np.float32
            )
            ccz4 = np.asarray(
                checkpoint["ccz4_state"], dtype=np.float32
            )
            time = float(checkpoint["time"])
            steps = int(checkpoint["steps"])
        tracer = conserved[:, 10] / np.maximum(
            conserved[:, 0], np.float32(1.0e-12)
        )
        restored = self.initialize(
            primitive, vector_state, ccz4, tracer
        )
        uploaded = self.spy.Tensor.from_numpy(self.device, conserved)
        state_index = (self._current_state + 1) % 5
        self._state_buffers[state_index] = uploaded
        self._current_state = state_index
        recovered_index = (self._current_primitive + 1) % 4
        self._recover(
            uploaded,
            restored.matter.primitive,
            self._primitive_buffers[recovered_index],
        )
        self._current_primitive = recovered_index
        restored.matter.tensor = uploaded
        restored.matter.primitive = self._primitive_buffers[
            recovered_index
        ]
        restored.matter.diagnostics = self._diagnostic_buffers[1]
        self._refresh_evolved_vector_input(
            self._vector_dynamic_buffers[
                self._current_vector_dynamic
            ]
        )
        self._vector_stress_dispatch(restored.matter.primitive)
        restored.matter.vector_stress = self._vector_stress
        restored.matter.geometry = self._geometry
        restored.matter.time = time
        restored.matter.steps = steps
        restored.time = time
        restored.steps = steps
        return restored


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
        derivative_order: int = 2,
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
        if derivative_order not in {2, 4}:
            raise ValueError("derivative_order must be 2 or 4")
        if derivative_order == 4 and any(
            value < 8 for value in expanded[: len(shape)]
        ):
            raise ValueError(
                "fourth-order CCZ4 derivatives need eight cells per axis"
            )
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
            derivativeOrder=int(derivative_order),
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
