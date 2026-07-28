"""GPU-resident vacuum CCZ4 evolution for puncture and horizon campaigns.

The production Carter backend already contains the same CCZ4 kernels, but a
vacuum puncture collision should not allocate or advance dummy matter fields.
This module therefore supplies a narrow resident control plane around the
projected SSP-RK3 CCZ4 operator.  Evolution stays on Vulkan; host transfer is
explicitly limited to snapshots and checkpoints requested by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .vulkan_backend import (
    BOUNDARY_MODES,
    CCZ4_PROJECTION_AUDIT_WIDTH,
    CCZ4_STATE_WIDTH,
    _slangpy,
)


@dataclass
class VulkanCCZ4VacuumState:
    """Opaque handle to one resident projected CCZ4 state."""

    tensor: Any
    time: float = 0.0
    steps: int = 0
    host_reads: int = 0


@dataclass(frozen=True)
class VulkanCCZ4VacuumEvolutionSnapshot:
    """Explicit host observation of a resident vacuum evolution."""

    ccz4_state: np.ndarray
    projection_audit: np.ndarray
    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]
    origins: tuple[float, float, float]
    derivative_order: int
    time: float
    steps: int


class VulkanCCZ4VacuumEvolutionBackend:
    """Projected FD2/FD4 vacuum CCZ4 evolution with physical boundaries.

    Seven state buffers hold the base and projected SSP-RK3 intermediates.
    Together with one RHS and one projection-audit buffer this is the complete
    resident allocation.  In particular, no Carter primitive, flux, recovery,
    or mixed-vector storage is created for a vacuum puncture run.
    """

    checkpoint_schema = "vulkan-tesseract.ccz4-vacuum-checkpoint.v1"

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        lengths: tuple[float, ...] | None = None,
        origins: tuple[float, ...] | None = None,
        boundary_mode: str = "radiative_outflow",
        kappa1: float = 0.1,
        kappa2: float = 0.0,
        kappa3: float = 1.0,
        cosmological_constant: float = 0.0,
        shift_speed: float = 1.0,
        eta_shift: float = 1.0,
        slicing: str = "1+log",
        evolve_shift: bool = True,
        derivative_order: int = 4,
        module_path: str | Path | None = None,
    ) -> None:
        if not 1 <= len(shape) <= 3:
            raise ValueError("shape must contain one to three dimensions")
        active_shape = tuple(int(value) for value in shape)
        if any(value < 4 for value in active_shape):
            raise ValueError("each active dimension needs at least four cells")
        self.ndim = len(active_shape)
        self.shape = active_shape + (1,) * (3 - self.ndim)
        self.points = int(np.prod(self.shape))
        active_lengths = (
            (1.0,) * self.ndim
            if lengths is None
            else tuple(float(value) for value in lengths)
        )
        if len(active_lengths) != self.ndim or any(
            value <= 0.0 for value in active_lengths
        ):
            raise ValueError("lengths must match active positive dimensions")
        self.lengths = active_lengths + (1.0,) * (3 - self.ndim)
        active_origins = (
            tuple(-0.5 * value for value in active_lengths)
            if origins is None
            else tuple(float(value) for value in origins)
        )
        if len(active_origins) != self.ndim or not np.all(
            np.isfinite(active_origins)
        ):
            raise ValueError("origins must match the active dimensions")
        self.origins = active_origins + (0.0,) * (3 - self.ndim)
        self.spacing = tuple(
            self.lengths[axis] / self.shape[axis] for axis in range(3)
        )
        if boundary_mode not in BOUNDARY_MODES:
            raise ValueError("unknown boundary mode")
        if boundary_mode == "radiative_outflow" and any(
            value < 6 for value in active_shape
        ):
            raise ValueError(
                "radiative_outflow needs six cells per active axis"
            )
        if derivative_order not in {2, 4}:
            raise ValueError("derivative_order must be 2 or 4")
        if derivative_order == 4 and any(value < 8 for value in active_shape):
            raise ValueError(
                "fourth-order CCZ4 derivatives need eight cells per axis"
            )
        if kappa1 < 0.0 or kappa2 <= -1.0:
            raise ValueError("invalid CCZ4 damping parameters")
        if shift_speed <= 0.0 or eta_shift < 0.0:
            raise ValueError("invalid shift parameters")
        if slicing not in {"1+log", "harmonic"}:
            raise ValueError("slicing must be '1+log' or 'harmonic'")

        self.boundary_mode = boundary_mode
        self.boundary_mode_id = BOUNDARY_MODES[boundary_mode]
        self.derivative_order = int(derivative_order)
        self.kappa1 = float(kappa1)
        self.kappa2 = float(kappa2)
        self.kappa3 = float(kappa3)
        self.cosmological_constant = float(cosmological_constant)
        self.shift_speed = float(shift_speed)
        self.eta_shift = float(eta_shift)
        self.slicing = slicing
        self.slicing_mode = 0 if slicing == "1+log" else 1
        self.evolve_shift = bool(evolve_shift)

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
        matrix = lambda width: self.spy.Tensor.empty(
            self.device,
            shape=(self.points, width),
            dtype=self.module.float,
        )
        self._buffers = [matrix(CCZ4_STATE_WIDTH) for _ in range(7)]
        self._rhs = matrix(CCZ4_STATE_WIDTH)
        self._projection_audit = matrix(CCZ4_PROJECTION_AUDIT_WIDTH)
        self._current = 0

    @property
    def resident_storage_bytes(self) -> int:
        """Explicit state/RHS/audit storage, excluding runtime overhead."""

        scalar_count = (
            7 * CCZ4_STATE_WIDTH
            + CCZ4_STATE_WIDTH
            + CCZ4_PROJECTION_AUDIT_WIDTH
        )
        return self.points * scalar_count * np.dtype(np.float32).itemsize

    def _host_state(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (self.points, CCZ4_STATE_WIDTH):
            raise ValueError(
                "ccz4_state must have shape "
                f"{(self.points, CCZ4_STATE_WIDTH)}, got {result.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("ccz4_state must be finite")
        return np.ascontiguousarray(result)

    def _project(self, source: Any, target: Any) -> None:
        self.module.ccz4ProjectGeometryKernel(
            point=self.grid,
            stateInput=source,
            stateOutput=target,
            audit=self._projection_audit,
        )

    def _evaluate_rhs(self, state: Any) -> None:
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
            evolveShift=int(self.evolve_shift),
            state=state,
            rhs=self._rhs,
        )
        if self.boundary_mode_id == BOUNDARY_MODES["radiative_outflow"]:
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
                state=state,
                rhs=self._rhs,
            )

    def _stage(
        self,
        base: Any,
        evolved: Any,
        output: Any,
        dt: float,
        base_weight: float,
        evolved_weight: float,
    ) -> None:
        self.module.ccz4StageKernel(
            point=self.grid,
            dt=float(dt),
            baseWeight=float(base_weight),
            evolvedWeight=float(evolved_weight),
            baseState=base,
            stageState=evolved,
            rhs=self._rhs,
            output=output,
        )

    def initialize(self, ccz4_state: np.ndarray) -> VulkanCCZ4VacuumState:
        """Upload and project one packed x-fastest CCZ4 state."""

        uploaded = self.spy.Tensor.from_numpy(
            self.device, self._host_state(ccz4_state)
        )
        self._buffers[0] = uploaded
        self._project(uploaded, self._buffers[1])
        self._current = 1
        return VulkanCCZ4VacuumState(self._buffers[self._current])

    def step(
        self, state: VulkanCCZ4VacuumState, dt: float
    ) -> VulkanCCZ4VacuumState:
        """Advance one projected SSP-RK3 step without host readback."""

        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be positive and finite")
        base_index = self._current
        indices = [(base_index + offset) % 7 for offset in range(1, 7)]
        base = self._buffers[base_index]

        self._evaluate_rhs(base)
        self._stage(base, base, self._buffers[indices[0]], dt, 0.0, 1.0)
        self._project(self._buffers[indices[0]], self._buffers[indices[1]])

        self._evaluate_rhs(self._buffers[indices[1]])
        self._stage(
            base,
            self._buffers[indices[1]],
            self._buffers[indices[2]],
            dt,
            0.75,
            0.25,
        )
        self._project(self._buffers[indices[2]], self._buffers[indices[3]])

        self._evaluate_rhs(self._buffers[indices[3]])
        self._stage(
            base,
            self._buffers[indices[3]],
            self._buffers[indices[4]],
            dt,
            1.0 / 3.0,
            2.0 / 3.0,
        )
        self._project(self._buffers[indices[4]], self._buffers[indices[5]])

        self._current = indices[5]
        state.tensor = self._buffers[self._current]
        state.time += float(dt)
        state.steps += 1
        return state

    def snapshot(
        self, state: VulkanCCZ4VacuumState
    ) -> VulkanCCZ4VacuumEvolutionSnapshot:
        """Perform the explicit diagnostic readback used by MOTS observers."""

        state.host_reads += 1
        return VulkanCCZ4VacuumEvolutionSnapshot(
            ccz4_state=state.tensor.to_numpy(),
            projection_audit=self._projection_audit.to_numpy(),
            shape=self.shape,
            lengths=self.lengths,
            origins=self.origins,
            derivative_order=self.derivative_order,
            time=state.time,
            steps=state.steps,
        )

    def save_checkpoint(
        self, state: VulkanCCZ4VacuumState, path: str | Path
    ) -> Path:
        """Save a restartable resident vacuum state after one explicit readback."""

        snapshot = self.snapshot(state)
        target = Path(path)
        np.savez_compressed(
            target,
            schema=np.array(self.checkpoint_schema),
            shape=np.asarray(self.shape, dtype=np.int64),
            lengths=np.asarray(self.lengths, dtype=np.float64),
            origins=np.asarray(self.origins, dtype=np.float64),
            boundary_mode=np.array(self.boundary_mode),
            derivative_order=np.array(self.derivative_order, dtype=np.int64),
            kappa1=np.array(self.kappa1),
            kappa2=np.array(self.kappa2),
            kappa3=np.array(self.kappa3),
            cosmological_constant=np.array(self.cosmological_constant),
            shift_speed=np.array(self.shift_speed),
            eta_shift=np.array(self.eta_shift),
            slicing=np.array(self.slicing),
            evolve_shift=np.array(self.evolve_shift),
            ccz4_state=snapshot.ccz4_state,
            time=np.array(snapshot.time),
            steps=np.array(snapshot.steps, dtype=np.int64),
        )
        return target

    def load_checkpoint(
        self, path: str | Path
    ) -> VulkanCCZ4VacuumState:
        """Restore a checkpoint only when its complete numerical ABI matches."""

        with np.load(Path(path), allow_pickle=False) as checkpoint:
            if str(checkpoint["schema"]) != self.checkpoint_schema:
                raise ValueError("unsupported vacuum CCZ4 checkpoint schema")
            exact = {
                "shape": tuple(checkpoint["shape"].tolist()) == self.shape,
                "lengths": np.array_equal(
                    checkpoint["lengths"], np.asarray(self.lengths)
                ),
                "origins": np.array_equal(
                    checkpoint["origins"], np.asarray(self.origins)
                ),
                "boundary_mode": (
                    str(checkpoint["boundary_mode"]) == self.boundary_mode
                ),
                "derivative_order": (
                    int(checkpoint["derivative_order"])
                    == self.derivative_order
                ),
                "slicing": str(checkpoint["slicing"]) == self.slicing,
                "evolve_shift": (
                    bool(checkpoint["evolve_shift"]) == self.evolve_shift
                ),
            }
            scalar_parameters = (
                "kappa1",
                "kappa2",
                "kappa3",
                "cosmological_constant",
                "shift_speed",
                "eta_shift",
            )
            for name in scalar_parameters:
                exact[name] = float(checkpoint[name]) == getattr(self, name)
            mismatch = [name for name, matched in exact.items() if not matched]
            if mismatch:
                raise ValueError(
                    "checkpoint configuration mismatch: "
                    + ", ".join(mismatch)
                )
            packed = np.asarray(checkpoint["ccz4_state"], dtype=np.float32)
            time = float(checkpoint["time"])
            steps = int(checkpoint["steps"])
        state = self.initialize(packed)
        state.time = time
        state.steps = steps
        return state
