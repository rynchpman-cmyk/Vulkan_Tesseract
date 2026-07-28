"""Resident Vulkan physical-boundary operators.

The kernels in this module mirror :mod:`tesseract_nr.boundary` without
round-tripping field data through Python.  Arrays use the solver's x-fastest
``(point, component)`` ABI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .vulkan_backend import CCZ4_STATE_WIDTH, MIXED_VECTOR_DYNAMIC_WIDTH


class VulkanBoundaryBackend:
    """Dispatch CCZ4, mixed-Proca, and linear characteristic boundaries."""

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        lengths: tuple[float, ...] | None = None,
        origins: tuple[float, ...] | None = None,
        derivative_order: int = 2,
        module_path: str | Path | None = None,
    ) -> None:
        if not 1 <= len(shape) <= 3:
            raise ValueError("shape must contain one to three dimensions")
        if any(int(value) < 3 for value in shape):
            raise ValueError("boundary stencils need three cells per axis")
        if derivative_order not in {2, 4}:
            raise ValueError("derivative_order must be 2 or 4")
        if derivative_order == 4 and any(
            int(value) < 6 for value in shape
        ):
            raise ValueError(
                "fourth-order boundary stencils need six cells per axis"
            )
        self.derivative_order = int(derivative_order)
        self.ndim = len(shape)
        self.shape = tuple(int(value) for value in shape) + (1,) * (
            3 - self.ndim
        )
        self.points = int(np.prod(self.shape))
        if lengths is None:
            lengths = (1.0,) * self.ndim
        if len(lengths) != self.ndim or any(value <= 0.0 for value in lengths):
            raise ValueError("lengths must match active positive dimensions")
        self.lengths = tuple(float(value) for value in lengths) + (1.0,) * (
            3 - self.ndim
        )
        if origins is None:
            origins = tuple(-0.5 * value for value in lengths)
        if len(origins) != self.ndim:
            raise ValueError("origins must match active dimensions")
        self.origins = tuple(float(value) for value in origins) + (0.0,) * (
            3 - self.ndim
        )
        self.spacing = tuple(
            self.lengths[axis] / self.shape[axis] for axis in range(3)
        )
        try:
            import slangpy as spy
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "VulkanBoundaryBackend requires the slangpy environment"
            ) from exc
        self.spy = spy
        root = Path(__file__).resolve().parents[1]
        self.module_path = (
            root / "tesseract_nr_gpu.slang"
            if module_path is None
            else Path(module_path).resolve()
        )
        self.device = spy.create_device(
            type=spy.DeviceType.vulkan,
            include_paths=[self.module_path.parent],
        )
        self.module = spy.Module.load_from_file(
            self.device, str(self.module_path)
        )
        self.grid = spy.grid(shape=(self.points,))

    def _array(self, values: np.ndarray, width: int, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (self.points, width):
            raise ValueError(
                f"{name} must have shape {(self.points, width)}, "
                f"got {result.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite")
        return np.ascontiguousarray(result)

    def _empty(self, width: int) -> Any:
        return self.spy.Tensor.empty(
            self.device,
            shape=(self.points, width),
            dtype=self.module.float,
        )

    def linear_flux(
        self,
        matrix: np.ndarray,
        state: np.ndarray,
        outward_sign: int,
    ) -> np.ndarray:
        """Remove incoming modes from a real two-field boundary flux."""
        if outward_sign not in {-1, 1}:
            raise ValueError("outward_sign must be -1 or +1")
        matrices = np.asarray(matrix, dtype=np.float32)
        if matrices.shape == (self.points, 2, 2):
            matrices = matrices.reshape(self.points, 4)
        matrices = self._array(matrices, 4, "matrix")
        states = self._array(state, 2, "state")
        output = self._empty(2)
        self.module.theory33LinearBoundaryFlux2Kernel(
            point=self.grid,
            outwardSign=outward_sign,
            matrix=self.spy.Tensor.from_numpy(self.device, matrices),
            state=self.spy.Tensor.from_numpy(self.device, states),
            flux=output,
        )
        return output.to_numpy()

    def ccz4_rhs(
        self,
        state: np.ndarray,
        rhs: np.ndarray,
        *,
        constraint_damping: float = 1.0,
    ) -> np.ndarray:
        """Apply outgoing CCZ4 radiation and incoming-Z4 projection."""
        if constraint_damping < 0.0:
            raise ValueError("constraint_damping must be nonnegative")
        state_values = self._array(state, CCZ4_STATE_WIDTH, "state")
        rhs_values = self._array(rhs, CCZ4_STATE_WIDTH, "rhs")
        state_tensor = self.spy.Tensor.from_numpy(self.device, state_values)
        rhs_tensor = self.spy.Tensor.from_numpy(self.device, rhs_values)
        nx, ny, nz = self.shape
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
            constraintDamping=float(constraint_damping),
            state=state_tensor,
            rhs=rhs_tensor,
        )
        return rhs_tensor.to_numpy()

    def complete_ccz4_gamma_hat(self, state: np.ndarray) -> np.ndarray:
        """Set GammaHat to the contracted conformal connection (Z_i=0)."""
        state_values = self._array(state, CCZ4_STATE_WIDTH, "state")
        state_tensor = self.spy.Tensor.from_numpy(self.device, state_values)
        output = self._empty(CCZ4_STATE_WIDTH)
        nx, ny, nz = self.shape
        self.module.theory33CCZ4GammaHatFromMetricKernel(
            point=self.grid,
            nx=nx,
            ny=ny,
            nz=nz,
            inverseDx=float(1.0 / self.spacing[0]),
            inverseDy=float(1.0 / self.spacing[1]),
            inverseDz=float(1.0 / self.spacing[2]),
            derivativeOrder=self.derivative_order,
            state=state_tensor,
            output=output,
        )
        return output.to_numpy()

    def mixed_vector_rhs(
        self,
        state: np.ndarray,
        raw_rhs: np.ndarray,
        *,
        Z_A: float = 1.0,
        Z_B: float = 1.0,
    ) -> np.ndarray:
        """Apply radiative Proca data with semidiscrete Gauss compatibility."""
        if Z_A <= 0.0 or Z_B <= 0.0:
            raise ValueError("Z_A and Z_B must be positive")
        state_values = self._array(
            state, MIXED_VECTOR_DYNAMIC_WIDTH, "state"
        )
        raw_values = self._array(
            raw_rhs, MIXED_VECTOR_DYNAMIC_WIDTH, "raw_rhs"
        )
        state_tensor = self.spy.Tensor.from_numpy(self.device, state_values)
        raw_tensor = self.spy.Tensor.from_numpy(self.device, raw_values)
        projected = self._empty(MIXED_VECTOR_DYNAMIC_WIDTH)
        final = self._empty(MIXED_VECTOR_DYNAMIC_WIDTH)
        nx, ny, nz = self.shape
        arguments = {
            "point": self.grid,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "inverseDx": float(1.0 / self.spacing[0]),
            "inverseDy": float(1.0 / self.spacing[1]),
            "inverseDz": float(1.0 / self.spacing[2]),
        }
        self.module.theory33VectorRadiativeBoundaryKernel(
            **arguments,
            originX=self.origins[0],
            originY=self.origins[1],
            originZ=self.origins[2],
            derivativeOrder=self.derivative_order,
            state=state_tensor,
            rawRhs=raw_tensor,
            projectedRhs=projected,
        )
        self.module.theory33VectorGaussBoundaryKernel(
            **arguments,
            derivativeOrder=self.derivative_order,
            ZA=float(Z_A),
            ZB=float(Z_B),
            rawRhs=raw_tensor,
            projectedRhs=projected,
            finalRhs=final,
        )
        return final.to_numpy()
