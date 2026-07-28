"""FP64 Vulkan isolated conformal transverse-traceless initial data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .initial_constraints import CTTParameters, CTTReport


@dataclass(frozen=True)
class VulkanCTTResult:
    conformal_factor: np.ndarray
    vector_potential: np.ndarray
    ccz4_state: np.ndarray
    report: CTTReport
    scalar_readbacks: int


class VulkanCTTBackend:
    """Solve the isolated CTT equations with resident fp64 field buffers."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        lengths: tuple[float, float, float],
        *,
        module_path: str | Path | None = None,
    ) -> None:
        if len(shape) != 3 or len(lengths) != 3:
            raise ValueError("the isolated CTT solve requires a 3D patch")
        if any(int(value) < 6 for value in shape) or any(
            value <= 0.0 for value in lengths
        ):
            raise ValueError("CTT axes need six cells and positive lengths")
        self.shape = tuple(int(value) for value in shape)
        self.lengths = tuple(float(value) for value in lengths)
        self.spacing = tuple(
            self.lengths[axis] / self.shape[axis] for axis in range(3)
        )
        self.points = int(np.prod(self.shape))
        try:
            import slangpy as spy
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "VulkanCTTBackend requires the slangpy environment"
            ) from exc
        self.spy = spy
        self.device = spy.create_device(type=spy.DeviceType.vulkan)
        root = Path(__file__).resolve().parents[1]
        self.module_path = (
            root / "tesseract_ctt_fp64.slang"
            if module_path is None
            else Path(module_path).resolve()
        )
        entries = {
            "initialize": "theory33CTTInitialize64Kernel",
            "derivatives": "theory33CTTDerivatives64Kernel",
            "residual": "theory33CTTResidual64Kernel",
            "update": "theory33CTTUpdate64Kernel",
            "reduce": "theory33CTTReduce64Kernel",
            "pack": "theory33CTTPackCCZ464Kernel",
        }
        self.kernels = {}
        for name, entry in entries.items():
            program = self.device.load_program(
                module_name=str(self.module_path),
                entry_point_names=[entry],
            )
            self.kernels[name] = self.device.create_compute_kernel(
                program=program
            )
        self.usage = (
            spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access
        )

    def _buffer(
        self,
        count: int,
        *,
        struct_size: int,
        data: np.ndarray | None = None,
    ):
        arguments = {
            "element_count": count,
            "struct_size": struct_size,
            "usage": self.usage,
        }
        if data is not None:
            arguments["data"] = np.ascontiguousarray(data).reshape(-1)
        return self.device.create_buffer(**arguments)

    def _operators(self, state, source, derivatives, residual) -> None:
        nx, ny, nz = self.shape
        common = {
            "numberPoints": self.points,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "inverseDxInput": float(1.0 / self.spacing[0]),
            "inverseDyInput": float(1.0 / self.spacing[1]),
            "inverseDzInput": float(1.0 / self.spacing[2]),
        }
        self.kernels["derivatives"].dispatch(
            thread_count=[self.points, 1, 1],
            **common,
            state=state,
            derivatives=derivatives,
        )
        self.kernels["residual"].dispatch(
            thread_count=[self.points, 1, 1],
            **common,
            kappaInput=self._kappa,
            state=state,
            source=source,
            derivatives=derivatives,
            residual=residual,
        )

    def solve(
        self,
        rho: np.ndarray,
        momentum_lower: np.ndarray,
        parameters: CTTParameters | None = None,
    ) -> VulkanCTTResult:
        parameters = parameters or CTTParameters()
        rho_values = np.asarray(rho, dtype=np.float64)
        momentum_values = np.asarray(momentum_lower, dtype=np.float64)
        if rho_values.shape != self.shape:
            raise ValueError("rho shape does not match the CTT patch")
        if momentum_values.shape != (3,) + self.shape:
            raise ValueError("momentum shape does not match the CTT patch")
        if np.any(rho_values < 0.0) or not np.all(
            np.isfinite(rho_values)
        ):
            raise ValueError("rho must be finite and nonnegative")
        if not np.all(np.isfinite(momentum_values)):
            raise ValueError("momentum must be finite")
        source_values = np.empty((self.points, 4), dtype=np.float64)
        source_values[:, 0] = rho_values.reshape(-1, order="F")
        for component in range(3):
            source_values[:, 1 + component] = momentum_values[
                component
            ].reshape(-1, order="F")
        state = [
            self._buffer(self.points * 4, struct_size=8),
            self._buffer(self.points * 4, struct_size=8),
        ]
        source = self._buffer(
            self.points * 4,
            struct_size=8,
            data=source_values,
        )
        derivatives = self._buffer(self.points * 13, struct_size=8)
        residual = self._buffer(self.points * 4, struct_size=8)
        control = self._buffer(4, struct_size=8)
        ccz4 = self._buffer(self.points * 31, struct_size=4)
        self.kernels["initialize"].dispatch(
            thread_count=[self.points, 1, 1],
            numberPoints=self.points,
            state=state[0],
        )
        self._kappa = float(parameters.kappa)
        dt = (
            parameters.relaxation
            * min(self.spacing) ** 2
            / 6.0
        )
        current = 0
        converged = False
        scalar_readbacks = 0
        controls = np.array([np.inf, np.inf, 1.0, 1.0])
        nx, ny, nz = self.shape
        for iteration in range(1, parameters.maximum_iterations + 1):
            self._operators(
                state[current], source, derivatives, residual
            )
            candidate = 1 - current
            self.kernels["update"].dispatch(
                thread_count=[self.points, 1, 1],
                numberPoints=self.points,
                nx=nx,
                ny=ny,
                nz=nz,
                dtInput=float(dt),
                minimumPsiInput=float(
                    parameters.minimum_conformal_factor
                ),
                maximumPsiInput=float(
                    parameters.maximum_conformal_factor
                ),
                state=state[current],
                residual=residual,
                candidate=state[candidate],
            )
            current = candidate
            if (
                iteration == 1
                or iteration % parameters.report_interval == 0
            ):
                self._operators(
                    state[current], source, derivatives, residual
                )
                self.kernels["reduce"].dispatch(
                    thread_count=[1, 1, 1],
                    numberPoints=self.points,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    state=state[current],
                    residual=residual,
                    control=control,
                )
                controls = control.to_numpy().view(np.float64)
                scalar_readbacks += 1
                if (
                    max(float(controls[0]), float(controls[1]))
                    <= parameters.tolerance
                ):
                    converged = True
                    break
                if (
                    not np.all(np.isfinite(controls))
                    or controls[3]
                    >= parameters.maximum_conformal_factor
                ):
                    break
        self._operators(state[current], source, derivatives, residual)
        self.kernels["reduce"].dispatch(
            thread_count=[1, 1, 1],
            numberPoints=self.points,
            nx=nx,
            ny=ny,
            nz=nz,
            state=state[current],
            residual=residual,
            control=control,
        )
        controls = control.to_numpy().view(np.float64)
        scalar_readbacks += 1
        self.kernels["pack"].dispatch(
            thread_count=[self.points, 1, 1],
            numberPoints=self.points,
            state=state[current],
            derivatives=derivatives,
            ccz4State=ccz4,
        )
        packed = (
            state[current]
            .to_numpy()
            .view(np.float64)
            .reshape(self.points, 4)
        )
        psi = packed[:, 0].reshape(self.shape, order="F").copy()
        vector = np.stack(
            [
                packed[:, 1 + component].reshape(
                    self.shape, order="F"
                )
                for component in range(3)
            ]
        )
        report = CTTReport(
            converged=converged,
            iterations=iteration,
            hamiltonian_residual_l2=float(controls[0]),
            momentum_residual_l2=float(controls[1]),
            minimum_conformal_factor=float(controls[2]),
        )
        return VulkanCTTResult(
            conformal_factor=psi,
            vector_potential=vector,
            ccz4_state=(
                ccz4.to_numpy()
                .view(np.float32)
                .reshape(self.points, 31)
            ),
            report=report,
            scalar_readbacks=scalar_readbacks,
        )
