"""FP64 Vulkan Tolman--Oppenheimer--Volkoff profile generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .tov import PolytropicEOS, TOVSolution


class VulkanTOVBackend:
    """Run the radial RK4 TOV recurrence on a Vulkan compute queue."""

    def __init__(self, module_path: str | Path | None = None) -> None:
        try:
            import slangpy as spy
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "VulkanTOVBackend requires the slangpy environment"
            ) from exc
        self.spy = spy
        self.device = spy.create_device(type=spy.DeviceType.vulkan)
        root = Path(__file__).resolve().parents[1]
        self.module_path = (
            root / "tesseract_tov_fp64.slang"
            if module_path is None
            else Path(module_path).resolve()
        )
        self.kernels = {}
        for name, entry_point in {
            "initialize": "theory33TOVInitialize64Kernel",
            "advance": "theory33TOVAdvance64Kernel",
            "finalize": "theory33TOVFinalize64Kernel",
            "sample": "theory33TOVSampleCartesianKernel",
        }.items():
            program = self.device.load_program(
                module_name=str(self.module_path),
                entry_point_names=[entry_point],
            )
            self.kernels[name] = self.device.create_compute_kernel(
                program=program
            )
        self._profile = None
        self._control = None
        self._maximum_samples = 0

    def solve(
        self,
        central_density: float,
        eos: PolytropicEOS | None = None,
        *,
        dr: float = 0.01,
        pressure_floor_fraction: float = 1.0e-12,
        maximum_radius: float = 100.0,
    ) -> TOVSolution:
        """Return the same profile ABI as :func:`tesseract_nr.tov.solve_tov`."""
        eos = eos or PolytropicEOS()
        if (
            central_density <= 0.0
            or dr <= 0.0
            or maximum_radius <= dr
            or pressure_floor_fraction <= 0.0
        ):
            raise ValueError("TOV integration controls are invalid")
        maximum_samples = int(np.ceil(maximum_radius / dr)) + 2
        usage = (
            self.spy.BufferUsage.shader_resource
            | self.spy.BufferUsage.unordered_access
        )
        profile = self.device.create_buffer(
            element_count=7 * maximum_samples,
            struct_size=8,
            usage=usage,
        )
        control = self.device.create_buffer(
            element_count=6,
            struct_size=8,
            usage=usage,
        )
        recurrence = self.device.create_buffer(
            element_count=9,
            struct_size=8,
            usage=usage,
        )
        common = {
            "centralDensityInput": float(central_density),
            "KInput": float(eos.K),
            "gammaInput": float(eos.gamma),
            "drInput": float(dr),
        }
        self.kernels["initialize"].dispatch(
            thread_count=[1, 1, 1],
            **common,
            recurrence=recurrence,
            control=control,
        )
        batch_steps = 24
        for _ in range(
            (maximum_samples + batch_steps - 1) // batch_steps + 1
        ):
            self.kernels["advance"].dispatch(
                thread_count=[1, 1, 1],
                maximumSamples=maximum_samples,
                batchSteps=batch_steps,
                **common,
                pressureFloorFractionInput=float(
                    pressure_floor_fraction
                ),
                maximumRadiusInput=float(maximum_radius),
                profile=profile,
                recurrence=recurrence,
                control=control,
            )
        self.kernels["finalize"].dispatch(
            thread_count=[maximum_samples, 1, 1],
            maximumSamples=maximum_samples,
            profile=profile,
            control=control,
        )
        controls = control.to_numpy().view(np.float64)
        count = int(round(float(controls[0])))
        if controls[4] < 0.5 or count < 1:
            raise FloatingPointError(
                "Vulkan TOV integration did not reach a regular surface"
            )
        values = (
            profile.to_numpy()
            .view(np.float64)[: 7 * count]
            .reshape(count, 7)
        )
        self._profile = profile
        self._control = control
        self._maximum_samples = maximum_samples
        return TOVSolution(
            radius=values[:, 0].copy(),
            density=values[:, 1].copy(),
            pressure=values[:, 2].copy(),
            energy_density=values[:, 3].copy(),
            enclosed_mass=values[:, 4].copy(),
            lapse=values[:, 5].copy(),
            radial_metric=values[:, 6].copy(),
            surface_radius=float(controls[1]),
            gravitational_mass=float(controls[2]),
            baryonic_mass=float(controls[3]),
        )

    def sample_cartesian(
        self,
        shape: tuple[int, ...],
        *,
        lengths: tuple[float, ...],
        origins: tuple[float, ...] | None = None,
        density_floor: float = 1.0e-10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the last resident TOV profile onto a Cartesian NR patch."""
        if self._profile is None or self._control is None:
            raise RuntimeError("solve a TOV profile before sampling it")
        if not 1 <= len(shape) <= 3 or len(lengths) != len(shape):
            raise ValueError("shape and lengths must describe 1D, 2D, or 3D")
        if any(int(value) < 1 for value in shape) or any(
            value <= 0.0 for value in lengths
        ):
            raise ValueError("shape and lengths must be positive")
        if density_floor <= 0.0:
            raise ValueError("density_floor must be positive")
        ndim = len(shape)
        expanded_shape = tuple(int(value) for value in shape) + (1,) * (
            3 - ndim
        )
        expanded_lengths = tuple(float(value) for value in lengths) + (
            1.0,
        ) * (3 - ndim)
        if origins is None:
            origins = tuple(-0.5 * value for value in lengths)
        if len(origins) != ndim:
            raise ValueError("origins must match shape")
        expanded_origins = tuple(float(value) for value in origins) + (
            0.0,
        ) * (3 - ndim)
        spacing = tuple(
            expanded_lengths[axis] / expanded_shape[axis]
            for axis in range(3)
        )
        points = int(np.prod(expanded_shape))
        usage = (
            self.spy.BufferUsage.shader_resource
            | self.spy.BufferUsage.unordered_access
        )

        def output(width: int):
            return self.device.create_buffer(
                element_count=points * width,
                struct_size=4,
                usage=usage,
            )

        primitive = output(9)
        ccz4 = output(31)
        vector = output(16)
        self.kernels["sample"].dispatch(
            thread_count=[points, 1, 1],
            numberPoints=points,
            nx=expanded_shape[0],
            ny=expanded_shape[1],
            nz=expanded_shape[2],
            dx=spacing[0],
            dy=spacing[1],
            dz=spacing[2],
            originX=expanded_origins[0],
            originY=expanded_origins[1],
            originZ=expanded_origins[2],
            densityFloor=float(density_floor),
            profile=self._profile,
            control=self._control,
            primitive=primitive,
            ccz4State=ccz4,
            vectorState=vector,
        )
        return (
            primitive.to_numpy().view(np.float32).reshape(points, 9),
            ccz4.to_numpy().view(np.float32).reshape(points, 31),
            vector.to_numpy().view(np.float32).reshape(points, 16),
        )
