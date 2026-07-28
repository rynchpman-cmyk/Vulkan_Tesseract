"""Resident Vulkan conservative AMR transfer and synchronization primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .vulkan_backend import _slangpy


Array = np.ndarray


class VulkanAMRBackend:
    """Ratio-two conservative transfer, Gauss projection, and 1D reflux."""

    def __init__(self, module_path: str | Path | None = None) -> None:
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

    @staticmethod
    def _expanded(shape: tuple[int, ...]) -> tuple[int, int, int]:
        if not 1 <= len(shape) <= 3 or any(value < 1 for value in shape):
            raise ValueError("AMR shape needs one to three positive axes")
        return tuple(int(value) for value in shape) + (1,) * (
            3 - len(shape)
        )

    @staticmethod
    def _pack(
        values: Array, shape: tuple[int, ...]
    ) -> tuple[Array, tuple[int, ...]]:
        array = np.asarray(values, dtype=np.float32)
        if tuple(array.shape[-len(shape) :]) != shape:
            raise ValueError("field spatial shape does not match AMR grid")
        leading = array.shape[: array.ndim - len(shape)]
        components = int(np.prod(leading)) if leading else 1
        packed = array.reshape((components, *shape)).reshape(
            components, -1, order="F"
        ).T.copy()
        return packed, leading

    @staticmethod
    def _unpack(
        packed: Array,
        leading: tuple[int, ...],
        shape: tuple[int, ...],
    ) -> Array:
        components = packed.shape[1]
        values = np.asarray(packed).T.reshape(
            (components, *shape), order="F"
        )
        return values.reshape((*leading, *shape))

    def prolong(
        self,
        coarse: Array,
        spatial_ndim: int,
    ) -> Array:
        array = np.asarray(coarse)
        if not 1 <= spatial_ndim <= 3:
            raise ValueError("spatial_ndim must be one to three")
        coarse_shape = tuple(int(v) for v in array.shape[-spatial_ndim:])
        fine_shape = tuple(2 * value for value in coarse_shape)
        packed, leading = self._pack(array, coarse_shape)
        components = packed.shape[1]
        coarse_tensor = self.spy.Tensor.from_numpy(self.device, packed)
        fine_tensor = self.spy.Tensor.empty(
            self.device,
            shape=(int(np.prod(fine_shape)), components),
            dtype=self.module.float,
        )
        cnx, cny, cnz = self._expanded(coarse_shape)
        fnx, fny, fnz = self._expanded(fine_shape)
        self.module.theory33AMRProlongKernel(
            point=self.spy.grid(shape=(int(np.prod(fine_shape)),)),
            coarseNx=cnx,
            coarseNy=cny,
            coarseNz=cnz,
            fineNx=fnx,
            fineNy=fny,
            fineNz=fnz,
            componentCount=components,
            coarse=coarse_tensor,
            fine=fine_tensor,
        )
        return self._unpack(
            fine_tensor.to_numpy(), leading, fine_shape
        )

    def restrict(
        self,
        fine: Array,
        spatial_ndim: int,
    ) -> Array:
        array = np.asarray(fine)
        if not 1 <= spatial_ndim <= 3:
            raise ValueError("spatial_ndim must be one to three")
        fine_shape = tuple(int(v) for v in array.shape[-spatial_ndim:])
        if any(value % 2 for value in fine_shape):
            raise ValueError("fine AMR shape must be divisible by two")
        coarse_shape = tuple(value // 2 for value in fine_shape)
        packed, leading = self._pack(array, fine_shape)
        components = packed.shape[1]
        fine_tensor = self.spy.Tensor.from_numpy(self.device, packed)
        coarse_tensor = self.spy.Tensor.empty(
            self.device,
            shape=(int(np.prod(coarse_shape)), components),
            dtype=self.module.float,
        )
        cnx, cny, cnz = self._expanded(coarse_shape)
        fnx, fny, fnz = self._expanded(fine_shape)
        self.module.theory33AMRRestrictKernel(
            point=self.spy.grid(shape=(int(np.prod(coarse_shape)),)),
            coarseNx=cnx,
            coarseNy=cny,
            coarseNz=cnz,
            fineNx=fnx,
            fineNy=fny,
            fineNz=fnz,
            componentCount=components,
            fine=fine_tensor,
            coarse=coarse_tensor,
        )
        return self._unpack(
            coarse_tensor.to_numpy(), leading, coarse_shape
        )

    def project_gauss(
        self,
        pi_a: Array,
        pi_b: Array,
        *,
        shape: tuple[int, ...],
        lengths: tuple[float, ...] | None = None,
        z_a: float = 1.0,
        z_b: float = 1.0,
    ) -> tuple[Array, Array]:
        expanded = self._expanded(shape)
        if z_a <= 0.0 or z_b <= 0.0:
            raise ValueError("kinetic coefficients must be positive")
        if lengths is None:
            lengths = (1.0,) * len(shape)
        if len(lengths) != len(shape):
            raise ValueError("lengths must match shape")
        packed_a, leading_a = self._pack(pi_a, shape)
        packed_b, leading_b = self._pack(pi_b, shape)
        if leading_a != (3,) or leading_b != (3,):
            raise ValueError("Gauss projection needs three-vector momenta")
        points = int(np.prod(shape))
        output = self.spy.Tensor.empty(
            self.device,
            shape=(points, 2),
            dtype=self.module.float,
        )
        spacing = tuple(
            float(lengths[axis]) / shape[axis]
            for axis in range(len(shape))
        ) + (1.0,) * (3 - len(shape))
        self.module.theory33AMRGaussProjectKernel(
            point=self.spy.grid(shape=(points,)),
            nx=expanded[0],
            ny=expanded[1],
            nz=expanded[2],
            inverseDx=1.0 / spacing[0],
            inverseDy=1.0 / spacing[1],
            inverseDz=1.0 / spacing[2],
            ZA=float(z_a),
            ZB=float(z_b),
            piA=self.spy.Tensor.from_numpy(self.device, packed_a),
            piB=self.spy.Tensor.from_numpy(self.device, packed_b),
            longitudinal=output,
        )
        values = output.to_numpy()
        return (
            values[:, 0].reshape(shape, order="F"),
            values[:, 1].reshape(shape, order="F"),
        )

    def reflux_1d(
        self,
        coarse: Array,
        *,
        lower: int,
        upper: int,
        coarse_lower_flux: Array,
        fine_lower_flux: Array,
        coarse_upper_flux: Array,
        fine_upper_flux: Array,
        dt: float,
        dx: float,
    ) -> Array:
        values = np.asarray(coarse, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
            scalar = True
        elif values.ndim == 2:
            scalar = False
        else:
            raise ValueError("1D reflux expects cells or cells-by-components")
        points, components = values.shape
        if not 0 <= lower <= upper < points or dt <= 0.0 or dx <= 0.0:
            raise ValueError("invalid reflux patch or spacing")

        def flux(value: Array, name: str) -> Any:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            if array.size != components:
                raise ValueError(f"{name} does not match component count")
            return self.spy.Tensor.from_numpy(self.device, array)

        correction = self.spy.Tensor.empty(
            self.device,
            shape=(points, components),
            dtype=self.module.float,
        )
        output = self.spy.Tensor.empty(
            self.device,
            shape=(points, components),
            dtype=self.module.float,
        )
        coarse_tensor = self.spy.Tensor.from_numpy(self.device, values)
        self.module.theory33AMRFluxRegister1DKernel(
            point=self.spy.grid(shape=(points,)),
            componentCount=components,
            lower=lower,
            upper=upper,
            dtOverDx=float(dt / dx),
            coarseLowerFlux=flux(
                coarse_lower_flux, "coarse_lower_flux"
            ),
            fineLowerFlux=flux(fine_lower_flux, "fine_lower_flux"),
            coarseUpperFlux=flux(
                coarse_upper_flux, "coarse_upper_flux"
            ),
            fineUpperFlux=flux(fine_upper_flux, "fine_upper_flux"),
            correction=correction,
        )
        self.module.theory33AMRRefluxKernel(
            point=self.spy.grid(shape=(points,)),
            componentCount=components,
            coarse=coarse_tensor,
            correction=correction,
            output=output,
        )
        result = output.to_numpy()
        return result[:, 0] if scalar else result
