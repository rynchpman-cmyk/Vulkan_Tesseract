"""Bounded in-situ observation staging for resident Vulkan evolution.

SlangPy owns command submission and synchronization.  This layer therefore
does not claim explicit Vulkan timeline-semaphore control: it dispatches a
compact observation kernel into a rotating device buffer, performs the
readback on one worker, and drops capture requests while that worker is busy.
The solver never waits for visualization unless the caller explicitly invokes
``wait`` during shutdown.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
from typing import Any

import numpy as np

from .scene import OBSERVATION_FIELDS
from .scene import LiveSceneWriter


@dataclass(frozen=True)
class VulkanObservationFrame:
    values: np.ndarray
    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]
    origins: tuple[float, float, float]
    derivative_order: int
    step: int
    time: float
    dropped_before: int

    def field(self, identifier: str) -> np.ndarray:
        names = [item[0] for item in OBSERVATION_FIELDS]
        try:
            column = names.index(identifier)
        except ValueError as exc:
            raise KeyError(identifier) from exc
        return self.values[:, column].reshape(self.shape, order="F")


class VulkanCCZ4ObservationRing:
    """Latest-frame, nonblocking bridge from resident CCZ4 to host fields."""

    width = len(OBSERVATION_FIELDS)

    def __init__(
        self,
        backend: Any,
        slots: int = 3,
        maximum_axis: int = 64,
    ) -> None:
        if slots < 2:
            raise ValueError("observation staging needs at least two slots")
        if maximum_axis < 8:
            raise ValueError("observation maximum_axis must be at least eight")
        required = (
            "spy",
            "device",
            "module",
            "grid",
            "shape",
            "lengths",
            "origins",
            "spacing",
            "derivative_order",
            "cosmological_constant",
        )
        missing = [name for name in required if not hasattr(backend, name)]
        if missing:
            raise TypeError(
                "backend does not expose observation ABI: "
                + ", ".join(missing)
            )
        self.backend = backend
        self.source_shape = tuple(int(value) for value in backend.shape)
        self.shape = tuple(
            min(int(value), int(maximum_axis))
            for value in self.source_shape
        )
        self.lengths = tuple(float(value) for value in backend.lengths)
        self.origins = tuple(float(value) for value in backend.origins)
        self.source_spacing = tuple(
            float(value) for value in backend.spacing
        )
        self.spacing = tuple(
            length / count
            for length, count in zip(self.lengths, self.shape)
        )
        self.points = int(np.prod(self.shape))
        self.derivative_order = int(backend.derivative_order)
        self._buffers = [
            backend.spy.Tensor.empty(
                backend.device,
                shape=(self.points, self.width),
                dtype=backend.module.float,
            )
            for _ in range(slots)
        ]
        self._grid = backend.spy.grid(shape=(self.points,))
        self._next_slot = 0
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tesseract-observer"
        )
        self._future: Future[np.ndarray] | None = None
        self._metadata: tuple[int, float, int] | None = None
        self._lock = threading.Lock()
        self.dropped = 0
        self.captured = 0

    def _state_tensor(self, state: Any) -> Any:
        if hasattr(state, "tensor"):
            return state.tensor
        if hasattr(state, "ccz4"):
            return state.ccz4
        raise TypeError("state does not expose a resident CCZ4 tensor")

    def capture(self, state: Any) -> bool:
        """Schedule one observation or drop it when a readback is in flight."""

        with self._lock:
            if self._future is not None and not self._future.done():
                self.dropped += 1
                return False
            if self._future is not None:
                # A completed frame must be polled before its slot is reused.
                self.dropped += 1
                return False
            slot = self._next_slot
            self._next_slot = (self._next_slot + 1) % len(self._buffers)
            target = self._buffers[slot]
            nx, ny, nz = self.source_shape
            observation_nx, observation_ny, observation_nz = self.shape
            self.backend.module.tesseractObserveCCZ4Kernel(
                point=self._grid,
                nx=nx,
                ny=ny,
                nz=nz,
                observationNx=observation_nx,
                observationNy=observation_ny,
                observationNz=observation_nz,
                inverseDx=float(1.0 / self.source_spacing[0]),
                inverseDy=float(1.0 / self.source_spacing[1]),
                inverseDz=float(1.0 / self.source_spacing[2]),
                derivativeOrder=self.derivative_order,
                cosmologicalConstant=float(
                    self.backend.cosmological_constant
                ),
                state=self._state_tensor(state),
                observation=target,
            )
            self._metadata = (
                int(getattr(state, "steps", 0)),
                float(getattr(state, "time", 0.0)),
                self.dropped,
            )
            self._future = self._executor.submit(target.to_numpy)
            self.captured += 1
            return True

    def poll(self) -> VulkanObservationFrame | None:
        with self._lock:
            if self._future is None or not self._future.done():
                return None
            future = self._future
            metadata = self._metadata
            self._future = None
            self._metadata = None
        assert metadata is not None
        values = np.asarray(future.result(), dtype=np.float32)
        if values.shape != (self.points, self.width):
            raise RuntimeError("GPU observation returned an invalid shape")
        if not np.isfinite(values).all():
            raise FloatingPointError("GPU observation contains non-finite data")
        return VulkanObservationFrame(
            values,
            self.shape,
            self.lengths,
            self.origins,
            self.derivative_order,
            metadata[0],
            metadata[1],
            metadata[2],
        )

    def wait(self) -> VulkanObservationFrame | None:
        with self._lock:
            future = self._future
        if future is not None:
            future.result()
        return self.poll()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "VulkanCCZ4ObservationRing":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AsyncLiveSceneSink:
    """Serialize scene frames on one worker with latest-frame backpressure."""

    def __init__(self, writer: LiveSceneWriter) -> None:
        self.writer = writer
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tesseract-scene-writer"
        )
        self._future: Future[dict[str, Any]] | None = None
        self.written = 0
        self.dropped = 0

    def submit(self, frame: VulkanObservationFrame) -> bool:
        self.poll()
        if self._future is not None:
            self.dropped += 1
            return False
        self._future = self._executor.submit(
            self.writer.append,
            frame.values,
            step=frame.step,
            time=frame.time,
            dropped_before=frame.dropped_before + self.dropped,
        )
        return True

    def poll(self) -> dict[str, Any] | None:
        if self._future is None or not self._future.done():
            return None
        future = self._future
        self._future = None
        result = future.result()
        self.written += 1
        return result

    def wait(self) -> dict[str, Any] | None:
        if self._future is not None:
            self._future.result()
        return self.poll()

    def close(self) -> None:
        self.wait()
        self._executor.shutdown(wait=True, cancel_futures=False)
