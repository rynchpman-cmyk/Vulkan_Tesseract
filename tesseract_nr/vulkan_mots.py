"""Bridge resident Vulkan CCZ4 snapshots to the qualified KKT/MOTS tracker.

The horizon finder is a read-only diagnostic.  The spacetime evolves on the
GPU and is copied to the host only at an explicitly requested observation
cadence.  Hard KKT decisions, continuation seeds, and stability-sign branch
classification remain exactly those of :mod:`tesseract_nr.kkt_horizon`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ccz4 import CCZ4Parameters, CCZ4Solver, CCZ4State
from .grid import CartesianGrid
from .kkt_horizon import (
    CCZ4PhysicalNormalMOTSObservation,
    CCZ4PhysicalNormalMOTSTracker,
    MOTSBranchDefinition,
    ParametricAxisymmetricMOTSFinder,
    SpectralMOTSParameters,
)
from .vulkan_backend import CCZ4_STATE_WIDTH


@dataclass(frozen=True)
class VulkanCCZ4Layout:
    """Spatial metadata required to interpret the packed Vulkan ABI."""

    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]
    origins: tuple[float, float, float] | None = None
    derivative_order: int = 4

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        lengths = tuple(float(value) for value in self.lengths)
        origins = (
            tuple(-0.5 * value for value in lengths)
            if self.origins is None
            else tuple(float(value) for value in self.origins)
        )
        if len(shape) != 3 or len(lengths) != 3 or len(origins) != 3:
            raise ValueError("MOTS tracking requires one three-dimensional grid")
        if any(value < 8 for value in shape):
            raise ValueError("each MOTS grid axis needs at least eight cells")
        if any(value <= 0.0 for value in lengths):
            raise ValueError("grid lengths must be positive")
        if not np.all(np.isfinite(origins)):
            raise ValueError("grid origins must be finite")
        if self.derivative_order not in {2, 4}:
            raise ValueError("derivative_order must be 2 or 4")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "origins", origins)

    @property
    def points(self) -> int:
        return int(np.prod(self.shape))

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(
            length / count
            for length, count in zip(self.lengths, self.shape)
        )

    def diagnostics(
        self,
        *,
        reference_radius: float = 0.418,
        minimum_cells_per_radius: float = 16.0,
    ) -> dict[str, object]:
        cells = reference_radius / max(self.spacing)
        return {
            "shape": list(self.shape),
            "lengths": list(self.lengths),
            "origins": list(self.origins),
            "spacing": list(self.spacing),
            "derivative_order": self.derivative_order,
            "reference_radius": reference_radius,
            "reference_cells_per_radius": cells,
            "minimum_cells_per_radius": minimum_cells_per_radius,
            "reference_resolution_gate_passed": (
                cells >= minimum_cells_per_radius * (1.0 - 1.0e-12)
            ),
            "packed_snapshot_bytes": (
                self.points * CCZ4_STATE_WIDTH * np.dtype(np.float32).itemsize
            ),
        }


@dataclass(frozen=True)
class VulkanCCZ4HostSnapshot:
    """A backend-independent packed snapshot plus its spatial contract."""

    ccz4_state: np.ndarray
    layout: VulkanCCZ4Layout
    time: float
    steps: int
    source: str = "explicit Vulkan readback"


def _packed_ccz4(values: np.ndarray, layout: VulkanCCZ4Layout) -> np.ndarray:
    packed = np.asarray(values)
    if packed.shape != (layout.points, CCZ4_STATE_WIDTH):
        raise ValueError(
            f"packed CCZ4 shape must be {(layout.points, CCZ4_STATE_WIDTH)}, "
            f"got {packed.shape}"
        )
    if not np.all(np.isfinite(packed)):
        raise FloatingPointError("packed CCZ4 state contains non-finite values")
    return packed


def pack_ccz4_state(
    state: CCZ4State,
    layout: VulkanCCZ4Layout,
    *,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> np.ndarray:
    """Pack CPU CCZ4 fields using the resident x-fastest Vulkan layout."""

    packed = np.zeros((layout.points, CCZ4_STATE_WIDTH), dtype=dtype)

    def flatten(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.shape != layout.shape:
            raise ValueError(
                f"CCZ4 field shape {array.shape} does not match {layout.shape}"
            )
        return array.reshape(-1, order="F")

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
        packed[:, 28 + axis] = flatten(state.shift_driver[axis])
    return packed


def unpack_ccz4_state(
    packed: np.ndarray,
    layout: VulkanCCZ4Layout,
    *,
    time: float = 0.0,
) -> CCZ4State:
    """Reconstruct host CCZ4 tensors from the x-fastest resident ABI."""

    values = _packed_ccz4(packed, layout)

    def field(column: int) -> np.ndarray:
        return np.asarray(values[:, column], dtype=float).reshape(
            layout.shape, order="F"
        )

    metric = np.empty((3, 3) + layout.shape, dtype=float)
    conformal_a = np.empty_like(metric)
    gamma_hat = np.empty((3,) + layout.shape, dtype=float)
    shift = np.empty_like(gamma_hat)
    shift_driver = np.empty_like(gamma_hat)
    for row in range(3):
        for column in range(3):
            metric[row, column] = field(row * 3 + column)
            conformal_a[row, column] = field(
                9 + row * 3 + column
            )
    for axis in range(3):
        gamma_hat[axis] = field(21 + axis)
        shift[axis] = field(25 + axis)
        shift_driver[axis] = field(28 + axis)
    conformal_factor = field(18)
    lapse = field(24)
    if np.any(conformal_factor <= 0.0) or np.any(lapse <= 0.0):
        raise FloatingPointError(
            "Vulkan snapshot has non-positive conformal factor or lapse"
        )
    return CCZ4State(
        metric,
        conformal_a,
        conformal_factor,
        field(19),
        field(20),
        gamma_hat,
        lapse,
        shift,
        shift_driver,
        float(time),
    )


def snapshot_from_backend(
    value: Any,
    layout: VulkanCCZ4Layout | None = None,
) -> VulkanCCZ4HostSnapshot:
    """Normalize vacuum, coupled, or evolved backend snapshots."""

    snapshot = value.coupled if hasattr(value, "coupled") else value
    required = ("ccz4_state", "time", "steps")
    missing = [name for name in required if not hasattr(snapshot, name)]
    if missing:
        raise TypeError(
            "snapshot is missing Vulkan CCZ4 metadata: " + ", ".join(missing)
        )
    if layout is None:
        metadata = ("shape", "lengths", "origins")
        absent = [name for name in metadata if not hasattr(snapshot, name)]
        if absent:
            raise TypeError(
                "snapshot needs an explicit layout because it omits: "
                + ", ".join(absent)
            )
        layout = VulkanCCZ4Layout(
            tuple(snapshot.shape),
            tuple(snapshot.lengths),
            tuple(snapshot.origins),
            int(getattr(snapshot, "derivative_order", 4)),
        )
    elif hasattr(snapshot, "shape") and tuple(snapshot.shape) != layout.shape:
        raise ValueError("Vulkan snapshot shape does not match explicit layout")
    return VulkanCCZ4HostSnapshot(
        _packed_ccz4(snapshot.ccz4_state, layout),
        layout,
        float(snapshot.time),
        int(snapshot.steps),
    )


def load_vulkan_ccz4_checkpoint(path: str | Path) -> VulkanCCZ4HostSnapshot:
    """Read just the diagnostic CCZ4 slice from a Vulkan checkpoint."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as checkpoint:
        required = {
            "schema",
            "shape",
            "lengths",
            "origins",
            "derivative_order",
            "ccz4_state",
            "time",
            "steps",
        }
        missing = required - set(checkpoint.files)
        if missing:
            raise ValueError(
                f"incomplete Vulkan CCZ4 checkpoint: {sorted(missing)}"
            )
        schema = str(checkpoint["schema"])
        if not schema.startswith("vulkan-tesseract."):
            raise ValueError(f"unsupported checkpoint schema {schema!r}")
        layout = VulkanCCZ4Layout(
            tuple(int(value) for value in checkpoint["shape"]),
            tuple(float(value) for value in checkpoint["lengths"]),
            tuple(float(value) for value in checkpoint["origins"]),
            int(checkpoint["derivative_order"]),
        )
        packed = np.asarray(checkpoint["ccz4_state"], dtype=np.float32)
        time = float(checkpoint["time"])
        steps = int(checkpoint["steps"])
    return VulkanCCZ4HostSnapshot(
        _packed_ccz4(packed, layout),
        layout,
        time,
        steps,
        source=f"checkpoint:{source}",
    )


def brill_lindquist_puncture_ccz4(
    layout: VulkanCCZ4Layout,
    *,
    separation: float = 0.78,
    total_bare_mass: float = 1.0,
) -> np.ndarray:
    """Construct equal-mass time-symmetric puncture data directly in CCZ4.

    For ``gamma_ij = psi^4 delta_ij``, the CCZ4 conformal metric is exactly
    ``delta_ij`` and ``W = det(gamma)^(-1/6) = psi^-2``.  The pre-collapsed
    lapse is therefore ``alpha=W``.
    """

    if separation <= 0.0 or total_bare_mass <= 0.0:
        raise ValueError("separation and total bare mass must be positive")
    axes = [
        start + (np.arange(count, dtype=float) + 0.5) * spacing
        for start, count, spacing in zip(
            layout.origins, layout.shape, layout.spacing
        )
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    half_separation = 0.5 * separation
    puncture_floor = 0.1 * min(layout.spacing)
    r_plus = np.maximum(
        np.sqrt(x * x + y * y + (z - half_separation) ** 2),
        puncture_floor,
    )
    r_minus = np.maximum(
        np.sqrt(x * x + y * y + (z + half_separation) ** 2),
        puncture_floor,
    )
    individual_mass = 0.5 * total_bare_mass
    psi = (
        1.0
        + individual_mass / (2.0 * r_plus)
        + individual_mass / (2.0 * r_minus)
    )
    conformal_factor = psi ** (-2)
    packed = np.zeros(
        (layout.points, CCZ4_STATE_WIDTH), dtype=np.float32
    )
    for axis in range(3):
        packed[:, axis * 3 + axis] = 1.0
    packed[:, 18] = conformal_factor.reshape(-1, order="F")
    packed[:, 24] = packed[:, 18]
    return packed


def quadratic_minimum(coordinates: np.ndarray, values: np.ndarray) -> float:
    index = int(np.argmin(values))
    if index == 0 or index == len(values) - 1:
        return float(coordinates[index])
    left, center, right = values[index - 1 : index + 2]
    denominator = left - 2.0 * center + right
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float(coordinates[index])
    offset = 0.5 * (left - right) / denominator
    return float(
        coordinates[index]
        + offset * (coordinates[index + 1] - coordinates[index])
    )


def puncture_centers(
    grid: CartesianGrid, state: CCZ4State
) -> tuple[float, float]:
    """Locate the two moving punctures from axial minima of ``W``."""

    z = grid.coordinates()[2][0, 0, :]
    profile = np.min(state.conformal_factor, axis=(0, 1))
    negative = z < 0.0
    positive = z > 0.0
    if not np.any(negative) or not np.any(positive):
        raise FloatingPointError("grid does not straddle both punctures")
    lower = quadratic_minimum(z[negative], profile[negative])
    upper = quadratic_minimum(z[positive], profile[positive])
    if not lower < upper:
        raise FloatingPointError("puncture locator lost branch ordering")
    return lower, upper


def _modes(
    maximum_degree: int, even_only: bool
) -> tuple[tuple[int, int], ...]:
    start = 2 if even_only else 1
    stride = 2 if even_only else 1
    return (
        (0, 0),
        *(
            (ell, 0)
            for ell in range(start, maximum_degree + 1, stride)
        ),
    )


def _parameters(
    grid: CartesianGrid,
    maximum_degree: int,
    even_only: bool,
    expansion_tolerance: float,
    maximum_iterations: int,
) -> SpectralMOTSParameters:
    return SpectralMOTSParameters(
        modes=_modes(maximum_degree, even_only),
        include_dipole=not even_only,
        n_theta=max(20, 4 * maximum_degree),
        n_phi=max(12, 2 * maximum_degree),
        minimum_radius=max(0.55 * min(grid.spacing), 0.015),
        maximum_radius=0.46 * min(grid.lengths),
        maximum_mode_amplitude=2.0,
        expansion_tolerance=expansion_tolerance,
        projected_gradient_tolerance=1.0e-9,
        jacobian_step=2.0e-4,
        gauss_newton_damping=1.0e-10,
        trust_radius=0.25,
        maximum_iterations=maximum_iterations,
    )


def classify_common(
    observations: dict[str, CCZ4PhysicalNormalMOTSObservation],
) -> dict[str, object]:
    """Assign common branches by principal stability sign, not seed name."""

    candidates: dict[
        str, list[tuple[str, CCZ4PhysicalNormalMOTSObservation]]
    ] = {"common_outer": [], "common_inner": []}
    ambiguous: list[str] = []
    for seed_name, observation in observations.items():
        stability = observation.normal_stability
        if (
            not observation.branch.result.found
            or stability is None
            or not stability.valid
        ):
            continue
        real = float(stability.principal_eigenvalue.real)
        if real > 0.0:
            candidates["common_outer"].append((seed_name, observation))
        elif real < 0.0:
            candidates["common_inner"].append((seed_name, observation))
        else:
            ambiguous.append(seed_name)
    result: dict[str, object] = {"ambiguous_seed_names": ambiguous}
    for physical_name, values in candidates.items():
        if not values:
            result[physical_name] = None
            continue
        values.sort(
            key=lambda item: item[1].branch.result.spectral_residual_norm
        )
        seed_name, observation = values[0]
        diagnostic = observation.diagnostics()
        diagnostic["tracker_seed_name"] = seed_name
        result[physical_name] = diagnostic
        ambiguous.extend(name for name, _ in values[1:])
    return result


@dataclass(frozen=True)
class VulkanPunctureMOTSParameters:
    """Observer settings copied from the qualified CPU puncture campaign."""

    separation: float = 0.78
    maximum_degree: int = 16
    individual_maximum_degree: int = 8
    expansion_tolerance: float = 5.0e-4
    maximum_iterations: int = 60
    branch_split_log_radius: float = 0.2
    track_individual: bool = True
    minimum_cells_per_mean_radius: float = 16.0
    minimum_resolved_mode_magnitude: float = 5.0e-3
    mask_cells: float = 5.0
    boundary_mask_cells: float = 5.0
    compute_constraints: bool = True

    def __post_init__(self) -> None:
        if self.separation <= 0.0:
            raise ValueError("separation must be positive")
        if self.maximum_degree < 2 or self.maximum_degree % 2:
            raise ValueError("common maximum degree must be positive and even")
        if self.individual_maximum_degree < 1:
            raise ValueError("individual maximum degree must be positive")
        if self.expansion_tolerance <= 0.0 or self.maximum_iterations < 1:
            raise ValueError("invalid MOTS solve tolerance or iteration limit")
        if self.branch_split_log_radius <= 0.0:
            raise ValueError("branch split must be positive")
        if self.mask_cells <= 0.0 or self.boundary_mask_cells <= 0.0:
            raise ValueError("constraint mask widths must be positive")


class VulkanPunctureMOTSObserver:
    """Restartable common/individual tracker for Vulkan puncture snapshots."""

    checkpoint_schema = "vulkan-tesseract.puncture-mots-observer.v1"

    def __init__(
        self,
        layout: VulkanCCZ4Layout,
        parameters: VulkanPunctureMOTSParameters | None = None,
    ) -> None:
        self.layout = layout
        self.parameters = parameters or VulkanPunctureMOTSParameters()
        self.grid = CartesianGrid(
            layout.shape,
            layout.lengths,
            origin=layout.origins,
            method=f"fd{layout.derivative_order}",
        )
        self.ccz4 = CCZ4Solver(
            self.grid,
            CCZ4Parameters(
                slicing="1+log",
                evolve_shift=True,
                radiative_boundaries=True,
            ),
        )
        p = self.parameters
        common_finder = ParametricAxisymmetricMOTSFinder(
            self.grid,
            _parameters(
                self.grid,
                p.maximum_degree,
                True,
                p.expansion_tolerance,
                p.maximum_iterations,
            ),
        )
        self.common = CCZ4PhysicalNormalMOTSTracker(
            self.ccz4,
            common_finder,
            (
                MOTSBranchDefinition(
                    "common_outer", (0.0, 0.0, 0.0), 0.58
                ),
                MOTSBranchDefinition(
                    "common_inner", (0.0, 0.0, 0.0), 0.32
                ),
            ),
            minimum_cells_per_mean_radius=p.minimum_cells_per_mean_radius,
            minimum_resolved_mode_magnitude=(
                p.minimum_resolved_mode_magnitude
            ),
        )
        self.individual: CCZ4PhysicalNormalMOTSTracker | None = None
        if p.track_individual:
            individual_finder = ParametricAxisymmetricMOTSFinder(
                self.grid,
                _parameters(
                    self.grid,
                    p.individual_maximum_degree,
                    False,
                    p.expansion_tolerance,
                    p.maximum_iterations,
                ),
            )
            self.individual = CCZ4PhysicalNormalMOTSTracker(
                self.ccz4,
                individual_finder,
                (
                    MOTSBranchDefinition(
                        "individual_minus",
                        (0.0, 0.0, -0.5 * p.separation),
                        0.18,
                    ),
                    MOTSBranchDefinition(
                        "individual_plus",
                        (0.0, 0.0, 0.5 * p.separation),
                        0.18,
                    ),
                ),
                minimum_cells_per_mean_radius=(
                    p.minimum_cells_per_mean_radius
                ),
                minimum_resolved_mode_magnitude=(
                    p.minimum_resolved_mode_magnitude
                ),
            )

    def observe_packed(
        self,
        packed: np.ndarray,
        *,
        time: float,
        steps: int,
    ) -> dict[str, object]:
        """Observe one explicit readback and update all continuation branches."""

        state = unpack_ccz4_state(packed, self.layout, time=time)
        lower, upper = puncture_centers(self.grid, state)
        midpoint = 0.5 * (lower + upper)
        for name in ("common_outer", "common_inner"):
            self.common.update_center(name, (0.0, 0.0, midpoint))
        common = self.common.observe(state)
        found_common = [
            value
            for value in common.values()
            if value.branch.result.found
            and value.normal_stability is not None
            and value.normal_stability.valid
        ]
        if len(found_common) == 1:
            source = found_common[0]
            seed = source.branch.result.coefficients.copy()
            split = self.parameters.branch_split_log_radius
            if source.normal_stability.principal_eigenvalue.real < 0.0:
                self.common.seed_branch("common_inner", seed)
                seed[0] += split
                self.common.seed_branch("common_outer", seed)
            else:
                self.common.seed_branch("common_outer", seed)
                seed[0] -= split
                self.common.seed_branch("common_inner", seed)
        individual = None
        if self.individual is not None:
            self.individual.update_center(
                "individual_minus", (0.0, 0.0, lower)
            )
            self.individual.update_center(
                "individual_plus", (0.0, 0.0, upper)
            )
            individual = self.individual.observe(state)
        matrices = np.moveaxis(
            state.conformal_metric, (0, 1), (-2, -1)
        )
        health: dict[str, object] = {
            "finite": bool(
                all(
                    np.all(np.isfinite(getattr(state, name)))
                    for name in (
                        "conformal_metric",
                        "conformal_A",
                        "conformal_factor",
                        "trace_K",
                        "theta",
                        "gamma_hat",
                        "lapse",
                        "shift",
                        "shift_driver",
                    )
                )
            ),
            "minimum_lapse": float(np.min(state.lapse)),
            "maximum_lapse": float(np.max(state.lapse)),
            "minimum_conformal_factor": float(
                np.min(state.conformal_factor)
            ),
            "maximum_conformal_factor": float(
                np.max(state.conformal_factor)
            ),
            "minimum_conformal_metric_eigenvalue": float(
                np.min(np.linalg.eigvalsh(matrices))
            ),
        }
        constraints = (
            self._constraint_diagnostics(state, (lower, upper))
            if self.parameters.compute_constraints
            else None
        )
        return {
            "step": int(steps),
            "time": float(time),
            "puncture_z": [lower, upper],
            "coordinate_separation": upper - lower,
            "health": health,
            "constraints": constraints,
            "common": {
                name: observation.diagnostics()
                for name, observation in common.items()
            },
            "common_classified": classify_common(common),
            "individual": (
                None
                if individual is None
                else {
                    name: observation.diagnostics()
                    for name, observation in individual.items()
                }
            ),
        }

    def _constraint_diagnostics(
        self,
        state: CCZ4State,
        punctures: tuple[float, float],
    ) -> dict[str, float]:
        """CPU oracle diagnostics on the already requested host snapshot."""

        hamiltonian, momentum = self.ccz4.constraints(state)
        x, y, z = self.grid.coordinates()
        mask_radius = self.parameters.mask_cells * max(self.grid.spacing)
        mask = np.ones(self.grid.shape, dtype=bool)
        for puncture in punctures:
            mask &= (
                x * x + y * y + (z - puncture) ** 2 >= mask_radius**2
            )
        boundary_distance = np.minimum.reduce(
            [
                x - self.grid.origin[0],
                self.grid.origin[0] + self.grid.lengths[0] - x,
                y - self.grid.origin[1],
                self.grid.origin[1] + self.grid.lengths[1] - y,
                z - self.grid.origin[2],
                self.grid.origin[2] + self.grid.lengths[2] - z,
            ]
        )
        boundary_radius = (
            self.parameters.boundary_mask_cells * max(self.grid.spacing)
        )
        interior = mask & (boundary_distance >= boundary_radius)
        if not np.any(mask) or not np.any(interior):
            raise FloatingPointError("constraint masks removed the entire grid")
        momentum_squared = np.sum(momentum * momentum, axis=0)
        return {
            "puncture_mask_radius": mask_radius,
            "boundary_mask_radius": boundary_radius,
            "unmasked_fraction": float(np.mean(mask)),
            "interior_fraction": float(np.mean(interior)),
            "hamiltonian_l2_full": float(
                np.sqrt(np.mean(hamiltonian**2))
            ),
            "hamiltonian_l2_masked": float(
                np.sqrt(np.mean(hamiltonian[mask] ** 2))
            ),
            "hamiltonian_l2_interior": float(
                np.sqrt(np.mean(hamiltonian[interior] ** 2))
            ),
            "momentum_l2_full": float(
                np.sqrt(np.mean(momentum_squared))
            ),
            "momentum_l2_masked": float(
                np.sqrt(np.mean(momentum_squared[mask]))
            ),
            "momentum_l2_interior": float(
                np.sqrt(np.mean(momentum_squared[interior]))
            ),
        }

    def observe_snapshot(self, snapshot: Any) -> dict[str, object]:
        host = snapshot_from_backend(snapshot, self.layout)
        if host.layout != self.layout:
            raise ValueError("Vulkan snapshot layout does not match observer")
        return self.observe_packed(
            host.ccz4_state, time=host.time, steps=host.steps
        )

    def checkpoint(self) -> dict[str, object]:
        layout = {
            "shape": list(self.layout.shape),
            "lengths": list(self.layout.lengths),
            "origins": list(self.layout.origins),
            "derivative_order": self.layout.derivative_order,
        }
        return {
            "format": self.checkpoint_schema,
            "layout": layout,
            "parameters": asdict(self.parameters),
            "common": self.common.checkpoint(),
            "individual": (
                None
                if self.individual is None
                else self.individual.checkpoint()
            ),
        }

    def restore(self, payload: dict[str, object]) -> None:
        if payload.get("format") != self.checkpoint_schema:
            raise ValueError("unsupported Vulkan puncture/MOTS checkpoint")
        expected_layout = {
            "shape": list(self.layout.shape),
            "lengths": list(self.layout.lengths),
            "origins": list(self.layout.origins),
            "derivative_order": self.layout.derivative_order,
        }
        if payload.get("layout") != expected_layout:
            raise ValueError("observer checkpoint grid layout does not match")
        if payload.get("parameters") != asdict(self.parameters):
            raise ValueError("observer checkpoint parameters do not match")
        common = payload.get("common")
        if not isinstance(common, dict):
            raise ValueError("observer checkpoint is missing common branches")
        self.common.restore(common)
        individual = payload.get("individual")
        if self.individual is None:
            if individual is not None:
                raise ValueError("checkpoint contains disabled individual branches")
        elif not isinstance(individual, dict):
            raise ValueError("checkpoint is missing individual branches")
        else:
            self.individual.restore(individual)
