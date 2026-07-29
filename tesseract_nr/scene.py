"""Tesseract Observatory scene contract and checkpoint exporters.

``.tessviz`` directories are derived visualization products.  They never
replace restart checkpoints: every scene records its source digests, numerical
coordinate contract, field derivations, precision, and evidence links.

The volume hierarchy is deliberately a small, dependency-free subset of Zarr
v3.  Each level is a regular little-endian float32 array with raw ``bytes``
chunks.  This keeps the first browser reader auditable while remaining
readable by conforming Zarr implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import pi
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np

from .kkt_horizon import evaluate_basis
from .vulkan_mots import (
    VulkanCCZ4HostSnapshot,
    load_vulkan_ccz4_checkpoint,
    unpack_ccz4_state,
)

SCENE_SCHEMA = "tesseract.scene.v1"
SCENE_FORMAT_VERSION = 1
DEFAULT_CHUNK_SHAPE = (32, 32, 32)


class SceneContractError(ValueError):
    """Raised when a scene violates the public Observatory contract."""


@dataclass(frozen=True)
class SceneExportResult:
    path: Path
    manifest: dict[str, Any]
    fields: tuple[str, ...]
    geometry_objects: int


OBSERVATION_FIELDS = (
    ("ccz4.lapse", "Lapse", "alpha", False),
    ("ccz4.conformal_factor", "Conformal factor", "chi", False),
    ("ccz4.trace_k", "Trace of extrinsic curvature", "K", True),
    ("ccz4.theta", "CCZ4 cleaning field", "Theta", True),
    ("ccz4.shift_magnitude", "Shift magnitude", "|beta|", False),
    (
        "constraints.hamiltonian",
        "Vacuum Hamiltonian constraint",
        "H",
        True,
    ),
    ("constraints.z_magnitude", "CCZ4 Z-constraint magnitude", "|Z|", False),
    (
        "constraints.conformal_determinant",
        "Conformal determinant residual",
        "|det(g_tilde)-1|",
        False,
    ),
)


class LiveSceneWriter:
    """Append compact GPU observations to one atomic live scene manifest."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        shape: tuple[int, int, int],
        lengths: tuple[float, float, float],
        origins: tuple[float, float, float],
        derivative_order: int,
        source: str = "resident Vulkan CCZ4 observation graph",
        maximum_levels: int = 4,
        chunk_shape: tuple[int, int, int] = DEFAULT_CHUNK_SHAPE,
    ) -> None:
        self.path = Path(output_path).resolve()
        if self.path.suffix != ".tessviz":
            raise ValueError("live scene output must use .tessviz")
        self.shape = tuple(int(value) for value in shape)
        self.lengths = tuple(float(value) for value in lengths)
        self.origins = tuple(float(value) for value in origins)
        self.spacing = tuple(
            length / count
            for length, count in zip(self.lengths, self.shape)
        )
        self.derivative_order = int(derivative_order)
        self.source = source
        self.maximum_levels = int(maximum_levels)
        self.chunk_shape = chunk_shape
        self.path.mkdir(parents=True, exist_ok=True)
        group = self.path / "frames.zarr" / "zarr.json"
        if not group.is_file():
            _write_json(
                group,
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {
                        "schema": SCENE_SCHEMA,
                        "streaming": True,
                    },
                },
            )
        scene_path = self.path / "scene.json"
        if scene_path.is_file():
            self.manifest = load_scene(scene_path)
        else:
            axes = [
                {
                    "name": name,
                    "type": "space",
                    "unit": "code_length",
                    "size": count,
                    "origin": origin,
                    "spacing": spacing,
                    "centering": "cell",
                }
                for name, count, origin, spacing in zip(
                    ("x", "y", "z"),
                    self.shape,
                    self.origins,
                    self.spacing,
                )
            ]
            self.manifest = {
                "schema": SCENE_SCHEMA,
                "format_version": SCENE_FORMAT_VERSION,
                "scene_id": f"live-{sha256(str(self.path).encode()).hexdigest()[:12]}",
                "title": "Live Tesseract spacetime observation",
                "description": (
                    "Append-only derived fields produced by the resident "
                    "Vulkan observation graph. Frames may be dropped under "
                    "visualization backpressure."
                ),
                "coordinates": {
                    "chart": "cartesian",
                    "handedness": "right",
                    "axis_order": ["x", "y", "z"],
                    "axes": axes,
                    "gauge": {
                        "slicing": "1+log",
                        "shift": "Gamma-driver",
                        "coordinate_warning": (
                            "Rendered distances are coordinate distances, not "
                            "unique gauge-invariant observables."
                        ),
                    },
                },
                "frames": [],
                "geometry": [],
                "worldtubes": [],
                "evidence": {"path": "evidence.json", "sources": []},
                "provenance": {
                    "producer": "Vulkan Tesseract live observation graph",
                    "source_checkpoint": "resident-state",
                    "source_checkpoint_sha256": "not-applicable-live-state",
                    "source_checkpoint_schema": "resident CCZ4 ABI",
                    "source_precision": "float32",
                    "derivative_order": self.derivative_order,
                    "derived_product": True,
                    "restart_capable": False,
                    "frames_may_be_dropped": True,
                },
            }
            _write_json(self.path / "evidence.json", {})
            _write_json(scene_path, self.manifest)

    def append(
        self,
        values: np.ndarray,
        *,
        step: int,
        time: float,
        dropped_before: int = 0,
    ) -> dict[str, Any]:
        packed = np.asarray(values, dtype=np.float32)
        if packed.shape != (int(np.prod(self.shape)), len(OBSERVATION_FIELDS)):
            raise ValueError("observation frame does not match scene layout")
        fields = []
        frame_root = Path("frames.zarr") / f"step_{int(step):07d}"
        for column, (identifier, name, symbol, signed) in enumerate(
            OBSERVATION_FIELDS
        ):
            volume = packed[:, column].reshape(self.shape, order="F")
            levels = []
            _ensure_zarr_group(
                self.path / frame_root,
                {"step": int(step), "time": float(time)},
            )
            _ensure_zarr_group(
                self.path / frame_root / identifier,
                {"field_id": identifier},
            )
            for level_index, level in enumerate(
                multiscale_pyramid(volume, self.maximum_levels)
            ):
                relative = (
                    frame_root / identifier / f"level_{level_index}"
                )
                descriptor = _write_zarr_array(
                    self.path / relative,
                    level,
                    chunk_shape=self.chunk_shape,
                    attributes={
                        "field_id": identifier,
                        "step": int(step),
                        "time": float(time),
                        "level": level_index,
                        "axis_order": ["x", "y", "z"],
                    },
                )
                descriptor.update(
                    {
                        "level": level_index,
                        "path": relative.as_posix(),
                        "scale": [2**level_index] * 3,
                    }
                )
                levels.append(descriptor)
            fields.append(
                _field_contract(
                    identifier,
                    name,
                    symbol,
                    levels,
                    derivation=(
                        f"{self.source}, compact observation column {column}"
                    ),
                    signed=signed,
                )
            )
        frame = {
            "id": f"step-{int(step):07d}",
            "step": int(step),
            "time": float(time),
            "time_unit": "code_time",
            "fields": fields,
            "geometry": [],
            "stream": {
                "dropped_before": int(dropped_before),
                "complete": True,
            },
        }
        self.manifest["frames"] = [
            item
            for item in self.manifest["frames"]
            if int(item["step"]) != int(step)
        ]
        self.manifest["frames"].append(frame)
        self.manifest["frames"].sort(key=lambda item: int(item["step"]))
        validate_scene(self.manifest)
        _write_json(self.path / "scene.json", self.manifest)
        return frame

    def sync_trajectory(self, trajectory_path: str | Path) -> None:
        """Attach the latest spectral MOTS surfaces after writer quiescence."""

        geometry, worldtubes = _trajectory_geometry(
            Path(trajectory_path), self.path
        )
        self.manifest["geometry"] = geometry
        self.manifest["worldtubes"] = worldtubes
        by_step: dict[int, list[str]] = {}
        for item in geometry:
            by_step.setdefault(int(item["step"]), []).append(str(item["id"]))
        for frame in self.manifest["frames"]:
            frame["geometry"] = by_step.get(int(frame["step"]), [])
        validate_scene(self.manifest)
        _write_json(self.path / "scene.json", self.manifest)


def _finite_json(value: Any) -> Any:
    """Convert numpy values and non-finite diagnostics into strict JSON."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        json.dumps(
            _finite_json(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def file_digest(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_scene(scene: dict[str, Any]) -> None:
    """Validate the semantic subset required by all Observatory readers."""

    if scene.get("schema") != SCENE_SCHEMA:
        raise SceneContractError(f"schema must be {SCENE_SCHEMA!r}")
    if scene.get("format_version") != SCENE_FORMAT_VERSION:
        raise SceneContractError("unsupported scene format version")
    for key in ("scene_id", "title", "coordinates", "frames", "provenance"):
        if key not in scene:
            raise SceneContractError(f"scene is missing {key!r}")
    coordinates = scene["coordinates"]
    if not isinstance(coordinates, dict):
        raise SceneContractError("coordinates must be an object")
    axes = coordinates.get("axes")
    if not isinstance(axes, list) or len(axes) != 3:
        raise SceneContractError("coordinates.axes must contain x, y, z")
    names = [axis.get("name") for axis in axes if isinstance(axis, dict)]
    if names != ["x", "y", "z"]:
        raise SceneContractError("scene axes must be ordered x, y, z")
    frames = scene["frames"]
    if not isinstance(frames, list) or not frames:
        raise SceneContractError("scene needs at least one frame")
    for frame in frames:
        if not isinstance(frame, dict):
            raise SceneContractError("each frame must be an object")
        if not isinstance(frame.get("step"), int):
            raise SceneContractError("frame step must be an integer")
        if not isinstance(frame.get("time"), (int, float)):
            raise SceneContractError("frame time must be numeric")
        fields = frame.get("fields")
        if not isinstance(fields, list) or not fields:
            raise SceneContractError("each frame needs fields")
        field_ids: set[str] = set()
        for field in fields:
            if not isinstance(field, dict):
                raise SceneContractError("field records must be objects")
            required = {
                "id",
                "name",
                "symbol",
                "tensor_type",
                "centering",
                "data_type",
                "storage",
                "multiscales",
                "derivation",
                "accuracy",
            }
            missing = required - set(field)
            if missing:
                raise SceneContractError(
                    f"field is missing {sorted(missing)}"
                )
            identifier = field["id"]
            if not isinstance(identifier, str) or not identifier:
                raise SceneContractError("field id must be non-empty")
            if identifier in field_ids:
                raise SceneContractError(f"duplicate field id {identifier!r}")
            field_ids.add(identifier)
            levels = field["multiscales"]
            if not isinstance(levels, list) or not levels:
                raise SceneContractError("field needs multiscale levels")
            previous = None
            for level in levels:
                shape = level.get("shape")
                chunk = level.get("chunk_shape")
                if (
                    not isinstance(shape, list)
                    or len(shape) != 3
                    or any(not isinstance(v, int) or v < 1 for v in shape)
                ):
                    raise SceneContractError("invalid multiscale shape")
                if (
                    not isinstance(chunk, list)
                    or len(chunk) != 3
                    or any(not isinstance(v, int) or v < 1 for v in chunk)
                ):
                    raise SceneContractError("invalid chunk shape")
                if previous is not None and any(
                    now > old for now, old in zip(shape, previous)
                ):
                    raise SceneContractError(
                        "multiscale shapes must be non-increasing"
                    )
                previous = shape


def load_scene(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        source = source / "scene.json"
    scene = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(scene, dict):
        raise SceneContractError("scene root must be an object")
    validate_scene(scene)
    return scene


def _downsample_mean(array: np.ndarray) -> np.ndarray:
    pads = [(0, value % 2) for value in array.shape]
    padded = np.pad(array, pads, mode="edge")
    nx, ny, nz = padded.shape
    return padded.reshape(nx // 2, 2, ny // 2, 2, nz // 2, 2).mean(
        axis=(1, 3, 5), dtype=np.float64
    ).astype(np.float32)


def multiscale_pyramid(
    array: np.ndarray, maximum_levels: int = 4
) -> list[np.ndarray]:
    value = np.ascontiguousarray(array, dtype=np.float32)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError("visualization fields must be finite 3D arrays")
    result = [value]
    while len(result) < maximum_levels and min(result[-1].shape) > 16:
        result.append(_downsample_mean(result[-1]))
    return result


def _write_zarr_array(
    root: Path,
    array: np.ndarray,
    *,
    chunk_shape: tuple[int, int, int],
    attributes: dict[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(value) for value in array.shape)
    actual_chunk = tuple(
        min(int(requested), size)
        for requested, size in zip(chunk_shape, shape)
    )
    metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(shape),
        "data_type": "float32",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(actual_chunk)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0.0,
        "codecs": [
            {
                "name": "bytes",
                "configuration": {"endian": "little"},
            }
        ],
        "attributes": _finite_json(attributes),
    }
    _write_json(root / "zarr.json", metadata)
    chunk_counts = tuple(
        (size + chunk - 1) // chunk
        for size, chunk in zip(shape, actual_chunk)
    )
    for ix in range(chunk_counts[0]):
        for iy in range(chunk_counts[1]):
            for iz in range(chunk_counts[2]):
                starts = (
                    ix * actual_chunk[0],
                    iy * actual_chunk[1],
                    iz * actual_chunk[2],
                )
                stops = tuple(
                    min(start + chunk, size)
                    for start, chunk, size in zip(
                        starts, actual_chunk, shape
                    )
                )
                chunk = np.ascontiguousarray(
                    array[
                        starts[0] : stops[0],
                        starts[1] : stops[1],
                        starts[2] : stops[2],
                    ],
                    dtype="<f4",
                )
                target = root / "c" / str(ix) / str(iy) / str(iz)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(chunk.tobytes(order="C"))
    return {
        "shape": list(shape),
        "chunk_shape": list(actual_chunk),
        "chunk_counts": list(chunk_counts),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _ensure_zarr_group(
    root: Path, attributes: dict[str, Any] | None = None
) -> None:
    metadata = root / "zarr.json"
    if not metadata.is_file():
        _write_json(
            metadata,
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": _finite_json(attributes or {}),
            },
        )


def _field_contract(
    identifier: str,
    name: str,
    symbol: str,
    levels: list[dict[str, Any]],
    *,
    derivation: str,
    signed: bool,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "name": name,
        "symbol": symbol,
        "tensor_type": "scalar",
        "components": [],
        "centering": "cell",
        "basis": "coordinate",
        "index_positions": [],
        "data_type": "float32",
        "storage": "zarr_v3_raw_chunks",
        "multiscales": levels,
        "display": {
            "signed": signed,
            "recommended_transform": "signed_log" if signed else "linear",
            "recommended_colormap": (
                "blue_white_red" if signed else "inferno"
            ),
        },
        "derivation": {
            "kind": "derived_observable",
            "expression": derivation,
            "implementation": "tesseract_nr.scene",
        },
        "accuracy": {
            "source_precision": "float32",
            "storage_precision": "float32",
            "lossy": False,
            "downsampling": "2x2x2 arithmetic mean",
        },
    }


def _surface_mesh(
    modes: Iterable[Iterable[int]],
    coefficients: Iterable[float],
    center: Iterable[float],
    *,
    theta_count: int = 48,
    phi_count: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, pi, theta_count, dtype=float)
    phi = np.linspace(0.0, 2.0 * pi, phi_count, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    mode_values = evaluate_basis(
        tuple((int(ell), int(order)) for ell, order in modes),
        theta_grid,
        phi_grid,
    )
    radius = np.exp(
        np.einsum(
            "mij,m->ij",
            mode_values,
            np.asarray(tuple(coefficients), dtype=float),
        )
    )
    sine = np.sin(theta_grid)
    directions = np.stack(
        (
            sine * np.cos(phi_grid),
            sine * np.sin(phi_grid),
            np.cos(theta_grid),
        ),
        axis=-1,
    )
    vertices = (
        np.asarray(tuple(center), dtype=float)[None, None, :]
        + radius[..., None] * directions
    ).reshape(-1, 3)
    triangles: list[tuple[int, int, int]] = []
    for row in range(theta_count - 1):
        for column in range(phi_count):
            following = (column + 1) % phi_count
            a = row * phi_count + column
            b = row * phi_count + following
            c = (row + 1) * phi_count + column
            d = (row + 1) * phi_count + following
            triangles.extend(((a, c, b), (b, c, d)))
    return (
        np.asarray(vertices, dtype="<f4"),
        np.asarray(triangles, dtype="<u4"),
    )


def _trajectory_geometry(
    trajectory_path: Path | None,
    target: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if trajectory_path is None or not trajectory_path.is_file():
        return [], []
    records = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("trajectory evidence must be an array")
    geometry: list[dict[str, Any]] = []
    tubes: dict[str, list[dict[str, Any]]] = {}
    geometry_root = target / "geometry"
    for frame_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        branches = record.get("common", {})
        if not isinstance(branches, dict):
            continue
        for branch_name, branch in branches.items():
            if not isinstance(branch, dict):
                continue
            modes = branch.get("modes")
            coefficients = branch.get("coefficients")
            center = branch.get("center")
            if not (
                isinstance(modes, list)
                and isinstance(coefficients, list)
                and isinstance(center, list)
                and len(center) == 3
                and len(modes) == len(coefficients)
            ):
                continue
            vertices, indices = _surface_mesh(
                modes, coefficients, center
            )
            identifier = (
                f"mots-{branch_name}-step-{int(record.get('step', 0)):07d}"
            )
            mesh_root = geometry_root / identifier
            mesh_root.mkdir(parents=True, exist_ok=True)
            (mesh_root / "vertices.f32").write_bytes(
                vertices.tobytes(order="C")
            )
            (mesh_root / "indices.u32").write_bytes(
                indices.tobytes(order="C")
            )
            item = {
                "id": identifier,
                "kind": "mots_surface",
                "branch": branch_name,
                "step": int(record.get("step", 0)),
                "time": float(record.get("time", 0.0)),
                "found": bool(branch.get("found", False)),
                "candidate": not bool(branch.get("found", False)),
                "vertices": {
                    "path": f"geometry/{identifier}/vertices.f32",
                    "count": int(vertices.shape[0]),
                    "data_type": "float32",
                    "components": 3,
                },
                "indices": {
                    "path": f"geometry/{identifier}/indices.u32",
                    "count": int(indices.size),
                    "data_type": "uint32",
                },
                "extraction": {
                    "representation": "log-radius real spherical harmonics",
                    "spectral_residual_norm": branch.get(
                        "spectral_residual_norm"
                    ),
                    "maximum_expansion": branch.get("maximum_expansion"),
                    "complementarity_residual": branch.get(
                        "complementarity_residual"
                    ),
                },
            }
            geometry.append(item)
            tubes.setdefault(branch_name, []).append(
                {
                    "step": item["step"],
                    "time": item["time"],
                    "surface": identifier,
                    "found": item["found"],
                }
            )
    worldtubes = [
        {
            "id": f"worldtube-{branch}",
            "kind": "mots_worldtube",
            "branch": branch,
            "samples": samples,
        }
        for branch, samples in sorted(tubes.items())
        if samples
    ]
    return geometry, worldtubes


def _checkpoint_fields(
    snapshot: VulkanCCZ4HostSnapshot,
    *,
    include_constraints: bool,
) -> list[tuple[str, str, str, np.ndarray, str, bool]]:
    shape = snapshot.layout.shape
    packed = snapshot.ccz4_state

    def field(column: int) -> np.ndarray:
        return np.asarray(packed[:, column], dtype=np.float32).reshape(
            shape, order="F"
        )

    result = [
        (
            "ccz4.lapse",
            "Lapse",
            "alpha",
            field(24),
            "resident CCZ4 state column 24",
            False,
        ),
        (
            "ccz4.conformal_factor",
            "Conformal factor",
            "chi",
            field(18),
            "resident CCZ4 state column 18",
            False,
        ),
        (
            "ccz4.trace_k",
            "Trace of extrinsic curvature",
            "K",
            field(19),
            "resident CCZ4 state column 19",
            True,
        ),
        (
            "ccz4.theta",
            "CCZ4 cleaning field",
            "Theta",
            field(20),
            "resident CCZ4 state column 20",
            True,
        ),
    ]
    shift = np.sqrt(sum(field(column) ** 2 for column in range(25, 28)))
    result.append(
        (
            "ccz4.shift_magnitude",
            "Shift magnitude",
            "|beta|",
            shift.astype(np.float32),
            "Euclidean display magnitude of CCZ4 columns 25:28",
            False,
        )
    )
    if include_constraints:
        from .ccz4 import CCZ4Solver
        from .grid import PeriodicGrid

        grid = PeriodicGrid(
            shape,
            snapshot.layout.lengths,
            method=f"fd{snapshot.layout.derivative_order}",
        )
        state = unpack_ccz4_state(
            snapshot.ccz4_state, snapshot.layout, time=snapshot.time
        )
        hamiltonian, momentum = CCZ4Solver(grid).constraints(state)
        result.extend(
            (
                (
                    "constraints.hamiltonian",
                    "Hamiltonian constraint",
                    "H",
                    np.asarray(hamiltonian, dtype=np.float32),
                    (
                        "independent CPU ADM constraint from the explicit "
                        f"snapshot using fd{snapshot.layout.derivative_order}"
                    ),
                    True,
                ),
                (
                    "constraints.momentum_magnitude",
                    "Momentum-constraint magnitude",
                    "|M|",
                    np.sqrt(np.sum(momentum**2, axis=0)).astype(np.float32),
                    (
                        "independent CPU ADM momentum constraint magnitude "
                        f"using fd{snapshot.layout.derivative_order}"
                    ),
                    False,
                ),
            )
        )
    return result


def export_puncture_scene(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    trajectory_path: str | Path | None = None,
    configuration_path: str | Path | None = None,
    observer_path: str | Path | None = None,
    include_constraints: bool = False,
    maximum_levels: int = 4,
    chunk_shape: tuple[int, int, int] = DEFAULT_CHUNK_SHAPE,
) -> SceneExportResult:
    """Export a puncture CCZ4 checkpoint as a replayable ``.tessviz`` scene."""

    checkpoint = Path(checkpoint_path).resolve()
    output = Path(output_path).resolve()
    if output.suffix != ".tessviz":
        raise ValueError("scene output must use the .tessviz suffix")
    if output.exists():
        if not output.is_dir():
            raise ValueError("existing .tessviz output is not a directory")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    snapshot = load_vulkan_ccz4_checkpoint(checkpoint)
    trajectory = (
        Path(trajectory_path).resolve()
        if trajectory_path is not None
        else checkpoint.parent / "trajectory.json"
    )
    configuration = (
        Path(configuration_path).resolve()
        if configuration_path is not None
        else checkpoint.parent / "configuration.json"
    )
    observer = (
        Path(observer_path).resolve()
        if observer_path is not None
        else checkpoint.parent / "final_observer.json"
    )
    evidence_payload: dict[str, Any] = {}
    evidence_sources: list[dict[str, Any]] = []
    for name, source in (
        ("trajectory", trajectory),
        ("configuration", configuration),
        ("observer", observer),
    ):
        if source.is_file():
            evidence_payload[name] = json.loads(
                source.read_text(encoding="utf-8")
            )
            evidence_sources.append(
                {
                    "kind": name,
                    "source_path": str(source),
                    "sha256": file_digest(source),
                }
            )
    _write_json(output / "evidence.json", evidence_payload)
    _write_json(
        output / "fields.zarr" / "zarr.json",
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "schema": SCENE_SCHEMA,
                "coordinate_order": ["x", "y", "z"],
            },
        },
    )
    fields = []
    field_names = []
    for identifier, name, symbol, values, expression, signed in (
        _checkpoint_fields(
            snapshot, include_constraints=include_constraints
        )
    ):
        field_names.append(identifier)
        levels = []
        _ensure_zarr_group(
            output / "fields.zarr" / identifier,
            {"field_id": identifier},
        )
        for level_index, level in enumerate(
            multiscale_pyramid(values, maximum_levels)
        ):
            relative = (
                Path("fields.zarr")
                / identifier
                / f"level_{level_index}"
            )
            descriptor = _write_zarr_array(
                output / relative,
                level,
                chunk_shape=chunk_shape,
                attributes={
                    "field_id": identifier,
                    "level": level_index,
                    "axis_order": ["x", "y", "z"],
                    "centering": "cell",
                },
            )
            descriptor.update(
                {
                    "level": level_index,
                    "path": relative.as_posix(),
                    "scale": [2**level_index] * 3,
                }
            )
            levels.append(descriptor)
        fields.append(
            _field_contract(
                identifier,
                name,
                symbol,
                levels,
                derivation=expression,
                signed=signed,
            )
        )
    geometry, worldtubes = _trajectory_geometry(
        trajectory if trajectory.is_file() else None, output
    )
    layout = snapshot.layout
    axes = []
    for name, size, origin, spacing in zip(
        ("x", "y", "z"),
        layout.shape,
        layout.origins,
        layout.spacing,
    ):
        axes.append(
            {
                "name": name,
                "type": "space",
                "unit": "code_length",
                "size": size,
                "origin": origin,
                "spacing": spacing,
                "centering": "cell",
            }
        )
    checkpoint_hash = file_digest(checkpoint)
    manifest = {
        "schema": SCENE_SCHEMA,
        "format_version": SCENE_FORMAT_VERSION,
        "scene_id": f"puncture-{snapshot.steps}-{checkpoint_hash[:12]}",
        "title": (
            f"Tesseract puncture spacetime at step {snapshot.steps}"
        ),
        "description": (
            "Derived visualization of an explicit resident Vulkan CCZ4 "
            "checkpoint. Coordinate geometry is gauge and slicing dependent."
        ),
        "coordinates": {
            "chart": "cartesian",
            "handedness": "right",
            "axis_order": ["x", "y", "z"],
            "axes": axes,
            "gauge": {
                "slicing": "1+log",
                "shift": "Gamma-driver",
                "coordinate_warning": (
                    "Rendered distances are coordinate distances, not unique "
                    "gauge-invariant spacetime observables."
                ),
            },
        },
        "frames": [
            {
                "id": f"step-{snapshot.steps:07d}",
                "step": snapshot.steps,
                "time": snapshot.time,
                "time_unit": "code_time",
                "fields": fields,
                "geometry": [
                    item["id"]
                    for item in geometry
                    if item["step"] == snapshot.steps
                ],
            }
        ],
        "geometry": geometry,
        "worldtubes": worldtubes,
        "evidence": {
            "path": "evidence.json",
            "sources": evidence_sources,
        },
        "provenance": {
            "producer": "Vulkan Tesseract Observatory exporter",
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_schema": (
                "vulkan-tesseract.ccz4-vacuum-checkpoint.v1"
            ),
            "source_checkpoint_sha256": checkpoint_hash,
            "source_precision": "float32",
            "derivative_order": layout.derivative_order,
            "derived_product": True,
            "restart_capable": False,
        },
    }
    validate_scene(manifest)
    _write_json(output / "scene.json", manifest)
    return SceneExportResult(
        output,
        manifest,
        tuple(field_names),
        len(geometry),
    )


def _binary_interface_mesh(
    phase: np.ndarray,
    *,
    origins: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Create an explicit cell-face mesh where a hard phase label changes."""

    labels = np.asarray(phase, dtype=bool)
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    shape = labels.shape
    for axis in range(3):
        difference = labels != np.roll(labels, -1, axis=axis)
        boundary = np.argwhere(difference)
        for cell in boundary:
            if cell[axis] == shape[axis] - 1:
                continue
            coordinates = [
                origins[index] + (cell[index] + 0.5) * spacing[index]
                for index in range(3)
            ]
            coordinates[axis] += 0.5 * spacing[axis]
            transverse = [index for index in range(3) if index != axis]
            corners = []
            for first, second in (
                (-0.5, -0.5),
                (0.5, -0.5),
                (0.5, 0.5),
                (-0.5, 0.5),
            ):
                point = list(coordinates)
                point[transverse[0]] += first * spacing[transverse[0]]
                point[transverse[1]] += second * spacing[transverse[1]]
                corners.append(tuple(point))
            base = len(vertices)
            vertices.extend(corners)
            triangles.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
    return (
        np.asarray(vertices, dtype="<f4").reshape(-1, 3),
        np.asarray(triangles, dtype="<u4").reshape(-1, 3),
    )


def _characteristic_ribbons(
    vector: np.ndarray,
    *,
    origins: tuple[float, float, float],
    lengths: tuple[float, float, float],
    seeds_per_axis: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace short nearest-cell characteristic ribbons for communication."""

    field = np.asarray(vector, dtype=float)
    shape = field.shape[1:]
    spacing = tuple(length / count for length, count in zip(lengths, shape))
    minimum_spacing = min(spacing)
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    seeds = [
        np.linspace(0, size - 1, seeds_per_axis, dtype=int)
        for size in shape
    ]
    for ix in seeds[0]:
        for iy in seeds[1]:
            for iz in seeds[2]:
                position = np.array(
                    [
                        origins[axis]
                        + (index + 0.5) * spacing[axis]
                        for axis, index in enumerate((ix, iy, iz))
                    ],
                    dtype=float,
                )
                for _ in range(10):
                    index = tuple(
                        int(
                            np.clip(
                                (position[axis] - origins[axis])
                                / spacing[axis],
                                0,
                                shape[axis] - 1,
                            )
                        )
                        for axis in range(3)
                    )
                    direction = field[(slice(None),) + index]
                    norm = float(np.linalg.norm(direction))
                    if not np.isfinite(norm) or norm < 1.0e-8:
                        break
                    direction = direction / norm
                    following = position + direction * minimum_spacing * 1.4
                    if any(
                        following[axis] < origins[axis]
                        or following[axis] > origins[axis] + lengths[axis]
                        for axis in range(3)
                    ):
                        break
                    reference = (
                        np.array([0.0, 0.0, 1.0])
                        if abs(direction[2]) < 0.9
                        else np.array([0.0, 1.0, 0.0])
                    )
                    normal = np.cross(direction, reference)
                    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
                    offset = normal * minimum_spacing * 0.055
                    base = len(vertices)
                    vertices.extend(
                        (
                            position - offset,
                            position + offset,
                            following - offset,
                            following + offset,
                        )
                    )
                    triangles.extend(
                        (
                            (base, base + 2, base + 1),
                            (base + 1, base + 2, base + 3),
                        )
                    )
                    position = following
    return (
        np.asarray(vertices, dtype="<f4").reshape(-1, 3),
        np.asarray(triangles, dtype="<u4").reshape(-1, 3),
    )


def _write_mesh(
    output: Path,
    *,
    identifier: str,
    kind: str,
    vertices: np.ndarray,
    indices: np.ndarray,
    step: int,
    time: float,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    if vertices.size == 0 or indices.size == 0:
        return None
    root = output / "geometry" / identifier
    root.mkdir(parents=True, exist_ok=True)
    (root / "vertices.f32").write_bytes(
        np.asarray(vertices, dtype="<f4").tobytes(order="C")
    )
    (root / "indices.u32").write_bytes(
        np.asarray(indices, dtype="<u4").tobytes(order="C")
    )
    return {
        "id": identifier,
        "kind": kind,
        "step": int(step),
        "time": float(time),
        "vertices": {
            "path": f"geometry/{identifier}/vertices.f32",
            "count": int(vertices.shape[0]),
            "data_type": "float32",
            "components": 3,
        },
        "indices": {
            "path": f"geometry/{identifier}/indices.u32",
            "count": int(indices.size),
            "data_type": "uint32",
        },
        "extraction": _finite_json(metadata),
    }


def export_evolved_scene(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    maximum_levels: int = 4,
    chunk_shape: tuple[int, int, int] = DEFAULT_CHUNK_SHAPE,
) -> SceneExportResult:
    """Export a full Carter–Proca–CCZ4 checkpoint, including neural layers."""

    from .frozen_closure import QualifiedFrozenClosure
    from .vulkan_mots import VulkanCCZ4Layout

    checkpoint = Path(checkpoint_path).resolve()
    output = Path(output_path).resolve()
    if output.suffix != ".tessviz":
        raise ValueError("scene output must use the .tessviz suffix")
    if output.exists():
        if not output.is_dir():
            raise ValueError("existing .tessviz output is not a directory")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    with np.load(checkpoint, allow_pickle=False) as archive:
        schema = str(archive["schema"])
        if schema != "vulkan-tesseract.evolved-checkpoint.v1":
            raise ValueError("full scene needs an evolved Tesseract checkpoint")
        shape = tuple(int(value) for value in archive["shape"])
        lengths = tuple(float(value) for value in archive["lengths"])
        origins = tuple(float(value) for value in archive["origins"])
        derivative_order = int(archive["derivative_order"])
        primitive = np.asarray(archive["primitive"], dtype=np.float32)
        matter_state = np.asarray(archive["matter_state"], dtype=np.float32)
        ccz4 = np.asarray(archive["ccz4_state"], dtype=np.float32)
        constitutive_model = str(archive["constitutive_model"])
        reconstruction = str(archive["reconstruction"])
        time = float(archive["time"])
        step = int(archive["steps"])
    points = int(np.prod(shape))
    if primitive.shape != (points, 9) or matter_state.shape != (points, 11):
        raise ValueError("evolved checkpoint matter ABI is invalid")
    layout = VulkanCCZ4Layout(
        shape, lengths, origins, derivative_order
    )
    snapshot = VulkanCCZ4HostSnapshot(
        ccz4, layout, time, step, source=f"checkpoint:{checkpoint}"
    )
    field_values = list(
        _checkpoint_fields(snapshot, include_constraints=False)
    )
    n = np.exp(np.asarray(primitive[:, 0], dtype=float))
    entropy = np.asarray(primitive[:, 1], dtype=float)
    q_n = np.asarray(primitive[:, 2:5], dtype=float)
    d = np.exp(np.asarray(primitive[:, 5], dtype=float))
    q_d = np.asarray(primitive[:, 6:9], dtype=float)
    gt = ccz4[:, :9].reshape(points, 3, 3)
    chi = np.maximum(ccz4[:, 18], 1.0e-12)
    metric = gt / chi[:, None, None] ** 2
    qn_squared = np.einsum("pi,pij,pj->p", q_n, metric, q_n)
    qd_squared = np.einsum("pi,pij,pj->p", q_d, metric, q_d)
    q_cross = np.einsum("pi,pij,pj->p", q_n, metric, q_d)
    w_n = np.sqrt(np.maximum(1.0 + qn_squared, 1.0))
    w_d = np.sqrt(np.maximum(1.0 + qd_squared, 1.0))
    relative_gamma = np.maximum(w_n * w_d - q_cross, 1.0)
    velocity_difference = (
        q_n / w_n[:, None] - q_d / w_d[:, None]
    )
    field_values.extend(
        (
            (
                "matter.baryon_density",
                "Baryon rest density",
                "n",
                n.reshape(shape, order="F").astype(np.float32),
                "exp(curved primitive column 0)",
                False,
            ),
            (
                "matter.carrier_density",
                "Carrier rest density",
                "d",
                d.reshape(shape, order="F").astype(np.float32),
                "exp(curved primitive column 5)",
                False,
            ),
            (
                "matter.entropy",
                "Specific entropy",
                "s",
                entropy.reshape(shape, order="F").astype(np.float32),
                "curved primitive column 1",
                False,
            ),
            (
                "matter.relative_lorentz_excess",
                "Relative Lorentz-factor excess",
                "gamma_rel-1",
                (relative_gamma - 1.0).reshape(shape, order="F").astype(
                    np.float32
                ),
                "curved-metric Carter relative Lorentz factor minus one",
                False,
            ),
            (
                "matter.characteristic_drift_magnitude",
                "Counterflow characteristic magnitude",
                "|v_N-v_D|",
                np.linalg.norm(velocity_difference, axis=1)
                .reshape(shape, order="F")
                .astype(np.float32),
                "coordinate components of q_N/W_N-q_D/W_D",
                False,
            ),
        )
    )
    phase = None
    switching_margin = None
    model_digest = None
    if constitutive_model == "frozen_neural":
        closure = QualifiedFrozenClosure(
            Path(__file__).resolve().parents[1]
            / "theory33_frozen_variants.json"
        )
        response = closure.evaluate(
            n * n,
            d * d,
            n * d * relative_gamma,
            entropy,
        )
        phase = np.asarray(response.phase_two, dtype=bool).reshape(
            shape, order="F"
        )
        switching_margin = (
            relative_gamma
            - 1.0
            - np.asarray(response.phase_threshold)
        ).reshape(shape, order="F")
        model_digest = closure.variant_digest
        field_values.extend(
            (
                (
                    "neural.switching_margin",
                    "Learned constitutive switching margin",
                    "m_phi",
                    switching_margin.astype(np.float32),
                    "gamma_rel-1 minus digest-locked learned threshold",
                    True,
                ),
                (
                    "neural.phase_two",
                    "Learned constitutive phase",
                    "phase_2",
                    phase.astype(np.float32),
                    "hard digest-locked frozen-neural classifier decision",
                    False,
                ),
            )
        )
    _write_json(
        output / "fields.zarr" / "zarr.json",
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"schema": SCENE_SCHEMA},
        },
    )
    fields = []
    field_names = []
    for identifier, name, symbol, values, expression, signed in field_values:
        field_names.append(identifier)
        levels = []
        _ensure_zarr_group(
            output / "fields.zarr" / identifier,
            {"field_id": identifier},
        )
        for level_index, level in enumerate(
            multiscale_pyramid(values, maximum_levels)
        ):
            relative = (
                Path("fields.zarr")
                / identifier
                / f"level_{level_index}"
            )
            descriptor = _write_zarr_array(
                output / relative,
                level,
                chunk_shape=chunk_shape,
                attributes={
                    "field_id": identifier,
                    "level": level_index,
                    "axis_order": ["x", "y", "z"],
                },
            )
            descriptor.update(
                {
                    "level": level_index,
                    "path": relative.as_posix(),
                    "scale": [2**level_index] * 3,
                }
            )
            levels.append(descriptor)
        fields.append(
            _field_contract(
                identifier,
                name,
                symbol,
                levels,
                derivation=expression,
                signed=signed,
            )
        )
    geometry: list[dict[str, Any]] = []
    if phase is not None:
        vertices, indices = _binary_interface_mesh(
            phase, origins=origins, spacing=layout.spacing
        )
        item = _write_mesh(
            output,
            identifier=f"neural-phase-front-step-{step:07d}",
            kind="neural_phase_front",
            vertices=vertices,
            indices=indices,
            step=step,
            time=time,
            metadata={
                "isovalue": 0.0,
                "source": "hard learned phase-label crossings",
                "model_digest": model_digest,
            },
        )
        if item is not None:
            geometry.append(item)
    vector_grid = np.stack(
        [
            velocity_difference[:, axis].reshape(shape, order="F")
            for axis in range(3)
        ],
        axis=0,
    )
    vertices, indices = _characteristic_ribbons(
        vector_grid, origins=origins, lengths=lengths
    )
    item = _write_mesh(
        output,
        identifier=f"carter-characteristics-step-{step:07d}",
        kind="characteristic_geometry",
        vertices=vertices,
        indices=indices,
        step=step,
        time=time,
        metadata={
            "source": "short integral ribbons of q_N/W_N-q_D/W_D",
            "interpretation": (
                "counterflow direction proxy, not a complete null geodesic"
            ),
        },
    )
    if item is not None:
        geometry.append(item)
    axes = [
        {
            "name": name,
            "type": "space",
            "unit": "code_length",
            "size": count,
            "origin": origin,
            "spacing": spacing,
            "centering": "cell",
        }
        for name, count, origin, spacing in zip(
            ("x", "y", "z"), shape, origins, layout.spacing
        )
    ]
    digest = file_digest(checkpoint)
    manifest = {
        "schema": SCENE_SCHEMA,
        "format_version": SCENE_FORMAT_VERSION,
        "scene_id": f"evolved-{step}-{digest[:12]}",
        "title": f"Full neural Carter–CCZ4 spacetime at step {step}",
        "description": (
            "Derived replay of the coupled Carter, frozen-neural, Proca and "
            "CCZ4 state. Coordinate geometry remains gauge dependent."
        ),
        "coordinates": {
            "chart": "cartesian",
            "handedness": "right",
            "axis_order": ["x", "y", "z"],
            "axes": axes,
            "gauge": {
                "slicing": "1+log",
                "shift": "Gamma-driver",
                "coordinate_warning": (
                    "Rendered distances are coordinate distances; learned "
                    "phase surfaces are constitutive decisions on this slice."
                ),
            },
        },
        "frames": [
            {
                "id": f"step-{step:07d}",
                "step": step,
                "time": time,
                "time_unit": "code_time",
                "fields": fields,
                "geometry": [item["id"] for item in geometry],
            }
        ],
        "geometry": geometry,
        "worldtubes": [],
        "evidence": {"path": "evidence.json", "sources": []},
        "provenance": {
            "producer": "Vulkan Tesseract full-stack scene exporter",
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_schema": schema,
            "source_checkpoint_sha256": digest,
            "source_precision": "float32",
            "derivative_order": derivative_order,
            "constitutive_model": constitutive_model,
            "constitutive_model_digest": model_digest,
            "reconstruction": reconstruction,
            "derived_product": True,
            "restart_capable": False,
        },
    }
    _write_json(output / "evidence.json", {})
    validate_scene(manifest)
    _write_json(output / "scene.json", manifest)
    return SceneExportResult(
        output, manifest, tuple(field_names), len(geometry)
    )
