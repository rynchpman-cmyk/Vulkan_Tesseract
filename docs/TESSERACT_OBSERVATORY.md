# Tesseract Spacetime Observatory

The Observatory is a custom WebGPU scientific renderer for replaying and
watching Vulkan Tesseract evolutions. It is not a Matplotlib view and it does
not infer spatial fields from summary JSON. The renderer consumes explicitly
exported field volumes and geometric objects while retaining links to the
qualification and provenance evidence.

## Visual truth contract

Every scene uses `tesseract.scene.v1`, defined by
[`docs/schemas/tesseract.scene.v1.schema.json`](schemas/tesseract.scene.v1.schema.json).
A `.tessviz` path is a directory containing:

```text
scene.tessviz/
|-- scene.json
|-- evidence.json
|-- fields.zarr/ or frames.zarr/
|   `-- <field>/<level>/zarr.json and c/<x>/<y>/<z>
`-- geometry/
    `-- <object>/vertices.f32 and indices.u32
```

The scene records:

- coordinate chart, handedness, axes, origin, spacing and centering;
- gauge and slicing warning;
- iteration and coordinate time;
- field tensor type, basis, index positions and derivation;
- source and storage precision plus downsampling policy;
- source-checkpoint and evidence hashes;
- geometric extraction residuals and whether a MOTS surface is a candidate or
  a qualified found surface; and
- the explicit fact that a scene is derived and cannot restart evolution.

Rendered distances are coordinate distances. The Observatory does not present
a spatial slice as a unique gauge-invariant picture of spacetime.

## Export an existing puncture checkpoint

```powershell
.\.venv-slang\Scripts\python.exe .\export_tessviz.py `
  .\runs\vulkan_ccz4_puncture_mots_n120_smoke\final_checkpoint.npz `
  .\runs\observatory\puncture-n120.tessviz `
  --levels 4 `
  --chunk-size 32
```

Add `--constraints` to derive independent CPU ADM Hamiltonian and momentum
constraint volumes. This is deliberately optional because it is an additional
full-resolution diagnostic calculation.

An evolved `vulkan-tesseract.evolved-checkpoint.v1` is detected automatically.
Its exporter adds Carter densities, entropy, relative drift, digest-locked
frozen-neural phase and switching margin when applicable, a phase-front mesh,
and short counterflow characteristic ribbons.

## Live observation

The production puncture runner enables the Observatory through:

```powershell
.\run_vulkan_puncture_mots.ps1 `
  -Resolution 120 `
  -Steps 500 `
  -Observatory `
  -VisualizationCadence 10 `
  -VisualizationMaximumAxis 64
```

The solver resolution is unchanged. The observation graph evaluates eight
compact fields directly from the resident CCZ4 state:

1. lapse;
2. conformal factor;
3. trace of extrinsic curvature;
4. CCZ4 cleaning field;
5. shift magnitude;
6. vacuum Hamiltonian constraint;
7. CCZ4 Z-constraint magnitude; and
8. conformal determinant residual.

The GPU kernel samples these onto a capped observation grid. A three-slot
device ring and one readback worker implement latest-frame backpressure.
Another bounded worker writes Zarr chunks and publishes `scene.json`
atomically. Evolution does not wait for either worker; frames can be dropped
and the scene records that policy. The runner waits only during orderly final
shutdown.

SlangPy owns Vulkan submission and synchronization. The implementation does
not claim application-managed timeline semaphores or generic Vulkan zero-copy
ADIOS2 transfer.

## Browser renderer

Tesseract Studio discovers `.tessviz` directories below `runs/`. The
Observatory renderer uses:

- a coarse 3D texture that is made fully resident first;
- a fine 3D texture populated one Zarr chunk at a time;
- a GPU storage-buffer page table identifying resident fine bricks;
- coarse sampling whenever the corresponding fine brick is absent;
- manual trilinear filtering of `r32float` textures;
- front-to-back volume ray marching;
- signed and unsigned transfer functions;
- interactive orbit, zoom, density, field and time controls;
- spectral MOTS, neural phase-front and characteristic geometry overlays;
- a central-plane numerical probe reading the underlying float32 chunk; and
- provenance/evidence readout.

This first implementation uses raw little-endian Zarr v3 chunks. Indexed
sharding and compressed worker-side decoding are compatible next extensions;
the present representation prioritizes a small auditable reader and exact
base-level float32 values.

## Source map

| Path | Responsibility |
|---|---|
| `tesseract_nr/scene.py` | Contract validation, Zarr writer, static/full-stack exporters and geometry |
| `tesseract_nr/vulkan_observatory.py` | Bounded GPU observation and asynchronous scene sinks |
| `tesseract_nr_gpu.slang` | Resident CCZ4 observation kernel |
| `export_tessviz.py` | Checkpoint export CLI |
| `studio/static/observatory.js` | WebGPU volume, residency and geometry renderer |
| `tesseract_studio.py` | Safe scene discovery and file API |

## Current boundaries

- Base volume chunks are uncompressed; Zarr indexed sharding is represented in
  the architecture but not yet used by the v1 writer.
- Fine-brick scheduling is center-first. Transfer-function and view-dependent
  priority are future optimizations.
- The numerical probe samples the central coordinate plane, not the first
  opacity hit along a rendered ray.
- The Hamiltonian observable is vacuum CCZ4. Coupled matter observations must
  include the matter energy source before being labeled a full coupled
  Hamiltonian constraint.
- Characteristic ribbons currently show the Carter counterflow direction
  proxy. They are not null geodesics or the complete numerical eigensystem.
- WebGPU rendering requires a browser and adapter that expose WebGPU.

These are renderer and transport frontiers; none alters the qualified solver
or checkpoint ABI.
