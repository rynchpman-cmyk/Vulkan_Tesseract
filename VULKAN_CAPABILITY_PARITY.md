# Original Tesseract to Vulkan Tesseract Capability Alignment

## Result

Vulkan Tesseract is capability-aligned with the original Tesseract
computational project.

That statement is backed by
[`vulkan_parity_results.json`](vulkan_parity_results.json), not inferred from
the presence of copied files. The manifest currently requires:

- every declared capability surface and evidence artifact to exist;
- every resident Vulkan qualification artifact to pass all of its gates;
- the aggregate record to carry SHA-256 digests for every production Vulkan
  implementation file;
- all retained training, oracle, diagnostic, I/O, and visualization surfaces
  to remain available; and
- every computational `.py`, `.slang`, and `.comp` source, plus the qualified
  kinetic/noisy-trajectory input data, in the original `TESSERACT` tree to
  have a byte-identical counterpart here.

The source audit compares 75 files. Its purpose is to prevent “alignment” from
silently dropping an original workflow while the resident backend is being
extended.

## Meaning of 100% alignment

Capability alignment does not mean that every utility is a compute shader.
The execution boundary is explicit:

| Surface | Execution mode |
|---|---|
| Carter/KKT recurrence and reverse-mode AD | Retained original Slang/Vulkan workflows |
| Production Carter, neural closure, Proca, CCZ4, constrained data, boundaries, AMR primitives, and strong-field evolution | Resident Vulkan backend |
| Neural training, uncertainty, mobility, and event-BPTT studies | Retained optional PyTorch workflow |
| Independent physics oracle and convergence diagnostics | Retained float64 CPU workflow |
| Checkpoint orchestration, reporting, and visualization | Host control and postprocessing |
| MPI/domain decomposition from the original engine | Retained CPU orchestration; multi-GPU Vulkan decomposition is a separate scaling project |

This is the same division used by production accelerator codes: the evolving
physics state and numerical operators live on the device; setup, training,
independent verification, I/O, and visualization remain host workflows where
that is the correct role.

## Capability matrix

| Original capability | Vulkan Tesseract status | Primary evidence |
|---|---|---|
| Literal Carter/KKT recurrence | Retained and live-qualified in Slang/Vulkan | `vulkan_original_workflows_results.json` |
| Reverse-mode recurrence AD and selected KKT VJP | Retained and live-qualified in Slang/Vulkan | `vulkan_original_workflows_results.json` |
| Scale-1, 128-step Lyapunov-conditioned literal map | Retained and audit-qualified | `vulkan_original_workflows_results.json` |
| Standalone Theory 3.3 timelike-current hybrid and native constitutive reverse pass | Retained and live-qualified in Slang/Vulkan | `vulkan_original_workflows_results.json` |
| Theory 3.3 analytic-M1 hybrid | Resident-qualified | `vulkan_theory33_results.json` |
| Frozen neural constitutive replay and hard switching | Resident-qualified | `vulkan_frozen_neural_results.json` |
| Independent data, PSD mobility, noisy-threshold uncertainty, and event-BPTT estimators | Retained qualified training/experiment workflow | `theory33_advanced_results.json`, `theory33_experiment_results.json` |
| Curved nine-variable primitive recovery | Resident-qualified | `vulkan_full_carter_ccz4_results.json` |
| Eleven-field Carter transport with entropy and tracer | Resident-qualified in 1D, 2D, and 3D | `vulkan_frontier_results.json` |
| Complete numerical Carter flux/path eigensystem | Resident-qualified in every active direction | `vulkan_carter_eigensystem_results.json`, `vulkan_full_carter_ccz4_results.json` |
| Full Carter Roe plus DLM path fluctuation | Resident-qualified | `vulkan_original_neural_phase_full_roe_results.json` |
| Component and characteristic WENO5-Z | Resident-qualified | `vulkan_weno_results.json`, `vulkan_full_carter_weno_results.json` |
| Positivity-preserving conservative flux blending | Resident-qualified | `vulkan_weno_results.json` |
| Analytic and learned-phase Riemann ladders | Resident-qualified at 16/32/64/128 | `vulkan_carter_riemann_results.json`, `vulkan_original_neural_phase_full_roe_results.json` |
| Mixed-vector stress and drag/source coupling | Resident-qualified | `vulkan_mixed_source_results.json` |
| Evolved mixed-Proca fields, magnetic reconstruction, and Gauss cleaning | Resident-qualified | `vulkan_mixed_proca_results.json` |
| Derivative-containing CCZ4 vacuum and matter operator | Resident-qualified with FD2 and FD4 | `vulkan_multidimensional_results.json`, `vulkan_ccz4_fd4_results.json` |
| Periodic constrained active-neural CMC/Picard data | Resident fp64-qualified | `vulkan_constrained_results.json` |
| 128-step constrained analytic and learned horizons | Resident-qualified | `vulkan_frontier_results.json` |
| 2D continuum ladder and 3D active-neural evolution | Resident-qualified | `vulkan_frontier_results.json` |
| Physical Carter, CCZ4, and Proca boundaries | Resident-qualified with FD2 and FD4 boundary stencils | `vulkan_boundary_results.json` |
| AMR prolongation, restriction, Berger–Colella reflux, and Gauss projection | Resident-qualified in 1D, 2D, and 3D | `vulkan_amr_results.json` |
| TOV construction and Cartesian strong-field sampling | Resident fp64-qualified | `vulkan_tov_results.json` |
| Isolated CTT constraint solve | Resident fp64-qualified | `vulkan_isolated_ctt_results.json` |
| Complete Carter–Proca–CCZ4 checkpoint/restart | Resident-qualified with deterministic continuation | `vulkan_tov_results.json` |
| CPU reference solver, diagnostics, I/O, MPI orchestration, and visualization | Retained | `tesseract_nr/`, `run_tesseract_nr.py`, `tesseract_full_stack_visualizer.py` |
| Original GLSL compute shader and interactive recurrence microscope | Retained and runnable | `run.ps1`, `run_visualization.ps1 -Mode literal` |

## Original neural-phase Riemann reproduction

The original and resident campaigns use the same resolutions, final time,
CFL, literal primitive state, full nine-field Carter Roe interface, DLM path,
and frozen variant digest:

`e3a8e89ca635cfe68d5bd963152097f85ec47b7960819830c9513f45b7bcf108`

| Quantity | Original Tesseract | Resident Vulkan |
|---|---:|---:|
| Arithmetic | float64 | float32 |
| Phase changes at 16/32/64/128 | 0 / 0 / 0 / 4 | 0 / 0 / 0 / 4 |
| Recovery failures | 0 | 0 |
| Learned branches represented | yes | yes |
| Dynamic phase crossing | yes | yes |
| Evolution-stage host readbacks | not applicable | 0 |

The complete reproduction is
[`vulkan_original_neural_phase_full_roe_results.json`](vulkan_original_neural_phase_full_roe_results.json).
The earlier five-field control remains available as
[`vulkan_original_neural_phase_roe_results.json`](vulkan_original_neural_phase_roe_results.json).

## Numerical precision boundary

The production evolved ABI is float32. Constraint construction for periodic
CMC, isolated CTT, and TOV recurrence uses fp64 Vulkan kernels. Independent
diagnostics use the original float64 engine.

At the very short constrained micro-horizon, subtractive cancellation makes a
three-level float32 energy-rate order unreliable, so that campaign gates the
absolute energy balance. The resolved 2D frontier independently measures
second-order geometric-work convergence with observed order `2.192645`.
This distinction is encoded in the tracked artifacts.

## Reproduce the audit

```powershell
.\run_vulkan_parity.ps1
```

The command regenerates the aggregate manifest and compares the local
computational source copy with `..\TESSERACT`.

## Research frontiers, not parity gaps

The remaining work is expansion beyond the original project: longer and more
violent strong-field ladders, broader shock-accessible neural training,
higher-order DLM quadrature, kinetic or microphysical reference data, and
multi-GPU Vulkan domain decomposition. None of those is represented as an
already-qualified capability.
