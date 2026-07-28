# Reproducibility and validation

## Environments

Vulkan Tesseract uses two Python environments.

CPU oracle and repository checks:

```powershell
python -m pip install "numpy>=2.3,<3" pytest
```

Slang/Vulkan runtime:

```powershell
python -m venv .venv-slang
.\.venv-slang\Scripts\python.exe -m pip install -r requirements-slang.txt
```

The GPU campaigns require Windows, a current Vulkan driver, and a
Vulkan-capable device. The CPU tests are platform-independent.

## Beginner-facing launcher

After creating `.venv-slang`, the same qualified campaigns can be launched
and monitored through Tesseract Studio:

```powershell
.\run_tesseract_studio.ps1
```

For a controller-only rehearsal that does not dispatch Vulkan work:

```powershell
.\run_tesseract_studio.ps1 -Demo
```

See [Tesseract Studio](TESSERACT_STUDIO.md) for its preset and safety
contracts.

## CPU-safe checks

Compile source and validate documentation/artifacts:

```powershell
python -m compileall -q .
python -m unittest tests.test_artifacts tests.test_studio -v
```

Run the numerical-relativity oracle and tracked Vulkan evidence contracts:

```powershell
python -m pytest tests\nr tests\test_artifacts.py -q
```

These checks do not compile or execute Vulkan shaders. They validate portable
models, generated-source contracts, machine-readable evidence, markdown
links, CPU physics, and the recorded CPU/GPU qualification thresholds.

## Live Vulkan campaigns

### Resident analytic-M1 vertical slice

```powershell
.\run_vulkan_theory33.ps1
```

Writes `vulkan_theory33_results.json` and qualifies resident recovery,
Carter flux/path evolution, entropy projection, SSP-RK3, and zero readbacks.

### Frozen-neural phase tape

```powershell
.\run_vulkan_frozen_neural.ps1
```

Writes `vulkan_frozen_neural_results.json` and checks digest-locked
constitutive replay, cellwise hard-phase agreement, dynamic branch movement,
physical margins, and conservation.

To run the literal initial state and horizon of the original learned-phase
Riemann campaign through the resident longitudinal Carter Roe path:

```powershell
.\run_vulkan_frozen_neural.ps1 `
    -Resolutions "16,32,64,128" `
    -FinalTime 0.02 `
    -Cfl 0.12 `
    -Problem original_neural_phase_riemann `
    -Reconstruction carter_characteristic_roe `
    -PositivityPreserving `
    -Output vulkan_original_neural_phase_roe_results.json
```

This remains the restricted five-field control. The complete original
nine-field plus entropy/tracer `full_carter_roe` reproduction is:

```powershell
.\run_vulkan_original_neural_phase_riemann.ps1
```

It writes `vulkan_original_neural_phase_full_roe_results.json` and requires
the original 0/0/0/4 phase-change pattern in addition to recovery,
eigensystem, causality, convexity, conservation, convergence, and
device-residency gates.

### Original analytic Carter shock ladders

```powershell
.\run_vulkan_carter_riemann.ps1
```

Writes `vulkan_carter_riemann_results.json` and reproduces the original
counterflow-shock and colliding-stream ladders with the complete resident
nine-field eigensystem, entropy/tracer extension, DLM path fluctuation, and
SSP-RK3 constitutive projection.

### WENO5-Z and positivity

```powershell
.\run_vulkan_weno.ps1
```

Writes `vulkan_weno_results.json` and measures fifth-order interfaces,
low/high conservative flux blending, limiter activation, smooth parity, and
learned-phase evolution.

The complete nine-physical-field plus entropy/tracer characteristic path is:

```powershell
.\run_vulkan_full_carter_weno.ps1
```

It writes `vulkan_full_carter_weno_results.json` and checks the resident RHS
against the float64 full-Carter WENO oracle before a three-level smooth
device-resident convergence study.

### Resident constrained CMC/Picard ladder

```powershell
.\run_vulkan_constrained.ps1
```

Writes `vulkan_constrained_results.json`. The runner constructs the analytic
and frozen active-counterflow states on Vulkan, solves the periodic CMC
equations through explicit fp64 Vulkan compute entry points, packs the result
into CCZ4, and evolves the `8/16/32` Carter/Proca/CCZ4 ladder without state
readbacks. Independent CPU diagnostics gate elliptic residuals, initial and
final second-order constraints, conservation, switching, characteristics,
absolute micro-horizon energy balance, and trajectory agreement. Resolved
geometric-work-rate convergence is gated separately by the 2D frontier
ladder.

### Numerical Carter eigensystem

```powershell
.\run_vulkan_carter_eigensystem.ps1
```

Writes `vulkan_carter_eigensystem_results.json` and audits numerical
flux/path symbols, real characteristic speeds, left/right basis inversion,
branch-aware face selection, and characteristic reconstruction.

### Mixed sources and full curved Carter/CCZ4 contracts

```powershell
.\run_vulkan_mixed_sources.ps1
.\run_vulkan_full_carter_ccz4.ps1
```

These write `vulkan_mixed_source_results.json` and
`vulkan_full_carter_ccz4_results.json`. They qualify mixed-vector stress,
implicit entropy-producing drag, the complete curved `9 x 9` Carter
eigensystem in every direction, CCZ4 algebraic projection, physical geometry,
and matter-source injection.

### Evolved mixed massive-vector fields

```powershell
.\run_vulkan_mixed_proca.ps1
```

Writes `vulkan_mixed_proca_results.json`. It independently compares the
curved mixed-Proca RHS, derived magnetic fields, Gauss cleaning, and SSP-RK3
evolution with the float64 Theory 3 oracle, then compares a complete
CCZ4/Proca/Carter coupled step with the CPU engine.

### Coupled multidimensional evolution

```powershell
.\run_vulkan_multidimensional.ps1
```

Writes `vulkan_multidimensional_results.json`. The tracked `8 x 8` campaign
requires:

- agreement of the derivative CCZ4 vacuum RHS with the float64 FD2 oracle;
- valid curved nine-variable recovery at every point;
- periodic baryon, carrier, and tracer balance;
- nonnegative entropy production from the implicit source schedule;
- determinant and tracefree CCZ4 projection;
- a resolved frozen-neural switching margin;
- finite matter, stress, and geometry outputs; and
- exact coupled checkpoint roundtrip with deterministic continuation; and
- zero host readbacks during evolution.

### Long-horizon and multidimensional frontier

```powershell
.\run_vulkan_frontier.ps1
```

Writes `vulkan_frontier_results.json`. It requires analytic and frozen
128-step device-resident evolution with complete high-order reconstruction,
a `6/12/24` two-dimensional continuum ladder, resolved geometric-work-rate
convergence, and an active three-dimensional learned-physics smoke run.

### Fourth-order CCZ4

```powershell
.\run_vulkan_ccz4_fd4.ps1
```

Writes `vulkan_ccz4_fd4_results.json`. It compares the resident FD4 vacuum
operator against the original float64 Cartesian FD4 oracle and gates the
complete coupled evolution, projection, recovery, conservation, restart, and
zero-readback contracts.

### Physical boundaries

```powershell
.\run_vulkan_boundaries.ps1
```

Writes `vulkan_boundary_results.json`. It compares real characteristic
projection, Carter no-inflow outflow, CCZ4 Sommerfeld/Z4 projection, and
mixed-Proca radiative/Gauss correction with the CPU operators. Both FD2 and
one-sided FD4 boundary stencils are tested, including a complete
Carter–Proca–CCZ4 FD4 resident step.

### AMR primitives

```powershell
.\run_vulkan_amr.ps1
```

Writes `vulkan_amr_results.json`. It qualifies conservative prolongation and
restriction in one through three dimensions, componentwise Berger–Colella
reflux, and the Theory 3 Gauss projection.

### TOV and isolated CTT

```powershell
.\run_vulkan_tov.ps1
.\run_vulkan_isolated_ctt.ps1
```

These write `vulkan_tov_results.json` and
`vulkan_isolated_ctt_results.json`. They qualify fp64 resident TOV recurrence,
Cartesian strong-field sampling, a finite active evolution with zero stage
readbacks, complete deterministic restart, and the fp64 isolated CTT
constraint solve against the original implementations.

### Exhaustive capability audit

```powershell
.\run_vulkan_original_workflows.ps1
.\run_vulkan_parity.ps1
```

The first command writes `vulkan_original_workflows_results.json` after
executing the complete Carter/KKT recurrence VJP, native master AD, recurrent
BPTT, Theory 3.3 reverse mode, frozen constitutive replay, and the conditioned
128-step audit. The second writes `vulkan_parity_results.json`; it requires
every declared resident artifact to pass, every retained host workflow to
exist, all original computational sources to match their local counterparts,
and a complete SHA-256 manifest of the production Vulkan implementation.

### Puncture/MOTS production replay

```powershell
.\run_vulkan_puncture_mots.ps1 -PlanOnly
.\run_vulkan_puncture_mots.ps1 -SaveSnapshots
```

The first command validates the `120^3`, length-3, FD4 numerical contract and
its 16-cell reference-radius gate without allocating a device. The second
launches the restartable resident vacuum CCZ4 evolution and requests host
snapshots only for KKT/MOTS observations, oracle constraints, saved
snapshots, and checkpoints. See
[the campaign contract](VULKAN_PUNCTURE_MOTS.md).

## Short live regressions

```powershell
.\.venv-slang\Scripts\python.exe .\test_vulkan_theory33_solver.py
.\.venv-slang\Scripts\python.exe .\test_vulkan_frozen_neural.py
.\.venv-slang\Scripts\python.exe .\test_vulkan_weno.py
.\.venv-slang\Scripts\python.exe .\test_vulkan_carter_eigensystem.py
.\.venv-slang\Scripts\python.exe .\test_vulkan_mixed_sources.py
.\.venv-slang\Scripts\python.exe .\test_vulkan_full_carter_ccz4.py
.\.venv-slang\Scripts\python.exe .\test_vulkan_multidimensional.py
.\.venv-slang\Scripts\python.exe .\test_vulkan_mots.py
```

## Tracked evidence policy

Each `vulkan_*_results.json` file is generated by its matching campaign and is
checked into the repository. A result is qualified only when every named
gate is true. Reports document the equations, thresholds, measured values,
and honest boundary of the corresponding experiment.

Device kernels use float32 while the independent oracle uses float64. Final
digits may differ within the reported bounds.

## Platform warning

SlangPy can print a D3D12 Agility SDK warning when Python and the SDK are on
different drives. Vulkan Tesseract explicitly creates a Vulkan device; this
warning does not change the validated execution path.
