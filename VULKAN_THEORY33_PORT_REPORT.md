# GPU-Resident Theory 3.3 Port

## Result

Vulkan Tesseract now advances its first complete PDE vertical slice without
reading the evolving state back to the host. Primitive recovery, analytic-M1
constitutive evaluation, Carter fluxes, the carrier canonical-momentum path
product, all SSP-RK3 stages, and energy-manifold entropy projection execute
in Slang on Vulkan.

The tracked machine record is
[`vulkan_theory33_results.json`](vulkan_theory33_results.json). All 24
qualification gates pass on a 16/32/64-cell ladder.

## Device state ABI

The seven float32 conserved fields are

\[
U=(D_N,D_D,S_x,E,P_x^D,D_Ns,D_NY).
\]

The device recovery variables are

\[
p=(\log n,q_N,\log d,q_D,s).
\]

State, recovery guesses, diagnostics, fluxes, DLM path data, right-hand
sides, and four Runge--Kutta state buffers remain allocated on the Vulkan
device. Python submits the dispatch sequence and updates scalar time metadata.
It performs no tensor read during `step()` or `evolve()`.

## GPU kernels

[`tesseract_nr_gpu.slang`](tesseract_nr_gpu.slang) implements:

- analytic Theory 3.3 M1 thermodynamics and Carter coefficients;
- conversion between primitive and conserved variables;
- a fixed-size pivoted \(5\times5\) Newton recovery with line search;
- physical recovery, convexity, and counterflow diagnostics;
- periodic local-Lax--Friedrichs/Rusanov fluxes;
- the literal midpoint path product in the carrier canonical-momentum
  equation;
- redundant-entropy projection onto the recovered energy manifold; and
- all three SSP-RK3 combination stages.

The Python control layer is
[`tesseract_nr/vulkan_backend.py`](tesseract_nr/vulkan_backend.py).
The independent float64 oracle in
[`tesseract_nr/vulkan_validation.py`](tesseract_nr/vulkan_validation.py)
uses the production CPU solver's energy recovery and master evaluation. It
does not reproduce the GPU Newton implementation.

## Recorded comparison

The smooth periodic problem evolves to \(t=0.002\) at CFL 0.1.

| Cells | Steps | Physical CPU/GPU relative error | Maximum GPU recovery residual |
|---:|---:|---:|---:|
| 16 | 1 | \(9.90\times10^{-8}\) | \(6.29\times10^{-7}\) |
| 32 | 1 | \(1.74\times10^{-7}\) | \(4.00\times10^{-7}\) |
| 64 | 2 | \(2.00\times10^{-7}\) | \(3.00\times10^{-5}\) |

At every resolution:

- host reads during evolution are exactly zero;
- every recovery is valid;
- baryon and carrier densities remain positive;
- relative counterflow remains inside the qualified domain;
- Legendre and thermodynamic margins remain positive; and
- periodic baryon and carrier-number balance errors remain below
  \(3.6\times10^{-7}\).

The minimum recorded Legendre and thermodynamic eigenvalues are 0.98459 and
0.89640. The largest relative Lorentz factor is 1.000248.

## Reproduction

Run the tracked campaign:

```powershell
.\run_vulkan_theory33.ps1
```

Run the short live Vulkan regression:

```powershell
.\.venv-slang\Scripts\python.exe .\test_vulkan_theory33_solver.py
```

Run the CPU-only contract test:

```powershell
python -m unittest tests.nr.test_vulkan_reference -v
```

## Honest scope

This is the first GPU-resident PDE vertical slice, not the completed NR port.
Its deliberate boundaries are:

- flat periodic one-dimensional spacetime;
- analytic M1 rather than the frozen learned phase closure;
- piecewise-constant Rusanov reconstruction rather than WENO5-Z;
- the material Carter subsystem without mixed-vector stress or drag;
- float32 device evolution with a float64 CPU oracle; and
- host-submitted kernels rather than a persistent device-side time loop.

The next promotion is the qualified frozen-neural closure and phase tape in
this resident state ABI, followed by higher-order reconstruction, the full
Carter eigensystem, CCZ4 geometry, multidimensional dispatch, and eventually
checkpointed PDE adjoints.

Subsequent milestones crossed the frozen-closure, source, full Carter,
multidimensional, and derivative CCZ4 boundaries. Current status is recorded
in the
[multidimensional Vulkan evolution report](VULKAN_MULTIDIMENSIONAL_EVOLUTION_REPORT.md).
