# Vulkan Multidimensional Carter/CCZ4 Evolution Report

## Milestone

This milestone closes the boundary left by the local full-Carter/CCZ4 port.
The Slang/Vulkan engine now advances a resident multidimensional state through:

- curved nine-variable Carter primitive recovery at every grid point;
- mixed-vector stress recomputation on the recovered curved state;
- flux and nonconservative Carter-path evaluation in every active direction;
- periodic Rusanov interface assembly and summed multidirectional RHS;
- SSP-RK3 matter evolution;
- implicit counterflow drag in a source half-step before and after transport;
- the complete derivative-containing CCZ4 vacuum operator;
- recovered total Eulerian Carter matter injection into CCZ4; and
- CCZ4 half-steps around the full matter step.

Python initializes the run, submits kernels, and performs explicit
qualification readbacks. The evolution stages, recoveries, constitutive
branch decisions, interface assembly, source solves, geometry reconstruction,
and CCZ4 derivatives remain on the Vulkan device.

## Resident ABI

The Carter conserved state is

`(D_N, D_D, S_x, S_y, S_z, E, P^D_x, P^D_y, P^D_z, D_N s, D_N Y)`.

The recovered primitive is

`(log n, s, q^N_x, q^N_y, q^N_z, log d, q^D_x, q^D_y, q^D_z)`.

The packed CCZ4 state remains

`(gt_ij, At_ij, phi, K, theta, GammaHat_i, alpha, beta_i, B_i)`.

Flattened grids use x-fastest indexing. One-, two-, and three-dimensional
periodic domains share the same shader ABI; inactive directions contribute
exactly zero derivatives and transport RHS.

## Curved recovery and multidirectional transport

Recovery subtracts the geometry-dependent vector energy and momentum from the
total state, then solves the full material nine-variable conserved map with a
scaled finite-difference Newton system and backtracking. A hard frozen-neural
phase is constant within a Newton solve and is reclassified only between
transition solves.

Each active direction evaluates its own curved Carter flux, canonical
momentum path data, coordinate light-speed bound, and face flux. Directional
RHS buffers are summed only after each complete face/path pass. The physical
nine-field block is extended by baryon-advected entropy and tracer densities.

The local drag solve holds material densities and energy-momentum fixed while
relaxing three-component counterflow. The evolved specific tracer is extracted
from the resident conserved state before source recomposition; it is not reset
from initialization data.

## Derivative-containing CCZ4 operator

The device vacuum operator implements the same periodic centered-FD2 equations
as `CCZ4Solver.rhs`:

- physical and conformal Christoffel symbols;
- physical Ricci tensor and scalar;
- lapse Hessian and covariant Laplacian;
- the CCZ4 `Z_i`, `D_i Z_j`, and divergence;
- Lie/advection derivatives of `gt_ij`, `At_ij`, `phi`, `K`, and `theta`;
- all `GammaHat_i` first- and second-derivative terms;
- 1+log or harmonic lapse;
- Gamma-driver shift evolution; and
- `kappa1`, `kappa2`, and `kappa3` constraint damping.

The existing matter kernel adds the recovered total
`(rho, S_i, S_ij)` source. Each geometry stage is algebraically projected
before the next RHS evaluation, preserving `det(gt)=1` and
`trace_gt(At)=0`.

## Coupled schedule

One full step is

1. projected CCZ4 SSP-RK3 half-step with stagewise Carter recovery;
2. implicit Carter drag half-step;
3. multidirectional Carter SSP-RK3 transport;
4. implicit Carter drag half-step; and
5. projected CCZ4 SSP-RK3 half-step with refreshed total matter stress.

This is a symmetric geometry/matter split, with a second symmetric split for
the stiff drag source inside the matter step.

## Qualification

The tracked campaign uses an `8 x 8` curved periodic grid, the qualified
frozen-neural closure, dynamically resolved phase margins, nonzero shift,
nontrivial conformal geometry, mixed vector fields, and two complete coupled
steps.

| Quantity | Measured |
|---|---:|
| CCZ4 vacuum RHS maximum CPU/GPU error | `2.325e-6` |
| CCZ4 vacuum RHS RMS CPU/GPU error | `2.295e-7` |
| Final maximum curved recovery residual | `4.303e-7` |
| Invalid final recoveries | `0` |
| Baryon balance error | `7.217e-7` |
| Carrier balance error | `9.979e-7` |
| Tracer balance error | `7.334e-7` |
| Total entropy change | `+1.942e-5` |
| Projected determinant residual | `0` |
| Projected trace-`A` residual | `7.276e-12` |
| Minimum lapse | `0.9940` |
| Minimum absolute neural switching margin | `5.045e-4` |
| Host readbacks during evolution | `0` |

All qualification gates pass. The independent oracle is the float64 CPU
CCZ4 implementation evaluated on the exact float32 state projected by the
GPU, so the derivative comparison isolates the operator rather than input
roundoff or projection differences.

## Reproduction

```powershell
.\run_vulkan_multidimensional.ps1
```

The command writes `vulkan_multidimensional_results.json`. The shorter live
regression is:

```powershell
.\.venv-slang\Scripts\python.exe .\test_vulkan_multidimensional.py
```

## Honest boundary

The full Carter matter and derivative CCZ4 equations now execute together on
a multidimensional resident Vulkan grid. This result does not yet claim
production-scale strong-field stability, high-order CCZ4 derivatives,
AMR/domain decomposition, physical boundary conditions, or a converged
coupled spacetime/shock campaign. The current transport face operator is the
robust Rusanov baseline; the previously qualified characteristic WENO layer
has not yet been generalized to this full curved eleven-component interface.

Those are now performance, accuracy, and strong-field qualification
frontiers. The missing foundational execution path identified in the prior
report is closed.
