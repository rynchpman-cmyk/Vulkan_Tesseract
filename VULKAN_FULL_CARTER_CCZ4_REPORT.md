# Vulkan Full Carter Sources and CCZ4 Geometry Report

## Scope

This milestone widens the qualified flat one-dimensional Vulkan slice without
changing or weakening it. The new device layer contains:

- the full three-vector Theory 3.3 primitive block
  `(log n, s, q_N^i, log d, q_D^i)`;
- the physical nine-field Carter conserved block
  `(D_N, D_D, S_i, E, P_i^D)`;
- mixed Proca stress-energy, including the off-diagonal mass mixing;
- an implicit local carrier-drag solve with entropy production;
- arbitrary spatial metric, lapse, shift, and all three directional fluxes;
- the literal canonical-momentum path contribution;
- the complete numerical `9 x 9` flux/path eigensystem; and
- packed CCZ4 algebraic projection, physical-geometry reconstruction, and
  Carter matter-source injection.

Entropy and tracer densities extend the physical nine-field block to the
eleven-component transport ABI. They are not counted as independent Carter
principal fields.

## Mixed-vector stress and drag

The vector sector is evaluated as two independent device passes followed by a
device combine pass. This is deliberate: on the tested Vulkan compiler,
nested fixed-array stress calls could alias temporary storage. Splitting the
sectors preserves the exact equations and removes that driver-dependent
temporary lifetime.

The drag kernel solves nine local unknowns:

`(log n, s, q_N^x, q_N^y, q_N^z, log d, q_D^x, q_D^y, q_D^z)`.

It holds both number densities and total material energy-momentum fixed while
exponentially relaxing `q_D^i - q_N^i`. The hard neural phase is fixed within
each Newton solve and reclassified between transition solves. States below the
counterflow-energy threshold remain exactly unchanged.

Tracked analytic/frozen qualification:

| Quantity | Maximum |
|---|---:|
| Flat mixed-stress CPU/GPU error | `8.382e-9` |
| Curved mixed-stress CPU/GPU error | about `5e-9` |
| Drag primitive CPU/GPU error | `3.762e-6` |
| Material conservation error | `2.980e-8` |
| GPU scaled drag residual | `3.346e-6` |
| Invalid drag points | `0` |

All twelve frozen-neural test points crossed the same phase boundary on CPU
and Vulkan. Every solve produced nonnegative entropy.

## Curved multidimensional Carter block

For each spatial direction the GPU evaluates:

1. densitized conserved variables `U(p)`;
2. the curved directional flux `F^i(p)`;
3. `chi_0`, `chi_i`, `D_D`, and carrier transport velocity;
4. centered branch-frozen Jacobians;
5. the path-corrected symbol
   `(F_,p + B_path,p) @ inverse(U_,p)`; and
6. its nine eigenvalues plus complete right and left bases.

The device eigensolver uses active-block shifted QR deflation followed by
inverse iteration. Repeated transverse families are orthogonalized within
their degenerate cluster before the right basis is inverted.

Across analytic M1 and both frozen-neural branches, in all three curved
directions:

| Quantity | Maximum |
|---|---:|
| Conserved-map absolute error | about `4.2e-8` |
| Flux absolute error | about `1.6e-8` |
| Path-data absolute error | about `5.2e-8` |
| Relative symbol error | below `1.4e-3` |
| Eigenvalue absolute error | below `4.0e-4` |
| Eigenpair residual | below `3.6e-4` |
| Inverse-basis residual | below `7e-6` |
| Invalid eigensystems | `0` |

The CPU spectra were real throughout. Coordinate characteristic speeds are
reported as diagnostics, not silently clipped; coordinate speeds depend on
the supplied lapse, shift, and spatial metric.

## CCZ4 geometry and matter coupling

The packed 31-component CCZ4 ABI is:

`(gt_ij, At_ij, phi, K, theta, GammaHat_i, alpha, beta_i, B_i)`.

The projection kernel:

- symmetrizes the conformal metric and conformal `A`;
- enforces `det(gt) = 1`;
- removes the conformal trace of `At`;
- preserves all unconstrained fields; and
- maintains positive conformal factor and lapse floors.

The physical-geometry kernel reconstructs

`h_ij = gt_ij / phi^2`

and

`K_ij = (At_ij + K gt_ij / 3) / phi^2`.

The matter-source kernel injects the full Eulerian `(rho, S_i, S_ij)` source
into the `At`, `K`, `theta`, and `GammaHat_i` CCZ4 equations.

Tracked errors:

| Quantity | Maximum |
|---|---:|
| Projected state error | `9.313e-8` |
| Physical geometry error | `1.712e-7` |
| Extrinsic curvature error | `7.989e-9` |
| CCZ4 matter-source error | `3.718e-7` |
| GPU determinant residual | `2.384e-7` |
| GPU trace-`A` residual | `6.786e-9` |
| Invalid projected geometries | `0` |

## Reproduction

```powershell
.\run_vulkan_full_carter_ccz4.ps1
```

The command writes
`vulkan_full_carter_ccz4_results.json`. A shorter live regression is:

```powershell
.\.venv-slang\Scripts\python.exe .\test_vulkan_full_carter_ccz4.py
```

## Honest boundary

This report qualifies the local contracts that preceded multidimensional
integration. The boundary described by its original version—curved grid
recovery, multidirectional interface/RHS assembly, Strang scheduling, and the
derivative CCZ4 vacuum terms—is now crossed and documented in the
[multidimensional Vulkan evolution report](VULKAN_MULTIDIMENSIONAL_EVOLUTION_REPORT.md).
