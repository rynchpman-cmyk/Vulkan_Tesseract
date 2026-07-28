# GPU Numerical Carter Flux/Path Eigensystem

## Result

The resident flat-1D Theory 3.3 solver now reconstructs its complete
five-field physical Carter block in characteristic space. The device forms
the literal flux/path principal symbol, computes and audits its real
eigensystem, freezes one basis over every WENO stencil, and selects a
qualified physical-side basis instead of averaging across a learned hard
phase boundary.

The tracked record is
[`vulkan_carter_eigensystem_results.json`](vulkan_carter_eigensystem_results.json).
All nine symbol, evolution, shock, neural-phase, and residency gates pass.

## Principal symbol

For primitive variables
\[
p=(\log n,q_N,\log d,q_D,s)
\]
and independent conserved variables
\[
U=(D_N,D_D,S_x,E,P_x^D),
\]
Slang forms centered frozen-branch Jacobians and evaluates
\[
A=(F_{,p}-B_{,p})U_{,p}^{-1}.
\]

Here \(B_{,p}\) is the literal smooth limit of the carrier
canonical-momentum DLM path product. The hard learned phase is held fixed
through every centered Jacobian evaluation.

## Device eigensolver

Each face performs:

1. a pivoted inverse of \(U_{,p}\);
2. 72 shifted QR iterations for the five real characteristic speeds;
3. inverse iteration for deterministic normalized right eigenvectors;
4. a pivoted inverse for the left basis; and
5. eigenpair, inverse-basis, conditioning, separation, and causality audits.

The frozen face basis projects every member of each WENO stencil. The
resulting modal split flux is reconstructed with WENO5-Z and mapped back with
the same right basis. The previously qualified conservative positivity
limiter remains outside this characteristic transform.

## Independent symbol comparison

An independent float64 NumPy implementation forms the same restricted symbol
using the production frozen closure and `numpy.linalg.eig`.

| Model | Maximum speed error | GPU eigenpair residual | Maximum speed | Phase faces |
|---|---:|---:|---:|---:|
| Analytic M1 | \(3.70\times10^{-4}\) | \(1.28\times10^{-4}\) | 0.95843 | 0 |
| Qualified frozen | \(1.55\times10^{-4}\) | \(1.50\times10^{-6}\) | 0.98502 | 2 |

Both CPU spectra are entirely real. Every GPU basis is invertible and valid;
the largest inverse-basis residual is \(1.42\times10^{-6}\), and the minimum
eigenvalue separation is 0.32275.

## Evolution ladders

The smooth 16/32/64 characteristic-WENO ladder has:

- zero invalid eigensystems and zero recovery failures;
- no positivity limiting on the admissible smooth state;
- maximum speed 0.95852;
- maximum evolved eigenpair residual \(7.68\times10^{-4}\); and
- CPU/GPU physical-state error at most \(2.05\times10^{-7}\).

The discontinuous analytic 16/32 ladder remains causal with maximum speed
0.98690. CPU/GPU error remains below \(4.76\times10^{-7}\), and both
periodic number balances close below \(5.92\times10^{-7}\).

The moving frozen-neural 16/32 ladder:

- detects exactly two phase-boundary faces at each resolution;
- changes 2 and 6 cell branches dynamically;
- has zero CPU/GPU phase disagreements;
- has zero invalid eigensystems and recovery failures;
- remains causal with maximum speed 0.98502; and
- agrees with the independent CPU characteristic solver within
  \(9.64\times10^{-7}\).

No tensor is read during any evolution interval.

## Reproduction

```powershell
.\run_vulkan_carter_eigensystem.ps1
```

Short live regression:

```powershell
.\.venv-slang\Scripts\python.exe .\test_vulkan_carter_eigensystem.py
```

## Honest boundary

This is the complete numerical Carter flux/path eigensystem for the five
independent physical fields in the resident flat-1D ABI. Entropy and tracer
remain passive reconstructed fields.

At this historical milestone it was the complete five-field flat-1D
eigensystem; the full 3D nine-field block, transverse momentum, and carrier
canonical-momentum modes were the next port boundary.

Those local nine-field, source, CCZ4, and multidimensional execution
boundaries were subsequently crossed. See the
[full Carter/CCZ4 report](VULKAN_FULL_CARTER_CCZ4_REPORT.md) and
[multidimensional evolution report](VULKAN_MULTIDIMENSIONAL_EVOLUTION_REPORT.md).
