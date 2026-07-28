# GPU WENO5-Z and Positivity Port

## Result

Vulkan Tesseract now has a conservative higher-order interface layer inside
the resident Theory 3.3 solver. Slang constructs both a componentwise
WENO5-Z flux and a piecewise-constant Rusanov control flux, audits the
forward-Euler high-order candidate, and blends troubled faces back toward
the monotone flux without breaking conservation.

The tracked record is
[`vulkan_weno_results.json`](vulkan_weno_results.json). Every interface,
smooth-evolution, positivity-torture, and moving neural-phase gate passes.

## Interface reconstruction

For each conserved component, the device reconstructs left and right
cell-average states and physical fluxes with the standard three WENO5
candidate polynomials and WENO-Z nonlinear weights. The local
Lax--Friedrichs interface uses unit causal speed, matching the qualified
first-order control path.

A periodic passive scalar supplies an exact interface audit:

| Cells | Maximum interface error | Observed order |
|---:|---:|---:|
| 8 | \(5.63\times10^{-6}\) | — |
| 16 | \(1.54\times10^{-7}\) | 5.191 |
| 32 | \(5.70\times10^{-9}\) | 4.755 |

The test stops before float32 roundoff masks the formal spatial order.

## Conservative positivity limiter

For every forward-Euler substep, the device forms low- and high-order
candidate states. It computes a cell coefficient \(\theta\in[0,1]\) from:

- baryon conserved-density positivity;
- carrier conserved-density positivity; and
- a dominant-energy interior proxy.

The coefficient assigned to a face is the minimum of the coefficients in
its two adjacent cells. The final interface is

\[
F^\mathrm{limited}_{i+1/2}
=F^\mathrm{low}_{i+1/2}
+\min(\theta_i,\theta_{i+1})
\left(F^\mathrm{high}_{i+1/2}-F^\mathrm{low}_{i+1/2}\right).
\]

Because one face receives one coefficient, the flux remains conservative.
The DLM canonical-momentum path term is unchanged by this conservative flux
blend.

## Recorded evolutions

On the smooth analytic-M1 16/32/64 ladder, the limiter remains exactly
inactive and GPU/CPU physical-state errors are
\(9.38\times10^{-8}\), \(1.75\times10^{-7}\), and
\(2.06\times10^{-7}\). Recovery succeeds at every cell and periodic number
balances remain below \(3.52\times10^{-7}\).

The discontinuous falsification state runs one CFL-0.8 SSP-RK3 step with an
elevated audit floor:

- GPU minimum \(\theta=0.481805\);
- independent CPU minimum \(\theta=0.481991\);
- CPU/GPU physical-state error \(6.27\times10^{-7}\);
- zero recovery failures;
- positive baryon and carrier conserved densities; and
- absolute number-balance errors below \(1.27\times10^{-7}\).

The WENO frozen-neural phase front changes 2 and 5 cell branches at 16 and
32 cells. GPU and CPU hard phases agree in every cell, with worst physical
error \(6.74\times10^{-7}\) and no host reads between snapshots.

## Reproduction

```powershell
.\run_vulkan_weno.ps1
```

Short live regression:

```powershell
.\.venv-slang\Scripts\python.exe .\test_vulkan_weno.py
```

CPU artifact tests:

```powershell
python -m unittest tests.nr.test_vulkan_reference -v
```

## Historical boundary and current status

This campaign originally qualified the componentwise WENO5-Z stencil,
low/high flux pair, admissibility audit, and conservative face limiter. Its
then-open characteristic promotion has since been completed with the full
eleven-field Carter flux/path eigenbasis and branch-aware one-sided bases.
See `vulkan_full_carter_weno_results.json` and the
[capability alignment manifest](VULKAN_CAPABILITY_PARITY.md).
