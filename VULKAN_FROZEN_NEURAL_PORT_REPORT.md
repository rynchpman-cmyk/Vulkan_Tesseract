# GPU-Resident Frozen-Neural Theory 3.3 Port

## Result

The qualified frozen Theory 3.3 master now participates directly in Vulkan
primitive recovery and PDE evolution. This crosses the constitutive boundary
left open by the first analytic-M1 GPU slice: the learned invariant gradient,
entrainment, stress-energy, carrier canonical momentum, Carter fluxes, and
hard two-phase classifier are active inside every SSP-RK3 stage.

The tracked record is
[`vulkan_frozen_neural_results.json`](vulkan_frozen_neural_results.json).
All qualification gates pass on the 16/32/64-cell moving phase-front ladder.

## Locked model and nonsmooth semantics

[`tesseract_nr_gpu.slang`](tesseract_nr_gpu.slang) imports the generated
[`theory33_qualified_frozen.slang`](theory33_qualified_frozen.slang) module.
The artifact digest is
`e3a8e89ca635cfe68d5bd963152097f85ec47b7960819830c9513f45b7bcf108`.

The phase decision is intentionally nonsmooth:

1. classify the primitive guess;
2. freeze that branch through the complete Newton solve and every centered
   numerical-Jacobian evaluation;
3. reclassify the recovered primitive;
4. repeat on the new branch only when the classifier changed.

No derivative is taken through the hard classifier. This matches the
piecewise-constant phase tape used by the qualified CPU production solver.

## Device execution

The seven conserved fields, five recovery variables, fluxes, DLM path data,
right-hand side, diagnostics, and all four RK state buffers stay allocated on
the Vulkan device. The expanded diagnostics record phase, threshold,
switching margin, and recovery transition count without changing the
conserved-state ABI.

Python uploads initial primitives, submits kernels, and reads snapshots. It
does not read a tensor during `step()` or `evolve()`.

## Dynamic phase-front comparison

The validation problem is a boosted periodic contact whose two sides occupy
opposite learned constitutive phases. It evolves to \(t=0.1\) at CFL 0.12,
long enough for the hard front to move across cells.

| Cells | Steps | CPU/GPU relative physical error | GPU/CPU phase disagreements | Cells changing branch |
|---:|---:|---:|---:|---:|
| 16 | 14 | \(3.54\times10^{-7}\) | 0 | 1 |
| 32 | 27 | \(4.54\times10^{-7}\) | 0 | 4 |
| 64 | 54 | \(5.27\times10^{-7}\) | 0 | 11 |

Across the ladder:

- host reads are unchanged between the initial and final snapshots;
- every GPU and CPU cell selects the same hard constitutive branch;
- every primitive recovery is valid;
- the largest recovery residual is \(2.91\times10^{-5}\);
- baryon and carrier densities remain positive;
- minimum Legendre and thermodynamic margins are 0.97485 and 0.90166; and
- periodic number-balance errors remain below \(7.70\times10^{-7}\).

## Reproduction

Run the tracked ladder:

```powershell
.\run_vulkan_frozen_neural.ps1
```

Run the shorter live qualification:

```powershell
.\.venv-slang\Scripts\python.exe .\test_vulkan_frozen_neural.py
```

Run the CPU-only artifact contracts:

```powershell
python -m unittest tests.nr.test_vulkan_reference -v
```

## Honest boundary

### Original learned-phase Riemann parity audit

The original Tesseract learned-phase Riemann campaign is now represented by a
separate literal initial-data option:

`original_neural_phase_riemann`

It uses `n=0.10`, `epsilon=0.12`, `v_N=0`, `d=0.005`, and carrier velocities
`0.112/0.103`, exactly matching the original left/right states. The resident
backend also now provides a branch-aware
`carter_characteristic_roe` reconstruction for the independent longitudinal
five-field block.

The corrected 16/32/64/128 GPU ladder at `t=0.02`, CFL `0.12`:

- passes all 37 resident qualification gates;
- performs no host readback during any evolution interval;
- has zero recovery failures and zero hard-phase disagreements with the
  matching float64 restricted oracle; and
- changes `0/0/0/3` cell branches.

The original full nine-field plus entropy/tracer float64
`full_carter_roe` campaign changes `0/0/0/4` branches. Therefore the
restricted resident Roe result is a successful GPU execution and a useful
localization experiment, but it is not claimed as exact original parity.
The boundary at this historical milestone was the complete eleven-field Roe
basis, its exact DLM path fluctuation, and arithmetic-sensitivity
qualification. That boundary is closed by
`vulkan_original_neural_phase_full_roe_results.json`.

Tracked evidence:
`vulkan_original_neural_phase_roe_results.json`.

This campaign qualified active neural constitutive execution in the resident
flat 1D material solver. At that historical milestone, the dependency chain
was:

1. higher-order reconstruction with positivity-preserving fallback;
2. the complete numerical Carter flux/path eigensystem;
3. mixed-vector stress, drag, and source coupling;
4. CCZ4 geometry and constraint damping;
5. multidimensional stencils, boundaries, and reductions; and
6. checkpointed or reversible PDE adjoints.

The frozen-neural closure and hard phase tape no longer block those ports.

All six items were subsequently promoted or retained in their correct
execution role and qualified. Current full-stack scope is recorded in the
[capability alignment manifest](VULKAN_CAPABILITY_PARITY.md).
