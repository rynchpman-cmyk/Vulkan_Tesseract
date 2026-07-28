# CPU numerical-relativity oracle

## Purpose

`tesseract_nr/` is the independent float64 reference engine used to qualify
the resident Slang/Vulkan implementation. It is intentionally transparent and
retains complete physical state, recovery, transport, constraint, and
continuum diagnostics.

The CPU engine is not the production GPU path. It supplies the oracle against
which Vulkan recovery, stress, eigensystems, derivatives, conservation, and
hard constitutive decisions are measured.

## Physical sectors

- CCZ4 conformal geometry and gauge
- conservative relativistic matter
- Carter baryon and carrier currents
- mixed massive-vector fields
- entropy-positive carrier drag
- analytic-M1 and frozen-neural constitutive response

## State contract

The CPU method-of-lines state contains 52 scalar fields per cell:

| Sector | Components |
|---|---:|
| CCZ4 geometry and gauge | 25 |
| Conservative material variables | 7 |
| Mixed-vector fields and cleaners | 16 |
| Carrier charge and canonical momentum | 4 |

The Vulkan backend uses a packed 31-component CCZ4 block and an
eleven-component Carter transport block. Explicit conversion and
qualification routines keep these ABIs comparable.

## Numerical roles

The oracle provides:

- physical/conformal metric inversion and curvature;
- Hamiltonian and momentum constraints;
- curved Carter primitive recovery;
- material and vector stress-energy;
- nonconservative path terms;
- numerical flux/path symbols and eigensystems;
- implicit drag and entropy production;
- periodic FD2/FD4/spectral derivatives;
- high-order transport and shock reference experiments; and
- nested-resolution error diagnostics.

## CPU checks

```powershell
python -m pytest tests\nr -q
```

The tracked Vulkan evidence contracts are included in:

```powershell
python -m pytest tests\nr\test_vulkan_reference.py -q
```

## Vulkan correspondence

The final coupled device campaign is documented in the
[multidimensional Vulkan evolution report](../VULKAN_MULTIDIMENSIONAL_EVOLUTION_REPORT.md).
It compares the derivative CCZ4 vacuum operator against this oracle on the
exact float32 state projected by the GPU and independently checks Carter
recovery, sources, balance laws, switching margins, and device residency.

## Qualification boundary

Agreement with the CPU engine is necessary but not by itself a claim of
universal production validity. The Vulkan engine is qualified for FD2 and
FD4 CCZ4 derivatives, Rusanov/full-Roe/characteristic-WENO Carter
interfaces, periodic and radiative/outflow domains, AMR transfer/reflux,
fp64 constrained and TOV setup, a tracked strong-field resident evolution,
128-step constrained horizons, a 2D continuum ladder, and a 3D smoke
campaign.

Those tests establish original-project capability alignment; they do not
replace problem-specific convergence studies. Longer strong-field ladders,
new microphysical closures, and multi-GPU Vulkan domain decomposition are
research and scaling extensions beyond the original qualification surface.
