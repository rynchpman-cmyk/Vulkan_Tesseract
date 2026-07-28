# Vulkan CCZ4 Puncture/MOTS Campaign

## Scope

This campaign evolves equal-mass, time-symmetric Brill–Lindquist puncture
data with the resident Vulkan CCZ4 operator and observes the resulting slices
with the existing constrained spectral MOTS tracker.

The division of work is explicit:

- CCZ4 projection, FD4 vacuum RHS evaluation, radiative boundary treatment,
  gauge evolution, and SSP-RK3 stages remain resident on Vulkan.
- A host snapshot is requested only at the configured observation or
  checkpoint cadence.
- The KKT surface solve, branch continuation, physical-normal stability
  operator, masked constraint oracle, JSON output, and checkpoint metadata are
  read-only diagnostics on those requested snapshots.

The tracker never modifies the CCZ4 state.

## Production contract

The default command is the planned qualified replay:

```powershell
.\run_vulkan_puncture_mots.ps1 -SaveSnapshots
```

It expands to:

| Setting | Value |
|---|---:|
| Grid | `120 x 120 x 120` |
| Box lengths | `3 x 3 x 3` |
| Cell width | `0.025` |
| Initial puncture separation | `0.78` |
| Total bare mass | `1.0` |
| CFL | `0.04` |
| Steps / final time | `500 / 0.5` |
| Observation cadence | `10` |
| Checkpoint cadence | `50` |
| Common surface modes | even `m=0`, through `l=16` |
| Individual surface modes | all `m=0`, through `l=8` |
| Surface tolerance | `5e-4` |
| Minimum horizon gate | `16` cells per mean radius |

The measured pilot outer radius, approximately `0.418`, occupies `16.72`
cells at this spacing. The length-3 box also moves the boundary farther from
the formation region than the earlier length-2 pilot.

Validate the contract and allocation without creating a Vulkan device:

```powershell
.\run_vulkan_puncture_mots.ps1 -PlanOnly
```

A short wiring smoke run is:

```powershell
.\run_vulkan_puncture_mots.ps1 `
  -Resolution 16 `
  -BoxLength 2 `
  -Steps 1 `
  -ObservationCadence 1 `
  -CheckpointCadence 1 `
  -MaximumDegree 2 `
  -IndividualMaximumDegree 1 `
  -MaximumIterations 2 `
  -SkipIndividual `
  -SkipConstraints `
  -OutputDirectory runs/vulkan_ccz4_puncture_mots_smoke
```

## Initial data and ABI

The packed resident state has 31 float32 components:

```text
gt_ij, At_ij, W, K, Theta, GammaHat_i, alpha, beta_i, B_i
```

Tensor entries are row-major within each point and the Cartesian grid uses
x-fastest flattening. For conformally flat puncture data,

```text
gamma_ij = psi^4 delta_ij
gt_ij    = delta_ij
W        = det(gamma)^(-1/6) = psi^-2
alpha    = W
```

Using `alpha=W`, rather than `sqrt(W)`, is a required initialization gate.

## Branch handling

The observer carries four independent continuation branches:

- `common_outer`, initially seeded at radius `0.58`;
- `common_inner`, initially seeded at radius `0.32`;
- `individual_minus`, centered on the negative puncture; and
- `individual_plus`, centered on the positive puncture.

Puncture centers are updated from axial minima of `W`. Common branches are
classified by the sign of the physical-normal principal stability mode, not
by their seed names. When only one member of a newly formed common pair is
found, its coefficients seed the missing partner with a controlled log-radius
split.

## Restart

Every checkpoint event writes:

- `final_checkpoint.npz`, containing the complete resident CCZ4 state and
  numerical backend contract; and
- `final_observer.json`, containing the KKT continuation state and exact
  campaign configuration.

Resume with:

```powershell
.\run_vulkan_puncture_mots.ps1 `
  -RestartFrom runs/vulkan_ccz4_puncture_mots_n120/final_checkpoint.npz
```

The evolution and observer step numbers and complete numerical
configurations must match before a restart is accepted.

## Honest qualification boundary

The MOTS tracker has already been exercised through `N=32` CPU puncture
collisions and recovers the expected common-surface saddle-node pair. The
Vulkan bridge preserves its seeds, modes, gates, and branch logic while
replacing only the CCZ4 evolution engine.

The N=120 run is configured and runnable here, but it is not a reported
production result until its trajectory has actually completed. A valid final
claim still requires:

- positive lapse, conformal factor, and conformal-metric spectrum;
- controlled puncture- and boundary-masked constraints;
- at least 16 cells per measured mean radius;
- a principal mode outside the calibrated grid-phase band; and
- a resolution ladder before interpreting a near-zero principal mode as a
  continuum event.
