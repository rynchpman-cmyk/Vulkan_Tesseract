# Vulkan Tesseract documentation

## Core guides

1. [Architecture](ARCHITECTURE.md) — resident GPU/CPU-oracle dataflow,
   constitutive branches, transport, sources, and CCZ4 coupling.
2. [Reproducibility](REPRODUCIBILITY.md) — environments, campaign commands,
   tracked evidence, and validation tiers.
3. [CPU numerical-relativity oracle](NR_PDE_SOLVER.md) — the independent
   float64 state contract used to qualify Vulkan physics.
4. [Publishing](PUBLISHING.md) — standalone repository metadata and public
   release checklist.
5. [Vulkan puncture/MOTS campaign](VULKAN_PUNCTURE_MOTS.md) — resident vacuum
   CCZ4 evolution and restartable constrained horizon observations.
6. [Tesseract Studio](TESSERACT_STUDIO.md) — beginner-facing presets,
   local visual control room, safe stop/resume, and extension contract.

## Vulkan progression reports

1. [Resident Theory 3.3 port](../VULKAN_THEORY33_PORT_REPORT.md)
2. [Frozen-neural constitutive port](../VULKAN_FROZEN_NEURAL_PORT_REPORT.md)
3. [WENO5-Z and positivity port](../VULKAN_WENO_PORT_REPORT.md)
4. [Numerical Carter eigensystem](../VULKAN_CARTER_EIGENSYSTEM_REPORT.md)
5. [Full Carter sources and CCZ4 geometry](../VULKAN_FULL_CARTER_CCZ4_REPORT.md)
6. [Multidimensional Carter/CCZ4 evolution](../VULKAN_MULTIDIMENSIONAL_EVOLUTION_REPORT.md)
7. [Original-to-Vulkan capability alignment](../VULKAN_CAPABILITY_PARITY.md)

The reports are ordered by implementation dependency. Earlier reports retain
their historical qualification boundary and link forward when a later
milestone crosses it.

## Machine-readable evidence

- `vulkan_theory33_results.json`
- `vulkan_frozen_neural_results.json`
- `vulkan_weno_results.json`
- `vulkan_carter_eigensystem_results.json`
- `vulkan_mixed_source_results.json`
- `vulkan_full_carter_ccz4_results.json`
- `vulkan_multidimensional_results.json`
- `vulkan_original_neural_phase_roe_results.json`
- `vulkan_original_neural_phase_full_roe_results.json`
- `vulkan_carter_riemann_results.json`
- `vulkan_full_carter_weno_results.json`
- `vulkan_mixed_proca_results.json`
- `vulkan_constrained_results.json`
- `vulkan_frontier_results.json`
- `vulkan_ccz4_fd4_results.json`
- `vulkan_amr_results.json`
- `vulkan_boundary_results.json`
- `vulkan_tov_results.json`
- `vulkan_isolated_ctt_results.json`
- `vulkan_original_workflows_results.json`
- `vulkan_parity_results.json`
