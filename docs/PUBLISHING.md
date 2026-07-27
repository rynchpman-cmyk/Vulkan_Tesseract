# GitHub publishing

## Recommended repository identity

- **Name:** `Vulkan_Tesseract`
- **Description:** `Turn your gaming PC into a GPU-resident neural Carter/CCZ4 spacetime simulator.`
- **Topics:**
  - `automatic-differentiation`
  - `differentiable-physics`
  - `dynamical-systems`
  - `kkt`
  - `lyapunov`
  - `numerical-relativity`
  - `ccz4`
  - `gpu-computing`
  - `neural-pde`
  - `shader`
  - `slang`
  - `vulkan`

## First publication

After installing and authenticating the GitHub CLI:

```powershell
gh auth login
gh auth status
```

Create the public repository:

```powershell
gh repo create Vulkan_Tesseract `
    --public `
    --source . `
    --remote origin `
    --push
```

Use `--private` only for a staging mirror. This project is distributed under
the Apache License 2.0.

Set topics:

```powershell
gh repo edit --add-topic automatic-differentiation `
    --add-topic differentiable-physics `
    --add-topic dynamical-systems `
    --add-topic kkt `
    --add-topic lyapunov `
    --add-topic numerical-relativity `
    --add-topic ccz4 `
    --add-topic gpu-computing `
    --add-topic neural-pde `
    --add-topic shader `
    --add-topic slang `
    --add-topic vulkan
```

## Before making the repository public

1. Confirm that `LICENSE`, `NOTICE`, and `CITATION.cff` are current.
2. Confirm whether portable trained weights should be publicly redistributed.
3. Confirm the desired author identity for commits and citation metadata.
4. Re-run the separate NumPy NR and PyTorch constitutive commands in
   [Reproducibility](REPRODUCIBILITY.md#1-artifactsource-checks).
5. Re-run `.\run_conditioned_literal.ps1`.
6. Confirm the dashboard image contains no unwanted local information.

Binary checkpoints, local virtual environments, downloaded tools, and SPIR-V
binaries are already excluded by `.gitignore`.
