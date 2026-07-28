param(
    [string]$Resolutions = "16,32,64",
    [double]$FinalTime = 0.1,
    [double]$Cfl = 0.12,
    [ValidateSet(
        "piecewise_constant",
        "weno5_z",
        "carter_characteristic_weno5_z",
        "carter_characteristic_roe"
    )]
    [string]$Reconstruction = "piecewise_constant",
    [switch]$PositivityPreserving,
    [ValidateSet(
        "moving_phase_contact",
        "original_neural_phase_riemann"
    )]
    [string]$Problem = "moving_phase_contact",
    [string]$Output = "vulkan_frozen_neural_results.json"
)

$ErrorActionPreference = "Stop"
$projectDir = $PSScriptRoot
$python = Join-Path $projectDir ".venv-slang\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    python -m venv (Join-Path $projectDir ".venv-slang")
}

& $python -m pip install --disable-pip-version-check --quiet `
    -r (Join-Path $projectDir "requirements-slang.txt")
if ($LASTEXITCODE -ne 0) {
    throw "SlangPy environment setup failed with exit code $LASTEXITCODE."
}

$arguments = @(
    (Join-Path $projectDir "run_vulkan_frozen_neural.py"),
    "--resolutions", $Resolutions,
    "--final-time", $FinalTime,
    "--cfl", $Cfl,
    "--reconstruction", $Reconstruction,
    "--problem", $Problem,
    "--output", $Output
)
if ($PositivityPreserving) {
    $arguments += "--positivity-preserving"
}
& $python @arguments
exit $LASTEXITCODE
