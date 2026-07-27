param(
    [string]$SmoothResolutions = "16,32,64",
    [string]$PhaseResolutions = "16,32",
    [string]$Output = "vulkan_weno_results.json"
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

& $python (Join-Path $projectDir "run_vulkan_weno.py") `
    --smooth-resolutions $SmoothResolutions `
    --phase-resolutions $PhaseResolutions `
    --output $Output
exit $LASTEXITCODE
