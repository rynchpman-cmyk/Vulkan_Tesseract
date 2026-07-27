param(
    [string]$Resolutions = "16,32,64",
    [double]$FinalTime = 0.1,
    [double]$Cfl = 0.12,
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

& $python (Join-Path $projectDir "run_vulkan_frozen_neural.py") `
    --resolutions $Resolutions `
    --final-time $FinalTime `
    --cfl $Cfl `
    --output $Output
exit $LASTEXITCODE
