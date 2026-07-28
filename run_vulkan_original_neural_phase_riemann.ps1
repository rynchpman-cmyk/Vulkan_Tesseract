param(
    [string]$Resolutions = "16,32,64,128",
    [double]$FinalTime = 0.02,
    [double]$Cfl = 0.12,
    [string]$Reference = "tesseract_neural_phase_riemann_results.json",
    [string]$Output = "vulkan_original_neural_phase_full_roe_results.json",
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"
$projectDir = $PSScriptRoot
$python = Join-Path $projectDir ".venv-slang\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    python -m venv (Join-Path $projectDir ".venv-slang")
}

if (-not $SkipSetup) {
    & $python -m pip install --disable-pip-version-check --quiet `
        -r (Join-Path $projectDir "requirements-slang.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "SlangPy environment setup failed with exit code $LASTEXITCODE."
    }
}

& $python `
    (Join-Path $projectDir "run_vulkan_original_neural_phase_riemann.py") `
    --resolutions $Resolutions `
    --final-time $FinalTime `
    --cfl $Cfl `
    --reference $Reference `
    --output $Output
exit $LASTEXITCODE
