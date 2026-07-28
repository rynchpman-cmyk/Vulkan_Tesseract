param(
    [string]$Resolutions = "8,16,32",
    [double]$FinalTime = 0.000001,
    [int]$PicardIterations = 3,
    [double]$EllipticTolerance = 0.00000002,
    [string]$Output = "vulkan_constrained_results.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv-slang\Scripts\python.exe"

& $Python (Join-Path $RepoRoot "run_vulkan_constrained.py") `
    --resolutions $Resolutions `
    --final-time $FinalTime `
    --picard-iterations $PicardIterations `
    --elliptic-tolerance $EllipticTolerance `
    --output (Join-Path $RepoRoot $Output)
