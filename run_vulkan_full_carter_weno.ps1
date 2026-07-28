param(
    [string]$Resolutions = "16,32,64",
    [double]$FinalTime = 0.005,
    [double]$Cfl = 0.1,
    [string]$Output = "vulkan_full_carter_weno_results.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv-slang\Scripts\python.exe"

& $Python (Join-Path $RepoRoot "run_vulkan_full_carter_weno.py") `
    --resolutions $Resolutions `
    --final-time $FinalTime `
    --cfl $Cfl `
    --output (Join-Path $RepoRoot $Output)
