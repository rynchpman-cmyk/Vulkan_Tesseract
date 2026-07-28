param(
    [string]$Resolutions = "16,32,64,128",
    [double]$FinalTime = 0.02,
    [double]$Cfl = 0.15,
    [string]$Output = "vulkan_carter_riemann_results.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv-slang\Scripts\python.exe"

& $Python (Join-Path $RepoRoot "run_vulkan_carter_riemann.py") `
    --resolutions $Resolutions `
    --final-time $FinalTime `
    --cfl $Cfl `
    --output (Join-Path $RepoRoot $Output)
