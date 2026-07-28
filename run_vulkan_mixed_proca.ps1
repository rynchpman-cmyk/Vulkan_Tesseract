param(
    [string]$Shape = "8,8",
    [int]$Steps = 4,
    [double]$Dt = 0.0002,
    [string]$Output = "vulkan_mixed_proca_results.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv-slang\Scripts\python.exe"

& $Python (Join-Path $RepoRoot "run_vulkan_mixed_proca.py") `
    --shape $Shape `
    --steps $Steps `
    --dt $Dt `
    --output (Join-Path $RepoRoot $Output)
