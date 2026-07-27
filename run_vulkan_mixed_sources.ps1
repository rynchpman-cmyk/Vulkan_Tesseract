param(
    [int]$Points = 12,
    [int]$Seed = 33,
    [string]$Output = "vulkan_mixed_source_results.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv-slang\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing .venv-slang. Install requirements-slang.txt first."
}

& $Python (Join-Path $RepoRoot "run_vulkan_mixed_sources.py") `
    --points $Points `
    --seed $Seed `
    --output (Join-Path $RepoRoot $Output)
