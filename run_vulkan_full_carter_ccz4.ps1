param(
    [int]$SourcePoints = 12,
    [int]$CarterPoints = 6,
    [int]$CCZ4Points = 8,
    [string]$Output = "vulkan_full_carter_ccz4_results.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv-slang\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing .venv-slang. Install requirements-slang.txt first."
}

& $Python (Join-Path $RepoRoot "run_vulkan_full_carter_ccz4.py") `
    --source-points $SourcePoints `
    --carter-points $CarterPoints `
    --ccz4-points $CCZ4Points `
    --output (Join-Path $RepoRoot $Output)
