param(
    [int]$Nx = 8,
    [int]$Ny = 8,
    [int]$Steps = 2,
    [string]$Output = "vulkan_multidimensional_results.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv-slang\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing .venv-slang. Install requirements-slang.txt first."
}
$OutputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    $Output
}
else {
    Join-Path $RepoRoot $Output
}

& $Python (Join-Path $RepoRoot "run_vulkan_multidimensional.py") `
    --nx $Nx `
    --ny $Ny `
    --steps $Steps `
    --output $OutputPath
exit $LASTEXITCODE
