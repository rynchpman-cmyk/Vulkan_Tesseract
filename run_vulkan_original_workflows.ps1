$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv-slang\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing Slang environment: $python"
}

& $python (Join-Path $root "run_vulkan_original_workflows.py") @args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
