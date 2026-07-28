$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentRoot = Join-Path $RepoRoot ".venv-slang"
$EnvironmentPython = Join-Path $EnvironmentRoot "Scripts\python.exe"
$Requirements = Join-Path $RepoRoot "requirements-slang.txt"

Write-Host ""
Write-Host "  TESSERACT STUDIO // FIRST-TIME SETUP" -ForegroundColor Cyan
Write-Host "  This prepares the local Vulkan runtime. Nothing is uploaded."
Write-Host ""

if (-not (Test-Path -LiteralPath $Requirements)) {
    throw "Missing requirements-slang.txt."
}

if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
    $Bootstrap = Get-Command py -ErrorAction SilentlyContinue
    $BootstrapArguments = @("-3", "-m", "venv", $EnvironmentRoot)
    if ($null -eq $Bootstrap) {
        $Bootstrap = Get-Command python -ErrorAction SilentlyContinue
        $BootstrapArguments = @("-m", "venv", $EnvironmentRoot)
    }
    if ($null -eq $Bootstrap) {
        throw "Python 3.11 or newer is required. Install Python, then retry."
    }
    Write-Host "[1/3] Creating the private Python environment..."
    & $Bootstrap.Source @BootstrapArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python environment creation failed with code $LASTEXITCODE."
    }
}
else {
    Write-Host "[1/3] Existing Python environment found."
}

Write-Host "[2/3] Installing the Vulkan and visualization runtime..."
& $EnvironmentPython -m pip install `
    --disable-pip-version-check `
    -r $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "Runtime installation failed with code $LASTEXITCODE."
}

Write-Host "[3/3] Checking the installation..."
& $EnvironmentPython -c `
    "import matplotlib, numpy, slangpy; print('      Python runtime ready.')"
if ($LASTEXITCODE -ne 0) {
    throw "Runtime verification failed with code $LASTEXITCODE."
}

Write-Host ""
Write-Host "  Setup complete." -ForegroundColor Green
Write-Host "  Double-click 'Launch Tesseract Studio.cmd' to begin."
Write-Host ""
