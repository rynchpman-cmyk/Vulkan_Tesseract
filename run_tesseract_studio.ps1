param(
    [int]$Port = 8765,
    [switch]$Demo,
    [switch]$Foreground,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv-slang\Scripts\python.exe"
$PythonWindowed = Join-Path $RepoRoot ".venv-slang\Scripts\pythonw.exe"
$Url = "http://127.0.0.1:$Port/"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv-slang. Create it and install requirements-slang.txt first."
}

function Test-StudioRunning {
    try {
        $response = Invoke-WebRequest `
            -Uri "${Url}api/status" `
            -UseBasicParsing `
            -TimeoutSec 1
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

if (-not (Test-StudioRunning)) {
    $Arguments = @(
        (Join-Path $RepoRoot "tesseract_studio.py"),
        "--port",
        $Port
    )
    if ($Demo) {
        $Arguments += "--demo"
    }
    if ($NoBrowser -or -not $Foreground) {
        $Arguments += "--no-browser"
    }

    if ($Foreground) {
        & $Python @Arguments
        exit $LASTEXITCODE
    }

    if (-not (Test-Path -LiteralPath $PythonWindowed)) {
        $PythonWindowed = $Python
    }
    Start-Process `
        -FilePath $PythonWindowed `
        -ArgumentList $Arguments `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden

    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 200
        if (Test-StudioRunning) {
            $ready = $true
            break
        }
    }
    if (-not $ready) {
        throw "Tesseract Studio did not start. See runs/studio/studio.log."
    }
}

if (-not $NoBrowser) {
    Start-Process $Url
}
