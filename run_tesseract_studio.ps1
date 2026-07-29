param(
    [int]$Port = 8765,
    [switch]$Demo,
    [switch]$Foreground,
    [switch]$NoBrowser,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv-slang\Scripts\python.exe"
$PythonWindowed = Join-Path $RepoRoot ".venv-slang\Scripts\pythonw.exe"
$Url = "http://127.0.0.1:$Port/"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv-slang. Create it and install requirements-slang.txt first."
}

function Test-StudioEndpoint {
    param([string]$Endpoint)
    try {
        $response = Invoke-WebRequest `
            -Uri "${Url}${Endpoint}" `
            -UseBasicParsing `
            -TimeoutSec 1
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Stop-ProjectStudio {
    $ScriptPath = Join-Path $RepoRoot "tesseract_studio.py"
    $PortPattern = "--port\s+$Port(?:\s|$)"
    $Processes = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and
        $_.CommandLine.Contains($ScriptPath) -and
        $_.CommandLine -match $PortPattern
    }
    foreach ($Process in $Processes) {
        Stop-Process -Id $Process.ProcessId -Force
    }
    if ($Processes) {
        Start-Sleep -Milliseconds 300
    }
}

if ($Restart) {
    Stop-ProjectStudio
}

$StudioRunning = Test-StudioEndpoint "api/status"
$ObservatoryReady = Test-StudioEndpoint "api/scenes"
if ($StudioRunning -and -not $ObservatoryReady) {
    throw "An older Tesseract Studio is using port $Port. Run this launcher with -Restart."
}

if (-not $StudioRunning) {
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
        if (
            (Test-StudioEndpoint "api/status") -and
            (Test-StudioEndpoint "api/scenes")
        ) {
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
