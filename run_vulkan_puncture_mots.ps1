param(
    [int]$Resolution = 120,
    [double]$BoxLength = 3.0,
    [double]$Separation = 0.78,
    [int]$Steps = 500,
    [double]$CFL = 0.04,
    [int]$ObservationCadence = 10,
    [int]$CheckpointCadence = 50,
    [int]$MaximumDegree = 16,
    [int]$IndividualMaximumDegree = 8,
    [int]$MaximumIterations = 60,
    [double]$ExpansionTolerance = 5e-4,
    [string]$OutputDirectory = "runs/vulkan_ccz4_puncture_mots_n120",
    [string]$RestartFrom = "",
    [switch]$PlanOnly,
    [switch]$SkipConstraints,
    [switch]$SkipIndividual,
    [switch]$SaveSnapshots
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv-slang\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing .venv-slang. Install requirements-slang.txt first."
}
$OutputPath = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory
}
else {
    Join-Path $RepoRoot $OutputDirectory
}

$Arguments = @(
    (Join-Path $RepoRoot "run_vulkan_puncture_mots.py"),
    "--resolution", $Resolution,
    "--box-length", $BoxLength,
    "--separation", $Separation,
    "--steps", $Steps,
    "--cfl", $CFL,
    "--observation-cadence", $ObservationCadence,
    "--checkpoint-cadence", $CheckpointCadence,
    "--maximum-degree", $MaximumDegree,
    "--individual-maximum-degree", $IndividualMaximumDegree,
    "--maximum-iterations", $MaximumIterations,
    "--expansion-tolerance", $ExpansionTolerance,
    "--output-directory", $OutputPath
)
if ($RestartFrom) {
    $RestartPath = if ([System.IO.Path]::IsPathRooted($RestartFrom)) {
        $RestartFrom
    }
    else {
        Join-Path $RepoRoot $RestartFrom
    }
    $Arguments += @("--restart-from", $RestartPath)
}
if ($PlanOnly) {
    $Arguments += "--plan-only"
}
if ($SkipConstraints) {
    $Arguments += "--skip-constraints"
}
if ($SkipIndividual) {
    $Arguments += "--skip-individual"
}
if ($SaveSnapshots) {
    $Arguments += "--save-snapshots"
}

& $Python @Arguments
exit $LASTEXITCODE
