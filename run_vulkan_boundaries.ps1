param(
    [string]$Python = ".\.venv-slang\Scripts\python.exe",
    [int]$Points = 48,
    [string]$Output = "vulkan_boundary_results.json"
)

& $Python .\run_vulkan_boundaries.py --points $Points --output $Output
exit $LASTEXITCODE
