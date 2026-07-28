param(
    [string]$Python = ".\.venv-slang\Scripts\python.exe",
    [int]$Points = 64,
    [int]$Steps = 8,
    [string]$Output = "vulkan_tov_results.json"
)

& $Python .\run_vulkan_tov.py --points $Points --steps $Steps --output $Output
exit $LASTEXITCODE
