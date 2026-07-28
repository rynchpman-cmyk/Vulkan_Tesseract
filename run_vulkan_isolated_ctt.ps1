param(
    [string]$Python = ".\.venv-slang\Scripts\python.exe",
    [string]$Shape = "10,10,10",
    [string]$Output = "vulkan_isolated_ctt_results.json"
)

& $Python .\run_vulkan_isolated_ctt.py --shape $Shape --output $Output
exit $LASTEXITCODE
