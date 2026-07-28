param(
    [string]$Python = ".\.venv-slang\Scripts\python.exe",
    [string]$Output = "vulkan_amr_results.json"
)

& $Python .\run_vulkan_amr.py --output $Output
exit $LASTEXITCODE
