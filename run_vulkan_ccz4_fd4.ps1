param(
    [string]$Python = ".\.venv-slang\Scripts\python.exe",
    [string]$Output = "vulkan_ccz4_fd4_results.json"
)

& $Python .\run_vulkan_ccz4_fd4.py --output $Output
exit $LASTEXITCODE
