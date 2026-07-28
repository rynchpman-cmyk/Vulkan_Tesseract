param(
    [string]$Python = ".\.venv-slang\Scripts\python.exe",
    [string]$ReferenceRoot = "..\TESSERACT",
    [string]$Output = "vulkan_parity_results.json"
)

& $Python .\run_vulkan_parity.py `
    --reference-root $ReferenceRoot `
    --output $Output
exit $LASTEXITCODE
