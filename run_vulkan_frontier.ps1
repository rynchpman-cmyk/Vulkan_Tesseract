param(
    [string]$Python = ".\.venv-slang\Scripts\python.exe",
    [string]$Artifact = "theory33_frozen_variants.json",
    [string]$Output = "vulkan_frontier_results.json"
)

& $Python .\run_vulkan_frontier.py `
    --artifact $Artifact `
    --output $Output
exit $LASTEXITCODE
