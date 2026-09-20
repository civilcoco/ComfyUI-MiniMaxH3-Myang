param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyRoot,
    [string]$Python
)

$ErrorActionPreference = "Stop"
$packageRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$comfyPath = [System.IO.Path]::GetFullPath($ComfyRoot)
if (-not (Test-Path -LiteralPath (Join-Path $comfyPath "nodes.py"))) {
    throw "ComfyRoot must point to a ComfyUI checkout: $comfyPath"
}

if (-not $Python) {
    $embedded = Join-Path (Split-Path -Parent $comfyPath) "python\python.exe"
    $Python = if (Test-Path -LiteralPath $embedded) { $embedded } else { "python" }
}

# Always test a sanitized Git snapshot. This excludes a user's private Skills,
# learned caches and research notes, which are intentionally ignored by the
# package and must not affect the release result.
& $Python -B (Join-Path $packageRoot "tools\run_suite.py") --comfy-root $comfyPath
exit $LASTEXITCODE
