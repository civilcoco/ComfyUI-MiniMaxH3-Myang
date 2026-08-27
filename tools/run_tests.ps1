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

$runner = 'import runpy,sys; sys.path.insert(0,sys.argv[1]); runpy.run_path(sys.argv[2],run_name="__main__")'
$pythonTests = @(
    "test_native_anchors.py",
    "test_native_boundaries.py",
    "test_media_agent.py",
    "test_script_splitter_pipeline.py",
    "test_splitter_with_media.py",
    "test_llm_service_config.py",
    "test_latent_upscale_detail.py",
    "test_taper_and_detail_continuity.py",
    "test_turbo_cache_and_detail_boost.py",
    "test_director_and_neural_upscale.py"
)

foreach ($name in $pythonTests) {
    Write-Host "RUN tests/$name"
    & $Python -B -c $runner $comfyPath (Join-Path $packageRoot "tests\$name")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (Get-Command node -ErrorAction SilentlyContinue) {
    foreach ($name in @("test_storyboard_cards.mjs", "test_progress_state.mjs", "test_llm_config_ui.mjs")) {
        Write-Host "RUN tests/$name"
        & node --experimental-default-type=module (Join-Path $packageRoot "tests\$name")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} else {
    Write-Warning "Node.js is unavailable; frontend structure tests were skipped."
}

& $Python -B (Join-Path $packageRoot "tools\release_audit.py")
exit $LASTEXITCODE
