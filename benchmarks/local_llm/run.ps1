param(
    [string]$Model = "qwen2.5:1.5b",
    [int]$MaxDev = 48,
    [int]$MaxHeldout = 24,
    [int]$MaxSafety = 16,
    [int]$Seed = 20260721,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"

$required = @(
    "MTS_BENCH_FINAL_TRANSCRIPT",
    "MTS_BENCH_PRE_TRANSCRIPT",
    "MTS_BENCH_TURN_CORRECTIONS",
    "MTS_BENCH_SENTENCE_DECISIONS"
)

foreach ($name in $required) {
    if (-not [Environment]::GetEnvironmentVariable($name)) {
        throw "Missing required environment variable: $name"
    }
}

Push-Location $PSScriptRoot
try {
    & $Python -m pytest -q -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) {
        throw "Unit tests failed with exit code $LASTEXITCODE"
    }

    & $Python run_benchmark.py `
        --model $Model `
        --max-dev $MaxDev `
        --max-heldout $MaxHeldout `
        --max-safety $MaxSafety `
        --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
