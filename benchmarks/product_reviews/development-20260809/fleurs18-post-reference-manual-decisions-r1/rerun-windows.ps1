param(
    [string]$Config = "D:\mts-eval\configs\product-semantic-candidates-20260807\ollama-qwen3.6-27b-q4-k-m.json",
    [string]$RunRoot = "D:\mts-eval\product-runs\pilot-fleurs18-codex-adjudicated-qwen36-27b-r3-20260809"
)

$ErrorActionPreference = "Stop"
$Repo = "D:\MediaTranscribeStudio"
$Python = Join-Path $Repo "runtime\media-asr\python.exe"
$Manifest = Join-Path $Repo ".runtime_cache\sample-library\global\blind-product-fleurs-development.v1.json"
$DecisionRoot = Join-Path $Repo "benchmarks\product_reviews\development-20260809\fleurs18-post-reference-manual-decisions-r1"
$Cases = @(
    "fleurs_ar_eg_validation_030",
    "fleurs_ar_eg_validation_060",
    "fleurs_ar_eg_validation_090",
    "fleurs_es_419_validation_009",
    "fleurs_es_419_validation_080",
    "fleurs_es_419_validation_232",
    "fleurs_hi_in_validation_030",
    "fleurs_hi_in_validation_060",
    "fleurs_hi_in_validation_090",
    "fleurs_ja_jp_validation_030",
    "fleurs_ja_jp_validation_060",
    "fleurs_ja_jp_validation_090",
    "fleurs_ko_kr_validation_030",
    "fleurs_ko_kr_validation_060",
    "fleurs_ko_kr_validation_090",
    "fleurs_yue_hant_hk_validation_030",
    "fleurs_yue_hant_hk_validation_060",
    "fleurs_yue_hant_hk_validation_090"
)

foreach ($RequiredPath in @($Python, $Config, $Manifest, $DecisionRoot)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path is missing: $RequiredPath"
    }
}
if (Test-Path -LiteralPath $RunRoot) {
    throw "Run root already exists; choose a new no-replace path: $RunRoot"
}

$RunArguments = @(
    "-m", "tools.run_sample_library",
    "--manifest", $Manifest,
    "--config", $Config,
    "--results-root", (Join-Path $RunRoot "results"),
    "--worker-output-root", (Join-Path $RunRoot "outputs"),
    "--speaker-count-mode", "manual",
    "--language-mode", "reference",
    "--local-llm-mode", "suggestion-only",
    "--require-semantic-composition",
    "--reuse-worker",
    "--max-jobs-per-worker-session", "4",
    "--idle-timeout-seconds", "900",
    "--cold-start-p95-seconds", "1200",
    "--rtf-p95", "180",
    "--deadline-safety-seconds", "300",
    "--minimum-hard-timeout-seconds", "1800",
    "--maximum-hard-timeout-seconds", "14400"
)
foreach ($CaseId in $Cases) {
    $DecisionPath = Join-Path $DecisionRoot "$CaseId.review-decisions.json"
    if (-not (Test-Path -LiteralPath $DecisionPath)) {
        throw "Review decision file is missing: $DecisionPath"
    }
    $RunArguments += @("--case", $CaseId)
    $RunArguments += @("--review-decisions", "$CaseId=$DecisionPath")
}

Set-Location -LiteralPath $Repo
& $Python @RunArguments
if ($LASTEXITCODE -ne 0) {
    throw "FLEURS 18-case adjudicated product run failed with exit code $LASTEXITCODE"
}
