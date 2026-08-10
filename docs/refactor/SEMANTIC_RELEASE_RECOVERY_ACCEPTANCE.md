# Semantic Release, Recovery, and Held-Out Acceptance

This runbook is for Windows PowerShell and keeps all mutable evidence on `D:`.
It does not grant the incumbent any release protection. `production` is only
the model selected by the current config pointer. Any compatible challenger
that wins a same-case-batch blind review may replace it immediately through
compare-and-swap (CAS), regardless of registry status. There is no required
margin, cooldown, protection period, or incumbent veto.

Do not mark the rollback or held-out items in `TASKS.md` complete from unit
tests. Check them only after the Windows commands below have produced the
referenced immutable receipts and review evidence.

## Common PowerShell setup

Use a new evidence directory for every real attempt. None of the receipt,
package, review, or authorization paths below may already exist.

```powershell
$Repo = 'D:\MediaTranscribeStudio'
$Eval = 'D:\mts-eval\semantic-release-20260809'
$Python = "$Repo\runtime\media-asr\python.exe"
$Java = "$Repo\.toolchain\jdk17\jdk-17.0.20+8\bin\java.exe"
$Registry = "$Repo\local-model-registry.json"
$Config = "$Repo\production.config.json"
$Evidence = "$Eval\evidence"
$RollbackRoot = "$Eval\production-config-rollbacks"
$RecoveryRoot = "$Eval\failed-active-configs"
New-Item -ItemType Directory -Force -Path $Evidence, $RollbackRoot, $RecoveryRoot | Out-Null

function Get-Sha256([string]$Path) {
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Utf8NoBom([string]$Path, [string[]]$Lines) {
  $Text = $Lines -join [Environment]::NewLine
  [IO.File]::WriteAllText($Path, $Text + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
}
```

## Verify local registry artifacts

Verify both the active model and every model that might be restored or
promoted. This hashes the registered manifests and local model blobs; it does
not run inference.

```powershell
& $Python "$Repo\tools\model_registry.py" `
  --registry $Registry --verify-local `
  --model '<active-registry-model-id>' `
  --model '<rollback-registry-model-id>' `
  --model '<challenger-registry-model-id>'
if ($LASTEXITCODE -ne 0) { throw 'Local model verification failed.' }
```

Save the registry and active config identities with the release evidence:

```powershell
Get-Sha256 $Registry | Set-Content "$Evidence\registry.sha256.txt"
Get-Sha256 $Config | Set-Content "$Evidence\pre-action-production-config.sha256.txt"
```

## Promote a blind-review winner

The decision must name the challenger as winner, bind the exact current config
SHA-256, and prove that incumbent and challenger were reviewed blindly on the
same frozen case batch. Capture stdout because it is the receipt required for
rollback.

```powershell
$Decision = "$Evidence\semantic-promotion-decision.v1.json"
$BlindReview = "$Evidence\blind-product-review.v1.json"
$Comparison = "$Evidence\semantic-model-comparison.v1.json"
$PromotionReceipt = "$Evidence\semantic-promotion-receipt.v1.json"
$ExpectedConfigSha256 = Get-Sha256 $Config

$PromotionLines = & $Python "$Repo\tools\promote_semantic_model.py" `
  --config $Config `
  --registry $Registry `
  --decision $Decision `
  --blind-review-artifact $BlindReview `
  --comparison-artifact $Comparison `
  --rollback-root $RollbackRoot `
  --expected-config-sha256 $ExpectedConfigSha256
if ($LASTEXITCODE -ne 0) { throw 'Semantic promotion failed.' }
Write-Utf8NoBom $PromotionReceipt $PromotionLines

$Promotion = Get-Content -LiteralPath $PromotionReceipt -Raw | ConvertFrom-Json
if ((Get-Sha256 $Config) -ne $Promotion.promotedConfigSha256) {
  throw 'Active config does not match the promotion receipt.'
}
Get-Sha256 $PromotionReceipt | Set-Content "$PromotionReceipt.sha256.txt"
Get-Sha256 $Promotion.rollbackConfigPath | Set-Content "$Evidence\rollback-config.sha256.txt"
```

## Isolated rollback drill

Run this before a live release. The drill modifies only a byte-for-byte copy of
the promoted config. It still verifies the real local rollback model and the
receipt-bound immutable rollback snapshot.

```powershell
$DrillRoot = "$Eval\rollback-drill-$(Get-Date -Format yyyyMMdd-HHmmss)"
$DrillConfig = "$DrillRoot\production.config.json"
$DrillRecovery = "$DrillRoot\failed-active-configs"
$DrillReceipt = "$DrillRoot\rollback-receipt.v1.json"
New-Item -ItemType Directory -Path $DrillRoot, $DrillRecovery | Out-Null
Copy-Item -LiteralPath $Config -Destination $DrillConfig

$DrillActiveSha256 = Get-Sha256 $DrillConfig
$DrillLines = & $Python "$Repo\tools\rollback_semantic_model.py" `
  --config $DrillConfig `
  --registry $Registry `
  --promotion-receipt $PromotionReceipt `
  --rollback-config $Promotion.rollbackConfigPath `
  --recovery-root $DrillRecovery `
  --expected-active-config-sha256 $DrillActiveSha256 `
  --reason operator-drill
if ($LASTEXITCODE -ne 0) { throw 'Isolated rollback drill failed.' }
Write-Utf8NoBom $DrillReceipt $DrillLines

$Drill = Get-Content -LiteralPath $DrillReceipt -Raw | ConvertFrom-Json
if ((Get-Sha256 $DrillConfig) -ne $Promotion.previousConfigSha256) {
  throw 'The isolated config was not restored byte-for-byte.'
}
if ((Get-Sha256 $Drill.failedActiveConfigSnapshotPath) -ne $DrillActiveSha256) {
  throw 'The failed active config was not preserved byte-for-byte.'
}
Get-Sha256 $DrillReceipt | Set-Content "$DrillReceipt.sha256.txt"
```

Required drill evidence is the promotion receipt, rollback receipt, restored
config hash, failed-active snapshot hash, registry hash, and the unchanged live
config hash. A successful drill does not by itself approve the model.

## Live rollback after an authorized failure

Select one explicit reason: `active-model-unavailable`,
`artifact-integrity-failure`, `operator-drill`, `resource-exhaustion`, or
`startup-failure`. The expected hash must be measured immediately before the
command. A stale pointer fails without replacing it.

```powershell
$LiveRollbackReceipt = "$Evidence\live-rollback-receipt.v1.json"
$ActiveSha256 = Get-Sha256 $Config
$RollbackLines = & $Python "$Repo\tools\rollback_semantic_model.py" `
  --config $Config `
  --registry $Registry `
  --promotion-receipt $PromotionReceipt `
  --rollback-config $Promotion.rollbackConfigPath `
  --recovery-root $RecoveryRoot `
  --expected-active-config-sha256 $ActiveSha256 `
  --reason startup-failure
if ($LASTEXITCODE -ne 0) { throw 'Live semantic rollback failed.' }
Write-Utf8NoBom $LiveRollbackReceipt $RollbackLines

$LiveRollback = Get-Content -LiteralPath $LiveRollbackReceipt -Raw | ConvertFrom-Json
if ((Get-Sha256 $Config) -ne $LiveRollback.restoredConfigSha256) {
  throw 'Live rollback receipt does not match the active pointer.'
}
Get-Sha256 $LiveRollbackReceipt | Set-Content "$LiveRollbackReceipt.sha256.txt"
```

Rollback gives the restored model no protection. After any challenger wins a
new same-batch blind decision, invoke `promote_semantic_model.py` again with the
restored config's current SHA-256. The CAS may replace the restored model
immediately; registry `production`/`challenger` labels do not block it.

```powershell
$NextExpectedSha256 = Get-Sha256 $Config
$NextPromotionLines = & $Python "$Repo\tools\promote_semantic_model.py" `
  --config $Config `
  --registry $Registry `
  --decision "$Evidence\next-challenger-decision.v1.json" `
  --blind-review-artifact "$Evidence\next-blind-product-review.v1.json" `
  --comparison-artifact "$Evidence\next-model-comparison.v1.json" `
  --rollback-root $RollbackRoot `
  --expected-config-sha256 $NextExpectedSha256
if ($LASTEXITCODE -ne 0) { throw 'Immediate challenger replacement failed.' }
Write-Utf8NoBom "$Evidence\next-promotion-receipt.v1.json" $NextPromotionLines
```

## Held-out review ordering

Never inspect the scorer truth, identity vault, reference transcript, or
automatic scores before the no-replace unblind authorization is published.
The authorizer accepts only digest commitments for the held-out freeze and
identity vault; it never opens either artifact.

First freeze the truth-redacted held-out matrix and record its digest. The
truth-bearing scorer vault remains in a separate location and process.

```powershell
$HeldOutFreeze = "$Eval\held-out\truth-redacted-held-out.v1.json"
$HeldOutFreezeSha256 = Get-Sha256 $HeldOutFreeze
$WinnerConfig = "$Eval\held-out\winner.production.config.json"
$RollbackConfig = $Promotion.rollbackConfigPath
$Recipe = "$Repo\configs\product-audio-e2e-output-recipe.v1.json"
Copy-Item -LiteralPath $Config -Destination $WinnerConfig
```

Run the winner and rollback configurations on that exact same frozen manifest.
Do not change case selection, media, output recipe, locks, or acoustic evidence
between runs.

```powershell
$WinnerRun = "$Eval\held-out\candidate-a"
$RollbackRun = "$Eval\held-out\candidate-b"

& $Python "$Repo\tools\run_sample_library.py" `
  --manifest $HeldOutFreeze --config $WinnerConfig `
  --results-root "$WinnerRun\results" `
  --worker-output-root "$WinnerRun\outputs" `
  --speaker-count-mode auto --language-mode auto `
  --local-llm-mode business --require-semantic-composition `
  --output-recipe $Recipe
if ($LASTEXITCODE -ne 0) { throw 'Winner held-out run failed.' }

& $Python "$Repo\tools\run_sample_library.py" `
  --manifest $HeldOutFreeze --config $RollbackConfig `
  --results-root "$RollbackRun\results" `
  --worker-output-root "$RollbackRun\outputs" `
  --speaker-count-mode auto --language-mode auto `
  --local-llm-mode business --require-semantic-composition `
  --output-recipe $Recipe
if ($LASTEXITCODE -ne 0) { throw 'Rollback held-out run failed.' }
```

Build the blind package. Candidate labels, original paths, model identities,
references, and automatic scores are retained only in `identity-vault`; the
reviewer receives only `reviewer-packet`.

```powershell
$BlindPackage = "$Eval\held-out\blind-e2e-package"
$BlindSeed = '<new sealed random seed>'
& $Python "$Repo\tools\build_blind_e2e_review_package.py" `
  --candidate "candidate-a=$WinnerRun" `
  --candidate "candidate-b=$RollbackRun" `
  --output-root $BlindPackage --seed $BlindSeed `
  --java-executable $Java `
  --pdf-scanner-jar "$Repo\pdf-renderer\target\pdf-renderer.jar"
if ($LASTEXITCODE -ne 0) { throw 'Blind package creation failed.' }

$PackageManifest = "$BlindPackage\package-manifest.v1.json"
$PackageManifestSha256 = Get-Sha256 $PackageManifest
$PackageManifestSha256 | Set-Content "$Evidence\held-out-package-manifest.sha256.txt"
```

Review only `$BlindPackage\reviewer-packet`. Fill a copy of
`human-review-form.v1.json` outside the immutable package. Every case must name
a preferred option or explicit tie and complete speaker timeline, raw ASR,
final transcript, and subtitle/PDF dimensions with severity, reason, and
timestamped evidence.

```powershell
$CompletedReview = "$Evidence\held-out-completed-review.v1.json"
Copy-Item `
  -LiteralPath "$BlindPackage\reviewer-packet\human-review-form.v1.json" `
  -Destination $CompletedReview
# Complete $CompletedReview without opening identity-vault or scorer truth.
```

Seal the completed review before any unblind or objective scoring. The output
must also be outside the immutable package and must not already exist.

```powershell
$Authorization = "$Evidence\held-out-unblind-authorization.v1.json"
$ReviewedAt = (Get-Date).ToString('o')
& $Python "$Repo\tools\authorize_held_out_unblind.py" `
  --package-root $BlindPackage `
  --completed-review $CompletedReview `
  --held-out-freeze-manifest-sha256 $HeldOutFreezeSha256 `
  --expected-package-manifest-sha256 $PackageManifestSha256 `
  --reviewer-source codex-agent `
  --reviewer 'Codex held-out reviewer' `
  --reviewed-at $ReviewedAt `
  --output $Authorization
if ($LASTEXITCODE -ne 0) { throw 'Held-out unblind authorization failed.' }
Get-Sha256 $CompletedReview | Set-Content "$CompletedReview.sha256.txt"
Get-Sha256 $Authorization | Set-Content "$Authorization.sha256.txt"
```

Only after that command succeeds may a separate process open
`identity-vault\unblind-mapping.v1.json` and the held-out scorer vault, compute
objective metrics, and compare them with the already sealed review. A failed
held-out result does not protect the incumbent or authorize tuning on those
cases: keep the current pointer, return to development data, and freeze a new
untouched held-out set for any later final confirmation.
