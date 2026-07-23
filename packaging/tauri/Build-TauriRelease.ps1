[CmdletBinding()]
param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$OutputDirectory,
    [ValidateSet("x64", "arm64")]
    [string]$Architecture = "x64",
    [string]$Channel = "stable",
    [int64]$SourceDateEpoch = 0,
    [string]$MinInstalledVersion,
    [string]$MaxInstalledVersion,
    [int]$DataSchemaReadableMin = 1,
    [int]$DataSchemaReadableMax = 1,
    [int]$DataSchemaWriteVersion = 1,
    [string]$PublisherThumbprint,
    [switch]$AllowUnsignedDevelopment,
    [switch]$SkipCompile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-ResultAndExit {
    param($Result, [int]$Code)
    $json = $Result | ConvertTo-Json -Depth 32
    if ($Code -eq 0) {
        [Console]::Out.WriteLine($json)
    }
    else {
        [Console]::Error.WriteLine(($Result | ConvertTo-Json -Depth 32 -Compress))
    }
    exit $Code
}

try {
    $project = [System.IO.Path]::GetFullPath($ProjectRoot)
    $desktopRoot = Join-Path $project "apps\desktop"
    $tauriRoot = Join-Path $desktopRoot "src-tauri"
    $packageJsonPath = Join-Path $desktopRoot "package.json"
    $tauriConfigPath = Join-Path $tauriRoot "tauri.conf.json"
    $cargoTomlPath = Join-Path $tauriRoot "Cargo.toml"
    $cargoLockPath = Join-Path $tauriRoot "Cargo.lock"
    $packageLockPath = Join-Path $desktopRoot "package-lock.json"

    foreach ($required in @($packageJsonPath, $tauriConfigPath, $cargoTomlPath, $cargoLockPath, $packageLockPath)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required locked build input is missing: $required"
        }
    }

    $packageJson = Get-Content -LiteralPath $packageJsonPath -Raw | ConvertFrom-Json
    $tauriConfig = Get-Content -LiteralPath $tauriConfigPath -Raw | ConvertFrom-Json
    $cargoToml = Get-Content -LiteralPath $cargoTomlPath -Raw
    $cargoVersionMatch = [regex]::Match($cargoToml, "(?ms)^\[package\].*?^version\s*=\s*""([^""]+)""")
    if (-not $cargoVersionMatch.Success) {
        throw "Unable to read [package].version from Cargo.toml."
    }

    $versions = @(
        @(
            [string]$packageJson.version,
            [string]$tauriConfig.version,
            [string]$cargoVersionMatch.Groups[1].Value
        ) | Select-Object -Unique
    )
    if ($versions.Count -ne 1) {
        throw "package.json, tauri.conf.json, and Cargo.toml versions must match."
    }
    $version = [string]$versions[0]

    $targetTriple = if ($Architecture -eq "arm64") {
        "aarch64-pc-windows-msvc"
    }
    else {
        "x86_64-pc-windows-msvc"
    }
    $entrypoint = "media-transcribe-studio.exe"
    $targetRoot = Join-Path $tauriRoot ("target\" + $targetTriple + "\release")
    $fallbackTargetRoot = Join-Path $tauriRoot "target\release"
    $payloadSource = Join-Path $targetRoot $entrypoint
    if (-not (Test-Path -LiteralPath $payloadSource) -and $Architecture -eq "x64") {
        $payloadSource = Join-Path $fallbackTargetRoot $entrypoint
    }

    if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        $OutputDirectory = Join-Path $project ("dist\tauri-release\" + $version + "-" + $Architecture)
    }
    $output = [System.IO.Path]::GetFullPath($OutputDirectory)

    $commands = @(
        "npm ci",
        ("npm run tauri -- build --target {0} --bundles nsis,msi" -f $targetTriple)
    )

    if ($DryRun) {
        Write-ResultAndExit ([ordered]@{
            ok = $true
            action = "BuildTauriRelease"
            dryRun = $true
            projectRoot = $project
            version = $version
            architecture = $Architecture
            targetTriple = $targetTriple
            outputDirectory = $output
            lockedInputs = @($packageLockPath, $cargoLockPath)
            commands = if ($SkipCompile) { @() } else { $commands }
            postBuild = @(
                "verify-version-coherence",
                "collect-tauri-native-executable",
                "collect-nsis-and-msi-installers",
                "verify-authenticode-or-explicit-development-mode",
                "emit-byte-hashed-release-manifest"
            )
        }) 0
    }

    if (-not $SkipCompile) {
        Push-Location $desktopRoot
        try {
            & npm ci
            if ($LASTEXITCODE -ne 0) {
                throw "npm ci failed with exit code $LASTEXITCODE."
            }
            & npm run tauri -- build --target $targetTriple --bundles "nsis,msi"
            if ($LASTEXITCODE -ne 0) {
                throw "Tauri build failed with exit code $LASTEXITCODE."
            }
        }
        finally {
            Pop-Location
        }
    }

    if (-not (Test-Path -LiteralPath $payloadSource -PathType Leaf)) {
        throw "Tauri native executable was not produced at the expected path: $payloadSource"
    }

    $temporaryPayload = Join-Path ([System.IO.Path]::GetTempPath()) ("mts-tauri-payload-" + [guid]::NewGuid().ToString("N"))
    $temporaryInstallers = Join-Path ([System.IO.Path]::GetTempPath()) ("mts-tauri-installers-" + [guid]::NewGuid().ToString("N"))
    try {
        New-Item -ItemType Directory -Path $temporaryPayload | Out-Null
        Copy-Item -LiteralPath $payloadSource -Destination (Join-Path $temporaryPayload $entrypoint)

        $installerCandidates = @(
            Get-ChildItem -LiteralPath $targetRoot -File -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.Extension -in @(".msi", ".exe") -and $_.FullName -ne $payloadSource }
        )
        $nativeInstallerDirectory = $null
        if ($installerCandidates.Count -gt 0) {
            New-Item -ItemType Directory -Path $temporaryInstallers | Out-Null
            foreach ($installer in $installerCandidates) {
                Copy-Item -LiteralPath $installer.FullName -Destination (Join-Path $temporaryInstallers $installer.Name)
            }
            $nativeInstallerDirectory = $temporaryInstallers
        }

        $gitCommit = "unknown"
        try {
            $gitCommit = (& git -C $project rev-parse HEAD).Trim()
        }
        catch {
            # A source archive may not contain Git metadata; "unknown" remains explicit.
        }

        $releaseScript = Join-Path $PSScriptRoot "New-TauriRelease.ps1"
        $releaseArguments = @{
            PayloadDirectory = $temporaryPayload
            OutputDirectory = $output
            Version = $version
            EntryPoint = $entrypoint
            AppId = [string]$tauriConfig.identifier
            ProductName = [string]$tauriConfig.productName
            Architecture = $Architecture
            Channel = $Channel
            SourceDateEpoch = $SourceDateEpoch
            GitCommit = $gitCommit
            TargetTriple = $targetTriple
            DataSchemaReadableMin = $DataSchemaReadableMin
            DataSchemaReadableMax = $DataSchemaReadableMax
            DataSchemaWriteVersion = $DataSchemaWriteVersion
            AllowUnsignedDevelopment = [bool]$AllowUnsignedDevelopment
        }
        if (-not [string]::IsNullOrWhiteSpace($MinInstalledVersion)) {
            $releaseArguments.MinInstalledVersion = $MinInstalledVersion
        }
        if (-not [string]::IsNullOrWhiteSpace($MaxInstalledVersion)) {
            $releaseArguments.MaxInstalledVersion = $MaxInstalledVersion
        }
        if (-not [string]::IsNullOrWhiteSpace($PublisherThumbprint)) {
            $releaseArguments.PublisherThumbprint = $PublisherThumbprint
        }
        if ($null -ne $nativeInstallerDirectory) {
            $releaseArguments.NativeInstallerDirectory = $nativeInstallerDirectory
        }

        & $releaseScript @releaseArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Release manifest generation failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        foreach ($temporary in @($temporaryPayload, $temporaryInstallers)) {
            if (Test-Path -LiteralPath $temporary) {
                Remove-Item -LiteralPath $temporary -Recurse -Force
            }
        }
    }
}
catch {
    Write-ResultAndExit ([ordered]@{
        ok = $false
        action = "BuildTauriRelease"
        error = $_.Exception.Message
    }) 1
}
