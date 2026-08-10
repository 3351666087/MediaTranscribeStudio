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
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [string]$TargetDirectory,
    [string]$PortableArchivePath,
    [switch]$AllowUnsignedDevelopment,
    [switch]$SkipCompile,
    [switch]$Force,
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

function Get-AbsolutePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Get-RelativePathFromRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $rootPath = (Get-AbsolutePath $Root).TrimEnd("\", "/") + "\"
    $pathValue = Get-AbsolutePath $Path
    if (-not $pathValue.StartsWith($rootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes its declared root: $Path"
    }
    return $pathValue.Substring($rootPath.Length).Replace("\", "/")
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )
    [System.IO.File]::WriteAllText(
        $Path,
        $Content,
        (New-Object System.Text.UTF8Encoding($false))
    )
}

function New-TauriWindowsOverlayConfig {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeSource,
        [string]$PublisherThumbprint,
        [string]$TimestampUrl
    )

    if (
        [string]::IsNullOrWhiteSpace($RuntimeSource) -or
        [System.IO.Path]::IsPathRooted($RuntimeSource) -or
        $RuntimeSource.Contains("\") -or
        $RuntimeSource.Split("/") -contains ".."
    ) {
        throw "Tauri runtime resource source must be a forward-slash relative path: $RuntimeSource"
    }

    $resources = [ordered]@{}
    # On Windows Tauri's $RESOURCES directory is beside the installed exe.
    # Keep the explicit resources/ prefix so worker_supervisor resolves the
    # same path for NSIS, MSI, and unpacked portable payloads.
    $resources[$RuntimeSource] = "resources/mts-runtime"
    $bundle = [ordered]@{ resources = $resources }
    if (-not [string]::IsNullOrWhiteSpace($PublisherThumbprint)) {
        $bundle.windows = [ordered]@{
            digestAlgorithm = "sha256"
            certificateThumbprint = $PublisherThumbprint.Replace(" ", "").ToUpperInvariant()
            timestampUrl = $TimestampUrl
        }
    }
    return [ordered]@{ bundle = $bundle }
}

function Write-TauriWindowsOverlay {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Config
    )

    Write-Utf8NoBom -Path $Path -Content (($config | ConvertTo-Json -Depth 16) + "`n")
}

function Get-Sha256Hex {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function New-LockedInputRecord {
    param([Parameter(Mandatory = $true)][string]$Path)
    $resolved = Get-AbsolutePath $Path
    return [ordered]@{
        path = $resolved
        size = [int64](Get-Item -LiteralPath $resolved).Length
        sha256 = Get-Sha256Hex -Path $resolved
    }
}

function Assert-LockedInputsUnchanged {
    param([Parameter(Mandatory = $true)][object[]]$Inputs)
    foreach ($inputRecord in $Inputs) {
        $path = [string]$inputRecord.path
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Locked build input disappeared during the build: $path"
        }
        $size = [int64](Get-Item -LiteralPath $path).Length
        $sha256 = Get-Sha256Hex -Path $path
        if ($size -ne [int64]$inputRecord.size -or $sha256 -ne [string]$inputRecord.sha256) {
            throw "Locked build input changed during the build: $path"
        }
    }
}

function New-DeterministicPortableArchive {
    param(
        [Parameter(Mandatory = $true)][string]$ReleaseRoot,
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [int64]$SourceDateEpoch = 0,
        [switch]$Force
    )

    $release = Get-AbsolutePath $ReleaseRoot
    $archive = Get-AbsolutePath $ArchivePath
    $checksum = "$archive.sha256"
    if (-not (Test-Path -LiteralPath $release -PathType Container)) {
        throw "Cannot create portable archive; release directory is missing: $release"
    }
    if ($archive.StartsWith($release.TrimEnd("\", "/") + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Portable archive must be outside the release directory: $archive"
    }
    if ((Test-Path -LiteralPath $archive) -and -not $Force) {
        throw "Portable archive already exists: $archive"
    }
    if ((Test-Path -LiteralPath $checksum) -and -not $Force) {
        throw "Portable archive checksum already exists: $checksum"
    }

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $parent = Split-Path -Parent $archive
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    $temporary = "$archive.$([guid]::NewGuid().ToString('N')).tmp"
    $epochFloor = 315532800L # 1980-01-01, the ZIP timestamp floor.
    $epochCeiling = 4354819199L # 2107-12-31, the ZIP timestamp ceiling.
    $effectiveEpoch = [Math]::Max($epochFloor, [Math]::Min($epochCeiling, $SourceDateEpoch))
    $zipTime = [DateTimeOffset]::FromUnixTimeSeconds($effectiveEpoch)
    $stream = $null
    $zip = $null
    $completed = $false
    try {
        try {
            $stream = New-Object System.IO.FileStream(
                $temporary,
                [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::None
            )
            $zip = New-Object System.IO.Compression.ZipArchive(
                $stream,
                [System.IO.Compression.ZipArchiveMode]::Create,
                $false
            )
            $files = @(Get-ChildItem -LiteralPath $release -File -Recurse -Force | Sort-Object FullName)
            foreach ($file in $files) {
                if (($file.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                    throw "Reparse points are forbidden in portable archives: $($file.FullName)"
                }
                $relative = Get-RelativePathFromRoot -Root $release -Path $file.FullName
                $entry = $zip.CreateEntry($relative, [System.IO.Compression.CompressionLevel]::Optimal)
                $entry.LastWriteTime = $zipTime
                $input = $null
                $output = $null
                try {
                    $input = [System.IO.File]::OpenRead($file.FullName)
                    $output = $entry.Open()
                    $input.CopyTo($output)
                }
                finally {
                    if ($null -ne $output) { $output.Dispose() }
                    if ($null -ne $input) { $input.Dispose() }
                }
            }
            $completed = $true
        }
        finally {
            if ($null -ne $zip) { $zip.Dispose() }
            if ($null -ne $stream) { $stream.Dispose() }
        }
        if (-not $completed) {
            throw "Portable archive creation did not complete: $archive"
        }
        Move-Item -LiteralPath $temporary -Destination $archive -Force
    }
    catch {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
        throw
    }

    Write-Utf8NoBom -Path $checksum -Content ((Get-Sha256Hex -Path $archive) + "`n")

    return [ordered]@{
        path = $archive
        checksumPath = $checksum
        size = [int64](Get-Item -LiteralPath $archive).Length
        sha256 = Get-Sha256Hex -Path $archive
        sourceDateEpoch = $SourceDateEpoch
        deterministic = $true
    }
}

try {
    $project = Get-AbsolutePath $ProjectRoot
    $desktopRoot = Join-Path $project "apps\desktop"
    $tauriRoot = Join-Path $desktopRoot "src-tauri"
    $packageJsonPath = Join-Path $desktopRoot "package.json"
    $tauriConfigPath = Join-Path $tauriRoot "tauri.conf.json"
    $cargoTomlPath = Join-Path $tauriRoot "Cargo.toml"
    $cargoLockPath = Join-Path $tauriRoot "Cargo.lock"
    $packageLockPath = Join-Path $desktopRoot "package-lock.json"
    $payloadBuilderPath = Join-Path $PSScriptRoot "New-WindowsReleasePayload.ps1"

    foreach ($required in @($packageJsonPath, $tauriConfigPath, $cargoTomlPath, $cargoLockPath, $packageLockPath, $payloadBuilderPath)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required locked build input is missing: $required"
        }
    }
    if ($SourceDateEpoch -lt 0) {
        throw "SourceDateEpoch must be non-negative."
    }
    if (-not $AllowUnsignedDevelopment -and [string]::IsNullOrWhiteSpace($PublisherThumbprint)) {
        throw "Production Windows builds require -PublisherThumbprint. Use -AllowUnsignedDevelopment only for local candidates."
    }
    if (-not [string]::IsNullOrWhiteSpace($PublisherThumbprint)) {
        $normalizedThumbprint = $PublisherThumbprint.Replace(" ", "").ToUpperInvariant()
        if ($normalizedThumbprint -notmatch "^[A-F0-9]{40,64}$") {
            throw "PublisherThumbprint is invalid."
        }
        $PublisherThumbprint = $normalizedThumbprint
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
    $targetTriple = if ($Architecture -eq "arm64") { "aarch64-pc-windows-msvc" } else { "x86_64-pc-windows-msvc" }
    $entrypoint = "media-transcribe-studio.exe"

    $targetBase = if (-not [string]::IsNullOrWhiteSpace($TargetDirectory)) {
        Get-AbsolutePath $TargetDirectory
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)) {
        Get-AbsolutePath $env:CARGO_TARGET_DIR
    }
    else {
        Get-AbsolutePath (Join-Path $tauriRoot "target")
    }
    $targetRoot = Join-Path $targetBase ("{0}\release" -f $targetTriple)
    $fallbackTargetRoot = Join-Path $targetBase "release"

    if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        $OutputDirectory = Join-Path $project ("dist\tauri-release\{0}-{1}" -f $version, $Architecture)
    }
    $output = Get-AbsolutePath $OutputDirectory
    if ([string]::IsNullOrWhiteSpace($PortableArchivePath)) {
        $PortableArchivePath = "$output.portable.zip"
    }
    $portableArchive = Get-AbsolutePath $PortableArchivePath
    if (-not $DryRun -and (Test-Path -LiteralPath $portableArchive) -and -not $Force) {
        throw "Portable archive already exists; choose a new path or pass -Force: $portableArchive"
    }
    if (-not $DryRun -and (Test-Path -LiteralPath "$portableArchive.sha256") -and -not $Force) {
        throw "Portable archive checksum already exists; choose a new path or pass -Force: $portableArchive.sha256"
    }

    $overlayPlaceholder = "<temporary-windows-tauri-overlay.json>"
    $commands = @(
        "npm ci",
        ("npm run tauri -- build --ci --target {0} --bundles nsis,msi --config {1} -- --locked" -f $targetTriple, $overlayPlaceholder)
    )
    $payloadPlanJson = & $payloadBuilderPath `
        -ProjectRoot $project `
        -OutputDirectory (Join-Path ([System.IO.Path]::GetTempPath()) "mts-windows-runtime-plan") `
        -RuntimeOnly `
        -DryRun
    $payloadPlan = ($payloadPlanJson | Out-String) | ConvertFrom-Json
    if ($null -eq $payloadPlan -or -not $payloadPlan.ok) {
        throw "Windows runtime bootstrap validation did not return a successful plan."
    }
    $lockedInputs = @(
        New-LockedInputRecord -Path $packageLockPath
        New-LockedInputRecord -Path $cargoLockPath
    )
    # Tauri 2.11 mis-normalizes absolute Windows resource-map keys by dropping
    # their drive prefix. The CLI changes its working directory to src-tauri,
    # so stage below that directory and give Tauri only a relative source path.
    $temporaryBuildName = ".mts-tauri-build-" + [guid]::NewGuid().ToString("N")
    $temporaryBuildRoot = Join-Path $tauriRoot $temporaryBuildName
    $embeddedRuntime = Join-Path $temporaryBuildRoot "mts-runtime"
    $runtimeResourceSource = Get-RelativePathFromRoot -Root $tauriRoot -Path $embeddedRuntime
    $windowsOverlay = New-TauriWindowsOverlayConfig `
        -RuntimeSource $runtimeResourceSource `
        -PublisherThumbprint $PublisherThumbprint `
        -TimestampUrl $TimestampUrl

    if ($DryRun) {
        Write-ResultAndExit ([ordered]@{
            ok = $true
            action = "BuildTauriRelease"
            dryRun = $true
            projectRoot = $project
            version = $version
            architecture = $Architecture
            targetTriple = $targetTriple
            targetDirectory = $targetBase
            outputDirectory = $output
            portableArchivePath = $portableArchive
            lockedInputs = $lockedInputs
            commands = if ($SkipCompile) { @() } else { $commands }
            runtimeBootstrap = $payloadPlan
            tauriOverlay = $windowsOverlay
            runtimeInjection = [ordered]@{
                destination = "resources/mts-runtime"
                strategy = "tauri-bundle-resources-overlay"
                bundledPythonRuntime = $false
                bundledModelArtifacts = $false
            }
            postBuild = @(
                "verify-version-coherence",
                "verify-locked-input-hashes-unchanged",
                "collect-tauri-native-executable",
                "embed-runtime-bootstrap-in-nsis-and-msi",
                "collect-nsis-and-msi-installers",
                "verify-authenticode-or-explicit-development-mode",
                "emit-byte-hashed-release-manifest",
                "emit-deterministic-portable-zip"
            )
        }) 0
    }

    $temporaryPayload = Join-Path ([System.IO.Path]::GetTempPath()) ("mts-tauri-payload-" + [guid]::NewGuid().ToString("N"))
    $temporaryInstallers = Join-Path ([System.IO.Path]::GetTempPath()) ("mts-tauri-installers-" + [guid]::NewGuid().ToString("N"))
    $overlayPath = Join-Path $temporaryBuildRoot "tauri.windows.overlay.json"
    $oldCargoTarget = $env:CARGO_TARGET_DIR
    $oldSourceDateEpoch = $env:SOURCE_DATE_EPOCH
    $releaseResult = $null
    try {
        if (-not $SkipCompile) {
            New-Item -ItemType Directory -Path $temporaryBuildRoot -Force | Out-Null
            $runtimeOutput = & $payloadBuilderPath `
                -ProjectRoot $project `
                -OutputDirectory $embeddedRuntime `
                -RuntimeOnly
            $runtimeResult = ($runtimeOutput | Out-String) | ConvertFrom-Json
            if ($null -eq $runtimeResult -or -not $runtimeResult.ok) {
                throw "Windows runtime bootstrap staging failed before the native Tauri build."
            }
            Write-TauriWindowsOverlay `
                -Path $overlayPath `
                -Config $windowsOverlay

            $env:CARGO_TARGET_DIR = $targetBase
            $env:SOURCE_DATE_EPOCH = [string]$SourceDateEpoch
            Push-Location $desktopRoot
            try {
                & npm ci
                if ($LASTEXITCODE -ne 0) {
                    throw "npm ci failed with exit code $LASTEXITCODE."
                }
                $tauriBuildArguments = @(
                    "run", "tauri", "--", "build", "--ci", "--target", $targetTriple,
                    "--bundles", "nsis,msi", "--config", $overlayPath, "--", "--locked"
                )
                & npm @tauriBuildArguments
                if ($LASTEXITCODE -ne 0) {
                    throw "Tauri Windows build failed with exit code $LASTEXITCODE."
                }
            }
            finally {
                Pop-Location
            }
        }

        Assert-LockedInputsUnchanged -Inputs $lockedInputs

        $payloadSource = Join-Path $targetRoot $entrypoint
        if (-not (Test-Path -LiteralPath $payloadSource -PathType Leaf) -and $Architecture -eq "x64") {
            $payloadSource = Join-Path $fallbackTargetRoot $entrypoint
        }
        if (-not (Test-Path -LiteralPath $payloadSource -PathType Leaf)) {
            throw "Tauri native executable was not produced at the expected path: $payloadSource"
        }

        $payloadResultJson = & $payloadBuilderPath `
            -ProjectRoot $project `
            -ApplicationExecutable $payloadSource `
            -OutputDirectory $temporaryPayload `
            -EntryPoint $entrypoint
        $payloadResult = ($payloadResultJson | Out-String) | ConvertFrom-Json
        if ($null -eq $payloadResult -or -not $payloadResult.ok) {
            throw "Windows runtime bootstrap staging did not return a successful result."
        }

        $artifactRoot = Join-Path (Split-Path -Parent $payloadSource) "bundle"
        if (-not (Test-Path -LiteralPath $artifactRoot -PathType Container)) {
            throw "Tauri native bundle directory is missing: $artifactRoot"
        }
        $installerCandidates = @(
            Get-ChildItem -LiteralPath $artifactRoot -File -Recurse -Force |
                Where-Object { $_.Extension.ToLowerInvariant() -in @(".msi", ".exe") }
        )
        $msiCandidates = @($installerCandidates | Where-Object { $_.Extension.ToLowerInvariant() -eq ".msi" })
        $nsisCandidates = @($installerCandidates | Where-Object { $_.Extension.ToLowerInvariant() -eq ".exe" })
        if ($msiCandidates.Count -eq 0 -or $nsisCandidates.Count -eq 0) {
            throw "Tauri build did not produce both MSI and NSIS installers (MSI=$($msiCandidates.Count), NSIS=$($nsisCandidates.Count))."
        }
        New-Item -ItemType Directory -Path $temporaryInstallers -Force | Out-Null
        foreach ($installer in $installerCandidates | Sort-Object FullName) {
            $relative = Get-RelativePathFromRoot -Root $artifactRoot -Path $installer.FullName
            $destination = Join-Path $temporaryInstallers ($relative.Replace("/", "\"))
            [System.IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
            Copy-Item -LiteralPath $installer.FullName -Destination $destination
        }

        $gitCommit = "unknown"
        try { $gitCommit = (& git -C $project rev-parse HEAD).Trim() } catch { }

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
            NativeInstallerDirectory = $temporaryInstallers
            Force = [bool]$Force
        }
        if (-not [string]::IsNullOrWhiteSpace($MinInstalledVersion)) { $releaseArguments.MinInstalledVersion = $MinInstalledVersion }
        if (-not [string]::IsNullOrWhiteSpace($MaxInstalledVersion)) { $releaseArguments.MaxInstalledVersion = $MaxInstalledVersion }
        if (-not [string]::IsNullOrWhiteSpace($PublisherThumbprint)) { $releaseArguments.PublisherThumbprint = $PublisherThumbprint }

        $releaseOutput = & $releaseScript @releaseArguments | Out-String
        if ($LASTEXITCODE -ne 0) {
            throw "Release manifest generation failed with exit code $LASTEXITCODE."
        }
        $releaseResult = $releaseOutput.Trim() | ConvertFrom-Json
        $portableResult = New-DeterministicPortableArchive `
            -ReleaseRoot $output `
            -ArchivePath $portableArchive `
            -SourceDateEpoch $SourceDateEpoch `
            -Force:$Force

        Write-ResultAndExit ([ordered]@{
            ok = $true
            action = "BuildTauriRelease"
            dryRun = $false
            release = $releaseResult
            portableArchive = $portableResult
            runtimeInjection = [ordered]@{
                destination = "resources/mts-runtime"
                strategy = "tauri-bundle-resources-overlay"
                bundledPythonRuntime = $false
                bundledModelArtifacts = $false
            }
        }) 0
    }
    finally {
        if ($null -eq $oldCargoTarget) { Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue } else { $env:CARGO_TARGET_DIR = $oldCargoTarget }
        if ($null -eq $oldSourceDateEpoch) { Remove-Item Env:SOURCE_DATE_EPOCH -ErrorAction SilentlyContinue } else { $env:SOURCE_DATE_EPOCH = $oldSourceDateEpoch }
        foreach ($temporary in @($temporaryPayload, $temporaryInstallers, $temporaryBuildRoot)) {
            if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Recurse -Force }
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
