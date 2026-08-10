[CmdletBinding()]
param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,

    [string]$ApplicationExecutable,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [string]$EntryPoint = "media-transcribe-studio.exe",

    [switch]$Force,

    # Used by the native Tauri build overlay.  The executable is supplied by
    # Tauri itself; this mode stages only the weight-free worker/runtime files.
    [switch]$RuntimeOnly,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Convert-ToForwardSlashPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return $Path.Replace("\", "/")
}

function Assert-SafeRelativePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $normalized = Convert-ToForwardSlashPath $Path
    if (
        [string]::IsNullOrWhiteSpace($normalized) -or
        [System.IO.Path]::IsPathRooted($normalized) -or
        $normalized.StartsWith("/") -or
        $normalized.Contains(":")
    ) {
        throw "Release payload path must be relative: $Path"
    }
    foreach ($segment in $normalized.Split("/")) {
        if ([string]::IsNullOrWhiteSpace($segment) -or $segment -in @(".", "..")) {
            throw "Release payload path contains an unsafe segment: $Path"
        }
    }
    return $normalized
}

function Get-RelativePathFromRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $rootPath = [System.IO.Path]::GetFullPath($Root).TrimEnd("\", "/") + "\"
    $pathValue = [System.IO.Path]::GetFullPath($Path)
    if (-not $pathValue.StartsWith($rootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Source file escapes its declared root: $Path"
    }
    return Convert-ToForwardSlashPath $pathValue.Substring($rootPath.Length)
}

function Write-JsonResult {
    param([Parameter(Mandatory = $true)]$Value)
    Write-Output ($Value | ConvertTo-Json -Depth 32)
}

function New-PortableConfigText {
    param([Parameter(Mandatory = $true)][string]$TemplatePath)

    try {
        $document = Get-Content -LiteralPath $TemplatePath -Raw | ConvertFrom-Json
    }
    catch {
        throw "Unable to parse Windows runtime configuration template: $TemplatePath"
    }

    if ($null -eq $document.paths -or $null -eq $document.models) {
        throw "Windows runtime configuration template must contain paths and models objects: $TemplatePath"
    }

    # These values are deliberately relative.  Initialize-MtsRuntime.ps1 binds
    # them to the operator's DataRoot/ModelRoot without mutating the shipped
    # template or retaining the build workstation's drive letters.
    $document.paths.allowedInputRoots = @("inputs")
    $document.paths.allowedOutputRoot = "exports"
    $document.paths.cacheRoot = "cache"
    foreach ($property in @("funasrVad", "qwen3Asr", "qwen3ForcedAligner", "camPlus", "pyannote")) {
        $value = $document.models.$property
        if ($value -is [string] -and $value -match "^(?i:D:/models/)") {
            $document.models.$property = "models/" + $value.Substring(10)
        }
    }
    $secondary = $document.models.secondarySpeakerVerifier
    if ($null -ne $secondary -and $secondary.path -is [string] -and
        $secondary.path -match "^(?i:D:/models/)") {
        $secondary.path = "models/" + $secondary.path.Substring(10)
    }
    if ($null -ne $document.executables -and $document.executables.pyannotePython -is [string] -and
        $document.executables.pyannotePython -match "^(?i:D:/MediaTranscribeStudio/runtime/)") {
        $document.executables.pyannotePython = "runtime/" + $document.executables.pyannotePython.Substring(33)
    }

    $json = ($document | ConvertTo-Json -Depth 64) + "`n"
    if ($json -match "(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]") {
        throw "Windows runtime configuration template still contains a drive-qualified path: $TemplatePath"
    }
    if ($json -match "(?i)\bsk-[A-Za-z0-9]{16,}\b") {
        throw "Possible API credential found in Windows runtime configuration template: $TemplatePath"
    }
    return $json
}

try {
    $project = [System.IO.Path]::GetFullPath($ProjectRoot)
    $output = [System.IO.Path]::GetFullPath($OutputDirectory)
    $entrypointPath = Assert-SafeRelativePath $EntryPoint
    if (-not (Test-Path -LiteralPath $project -PathType Container)) {
        throw "ProjectRoot does not exist: $project"
    }

    if (-not $RuntimeOnly -and [string]::IsNullOrWhiteSpace($ApplicationExecutable)) {
        throw "ApplicationExecutable is required unless -RuntimeOnly is supplied."
    }

    $entries = New-Object 'System.Collections.Generic.List[object]'
    $destinationKeys = @{}

    function Add-PayloadFile {
        param(
            [Parameter(Mandatory = $true)][string]$Source,
            [Parameter(Mandatory = $true)][string]$Destination,
            [string]$Role = "runtime-support",
            [switch]$AllowMissingInDryRun
        )

        $sourcePath = [System.IO.Path]::GetFullPath($Source)
        $destinationPath = Assert-SafeRelativePath $Destination
        $key = $destinationPath.ToLowerInvariant()
        if ($destinationKeys.ContainsKey($key)) {
            throw "Release payload has a duplicate destination: $destinationPath"
        }
        $exists = Test-Path -LiteralPath $sourcePath -PathType Leaf
        if (-not $exists -and -not ($DryRun -and $AllowMissingInDryRun)) {
            throw "Required Windows release input is missing: $sourcePath"
        }
        $destinationKeys[$key] = $true
        $entries.Add([pscustomobject][ordered]@{
            source = $sourcePath
            destination = $destinationPath
            role = $Role
            size = if ($exists) { [int64](Get-Item -LiteralPath $sourcePath).Length } else { 0 }
            sourcePresent = [bool]$exists
        }) | Out-Null
    }

    function Add-PayloadGeneratedFile {
        param(
            [Parameter(Mandatory = $true)][string]$Destination,
            [Parameter(Mandatory = $true)][string]$Content,
            [string]$Role = "runtime-support"
        )

        $destinationPath = Assert-SafeRelativePath $Destination
        $key = $destinationPath.ToLowerInvariant()
        if ($destinationKeys.ContainsKey($key)) {
            throw "Release payload has a duplicate destination: $destinationPath"
        }
        $destinationKeys[$key] = $true
        $entries.Add([pscustomobject][ordered]@{
            source = $null
            content = $Content
            generated = $true
            destination = $destinationPath
            role = $Role
            size = [int64][Text.Encoding]::UTF8.GetByteCount($Content)
            sourcePresent = $true
        }) | Out-Null
    }

    function Add-PayloadTree {
        param(
            [Parameter(Mandatory = $true)][string]$SourceDirectory,
            [Parameter(Mandatory = $true)][string]$DestinationDirectory,
            [Parameter(Mandatory = $true)][string[]]$Extensions,
            [string]$Role = "runtime-support"
        )

        $sourceRoot = [System.IO.Path]::GetFullPath($SourceDirectory)
        if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
            throw "Required Windows release directory is missing: $sourceRoot"
        }
        $files = @(
            Get-ChildItem -LiteralPath $sourceRoot -File -Recurse -Force |
                Where-Object {
                    $_.FullName -notmatch "(?i)[\\/]__pycache__[\\/]" -and
                    $Extensions -contains $_.Extension.ToLowerInvariant()
                } |
                Sort-Object FullName
        )
        if ($files.Count -eq 0) {
            throw "Windows release directory has no accepted files: $sourceRoot"
        }
        foreach ($file in $files) {
            if (($file.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Reparse points are forbidden in Windows release inputs: $($file.FullName)"
            }
            $relative = Get-RelativePathFromRoot -Root $sourceRoot -Path $file.FullName
            Add-PayloadFile `
                -Source $file.FullName `
                -Destination ((Convert-ToForwardSlashPath $DestinationDirectory).TrimEnd("/") + "/" + $relative) `
                -Role $Role
        }
    }

    if (-not $RuntimeOnly) {
        Add-PayloadFile `
            -Source $ApplicationExecutable `
            -Destination $entrypointPath `
            -Role "desktop-entrypoint" `
            -AllowMissingInDryRun
    }

    Add-PayloadTree `
        -SourceDirectory (Join-Path $project "backend") `
        -DestinationDirectory "backend" `
        -Extensions @(".py") `
        -Role "worker-code"
    Add-PayloadTree `
        -SourceDirectory (Join-Path $project "contracts") `
        -DestinationDirectory "contracts" `
        -Extensions @(".json", ".py") `
        -Role "worker-contract"
    Add-PayloadTree `
        -SourceDirectory (Join-Path $project "reporting") `
        -DestinationDirectory "reporting" `
        -Extensions @(".py") `
        -Role "report-runtime"

    $fixedFiles = @(
        @("configs\model-catalog.v1.json", "configs/model-catalog.v1.json", "portable-model-catalog"),
        @("configs\model-catalog.v1.schema.json", "configs/model-catalog.v1.schema.json", "model-catalog-contract"),
        @("tools\build_portable_model_catalog.py", "tools/build_portable_model_catalog.py", "portable-catalog-builder"),
        @("configs\llm-provider-presets.v1.json", "configs/llm-provider-presets.v1.json", "provider-presets"),
        @("configs\llm-provider-presets.schema.json", "configs/llm-provider-presets.schema.json", "provider-contract"),
        @("tools\model_manager.py", "tools/model_manager.py", "model-manager"),
        @("tools\model_registry.py", "tools/model_registry.py", "model-registry-validator"),
        @("tools\pyannote_runtime.py", "tools/pyannote_runtime.py", "pyannote-runtime"),
        @("requirements-media-asr.txt", "requirements-media-asr.txt", "runtime-requirements"),
        @("requirements-pyannote.txt", "requirements-pyannote.txt", "runtime-requirements"),
        @("pdf-renderer\target\pdf-renderer.jar", "pdf-renderer/target/pdf-renderer.jar", "pdf-runtime"),
        @("packaging\tauri\bootstrap\Initialize-MtsRuntime.ps1", "bootstrap/Initialize-MtsRuntime.ps1", "bootstrap-entrypoint"),
        @("packaging\tauri\bootstrap\Manage-MtsModels.ps1", "bootstrap/Manage-MtsModels.ps1", "model-manager-entrypoint"),
        @("packaging\tauri\bootstrap\README.md", "bootstrap/README.md", "bootstrap-guide")
    )

    foreach ($configName in @("production.config.example.json", "production.config.remote.example.json")) {
        $configPath = Join-Path $project $configName
        if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
            throw "Required Windows runtime input is missing: $configPath"
        }
        $configRole = if ($configName -eq "production.config.example.json") {
            "config-template"
        }
        else {
            "remote-config-template"
        }
        Add-PayloadGeneratedFile `
            -Destination $configName `
            -Content (New-PortableConfigText -TemplatePath $configPath) `
            -Role $configRole
    }

    foreach ($mapping in $fixedFiles) {
        Add-PayloadFile `
            -Source (Join-Path $project $mapping[0]) `
            -Destination $mapping[1] `
            -Role $mapping[2]
    }

    $profile = [ordered]@{
        schemaVersion = "1.0.0"
        artifactType = "mts-windows-runtime-bootstrap"
        runtimeStrategy = "external-or-operator-provided"
        bundledPythonRuntime = $false
        bundledModelArtifacts = $false
        defaultModelRootPolicy = @("MTS_MODEL_ROOT", "D:/models", "LOCALAPPDATA")
        productionConfigTemplate = "production.config.example.json"
        providerPresetCatalog = "configs/llm-provider-presets.v1.json"
        modelRegistry = "configs/model-catalog.v1.json"
        modelManager = "bootstrap/Manage-MtsModels.ps1"
        runtimeInitializer = "bootstrap/Initialize-MtsRuntime.ps1"
    }
    $profileJson = ($profile | ConvertTo-Json -Depth 16) + "`n"

    $plannedFileCount = $entries.Count + 1
    $plannedBytes = [int64](($entries | Measure-Object -Property size -Sum).Sum) + [Text.Encoding]::UTF8.GetByteCount($profileJson)
    $result = [ordered]@{
        ok = $true
        action = "CreateWindowsReleasePayload"
        dryRun = [bool]$DryRun
        outputDirectory = $output
        entrypoint = if ($RuntimeOnly) { $null } else { $entrypointPath }
        payloadKind = if ($RuntimeOnly) { "runtime-only" } else { "portable" }
        fileCount = $plannedFileCount
        totalBytes = $plannedBytes
        bundledPythonRuntime = $false
        bundledModelArtifacts = $false
        defaultModelRootPolicy = @("MTS_MODEL_ROOT", "D:/models", "LOCALAPPDATA")
        files = @($entries | ForEach-Object {
            [ordered]@{
                destination = $_.destination
                role = $_.role
                size = $_.size
                sourcePresent = $_.sourcePresent
            }
        }) + @([ordered]@{
            destination = "bootstrap/runtime-bootstrap.v1.json"
            role = "bootstrap-profile"
            size = [Text.Encoding]::UTF8.GetByteCount($profileJson)
            sourcePresent = $true
        })
    }

    if ($DryRun) {
        Write-JsonResult $result
        return
    }

    if (Test-Path -LiteralPath $output) {
        if (-not $Force) {
            throw "OutputDirectory already exists: $output"
        }
        Remove-Item -LiteralPath $output -Recurse -Force
    }

    $outputParent = Split-Path -Parent $output
    [System.IO.Directory]::CreateDirectory($outputParent) | Out-Null
    $stage = Join-Path $outputParent (".mts-windows-payload-" + [guid]::NewGuid().ToString("N"))
    try {
        [System.IO.Directory]::CreateDirectory($stage) | Out-Null
        foreach ($entry in $entries) {
            $destination = Join-Path $stage ($entry.destination.Replace("/", "\"))
            [System.IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
            if ($entry.generated) {
                [System.IO.File]::WriteAllText(
                    $destination,
                    [string]$entry.content,
                    (New-Object System.Text.UTF8Encoding($false))
                )
            }
            else {
                Copy-Item -LiteralPath $entry.source -Destination $destination
            }
        }
        $profilePath = Join-Path $stage "bootstrap\runtime-bootstrap.v1.json"
        [System.IO.Directory]::CreateDirectory((Split-Path -Parent $profilePath)) | Out-Null
        [System.IO.File]::WriteAllText(
            $profilePath,
            $profileJson,
            (New-Object System.Text.UTF8Encoding($false))
        )
        Move-Item -LiteralPath $stage -Destination $output
    }
    finally {
        if (Test-Path -LiteralPath $stage) {
            Remove-Item -LiteralPath $stage -Recurse -Force
        }
    }

    Write-JsonResult $result
}
catch {
    [Console]::Error.WriteLine((([ordered]@{
        ok = $false
        action = "CreateWindowsReleasePayload"
        error = $_.Exception.Message
    }) | ConvertTo-Json -Depth 8 -Compress))
    exit 1
}
