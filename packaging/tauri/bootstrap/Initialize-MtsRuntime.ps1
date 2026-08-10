[CmdletBinding()]
param(
    [string]$AppRoot,
    [string]$DataRoot,
    [string]$ModelRoot,
    [string]$WorkerPython,
    [string]$ProductionConfig,
    [string]$ConfigurationTemplate,
    [switch]$PersistUserEnvironment,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Resolve-OptionalFile {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }
    $candidate = [System.IO.Path]::GetFullPath($Path)
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Required file does not exist: $candidate"
    }
    return $candidate
}

function Convert-ToPortablePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return $Path.Replace("\", "/")
}

function New-BoundProductionConfig {
    param(
        [Parameter(Mandatory = $true)][string]$TemplatePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath,
        [Parameter(Mandatory = $true)][string]$DataDirectory,
        [Parameter(Mandatory = $true)][string]$ModelsDirectory,
        [Parameter(Mandatory = $true)][string]$ApplicationRoot
    )

    $document = Get-Content -LiteralPath $TemplatePath -Raw | ConvertFrom-Json
    $inputRoot = Join-Path $DataDirectory "inputs"
    $outputRoot = Join-Path $DataDirectory "exports"
    $cacheRoot = Join-Path $DataDirectory "cache"
    $document.paths.allowedInputRoots = @((Convert-ToPortablePath $inputRoot))
    $document.paths.allowedOutputRoot = Convert-ToPortablePath $outputRoot
    $document.paths.cacheRoot = Convert-ToPortablePath $cacheRoot

    $modelRootPortable = Convert-ToPortablePath $ModelsDirectory
    foreach ($property in @("funasrVad", "qwen3Asr", "qwen3ForcedAligner", "camPlus", "pyannote")) {
        $value = $document.models.$property
        if ($value -is [string] -and $value.StartsWith("D:/models/", [System.StringComparison]::OrdinalIgnoreCase)) {
            $suffix = $value.Substring("D:/models/".Length).Replace("/", "\")
            $document.models.$property = Convert-ToPortablePath (Join-Path $ModelsDirectory $suffix)
        }
        elseif ($value -is [string] -and $value.StartsWith("models/", [System.StringComparison]::OrdinalIgnoreCase)) {
            $suffix = $value.Substring("models/".Length).Replace("/", "\")
            $document.models.$property = Convert-ToPortablePath (Join-Path $ModelsDirectory $suffix)
        }
    }
    $secondary = $document.models.secondarySpeakerVerifier
    if ($null -ne $secondary -and $secondary.path -is [string] -and
        $secondary.path.StartsWith("D:/models/", [System.StringComparison]::OrdinalIgnoreCase)) {
        $suffix = $secondary.path.Substring("D:/models/".Length).Replace("/", "\")
        $secondary.path = Convert-ToPortablePath (Join-Path $ModelsDirectory $suffix)
    }
    elseif ($null -ne $secondary -and $secondary.path -is [string] -and
        $secondary.path.StartsWith("models/", [System.StringComparison]::OrdinalIgnoreCase)) {
        $suffix = $secondary.path.Substring("models/".Length).Replace("/", "\")
        $secondary.path = Convert-ToPortablePath (Join-Path $ModelsDirectory $suffix)
    }

    if ($null -ne $document.executables -and $document.executables.pdfRendererJar -is [string] -and
        $document.executables.pdfRendererJar.StartsWith("pdf-renderer/", [System.StringComparison]::OrdinalIgnoreCase)) {
        $document.executables.pdfRendererJar = Convert-ToPortablePath (
            Join-Path $ApplicationRoot ($document.executables.pdfRendererJar.Replace("/", "\"))
        )
    }

    $bundledPyannote = Join-Path $ApplicationRoot "runtime\pyannote\python.exe"
    if (Test-Path -LiteralPath $bundledPyannote -PathType Leaf) {
        $document.executables.pyannotePython = [System.IO.Path]::GetFullPath($bundledPyannote)
    }
    $json = ($document | ConvertTo-Json -Depth 64) + "`n"
    [System.IO.File]::WriteAllText(
        $DestinationPath,
        $json,
        (New-Object System.Text.UTF8Encoding($false))
    )
}

try {
    if ([string]::IsNullOrWhiteSpace($AppRoot)) {
        $AppRoot = Join-Path $PSScriptRoot ".."
    }
    $app = [System.IO.Path]::GetFullPath($AppRoot)
    if (-not (Test-Path -LiteralPath $app -PathType Container)) {
        throw "AppRoot does not exist: $app"
    }

    if ([string]::IsNullOrWhiteSpace($DataRoot)) {
        $localData = if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
            [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
        }
        else {
            $env:LOCALAPPDATA
        }
        $DataRoot = Join-Path $localData "MediaTranscribeStudio"
    }
    $data = [System.IO.Path]::GetFullPath($DataRoot)

    if ([string]::IsNullOrWhiteSpace($ModelRoot)) {
        if (-not [string]::IsNullOrWhiteSpace($env:MTS_MODEL_ROOT)) {
            $ModelRoot = $env:MTS_MODEL_ROOT
        }
        elseif (Test-Path -LiteralPath "D:\" -PathType Container) {
            $ModelRoot = "D:\models"
        }
        else {
            $ModelRoot = Join-Path $data "models"
        }
    }
    $models = [System.IO.Path]::GetFullPath($ModelRoot)

    if ([string]::IsNullOrWhiteSpace($ProductionConfig)) {
        $ProductionConfig = Join-Path $data "config\production.config.json"
    }
    $config = [System.IO.Path]::GetFullPath($ProductionConfig)
    if ([string]::IsNullOrWhiteSpace($ConfigurationTemplate)) {
        $ConfigurationTemplate = "production.config.example.json"
    }
    $configTemplate = if ([System.IO.Path]::IsPathRooted($ConfigurationTemplate)) {
        [System.IO.Path]::GetFullPath($ConfigurationTemplate)
    }
    else {
        [System.IO.Path]::GetFullPath((Join-Path $app $ConfigurationTemplate))
    }
    if (-not (Test-Path -LiteralPath $configTemplate -PathType Leaf)) {
        throw "Production configuration template is missing: $configTemplate"
    }

    $python = Resolve-OptionalFile $WorkerPython
    if ($null -eq $python -and -not [string]::IsNullOrWhiteSpace($env:MTS_WORKER_PYTHON)) {
        $python = Resolve-OptionalFile $env:MTS_WORKER_PYTHON
    }
    if ($null -eq $python) {
        $bundledPython = Join-Path $app "runtime\media-asr\python.exe"
        if (Test-Path -LiteralPath $bundledPython -PathType Leaf) {
            $python = [System.IO.Path]::GetFullPath($bundledPython)
        }
    }

    $operations = @("create-data-root", "create-model-root", "create-input-output-cache-roots")
    if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
        $operations += "seed-production-config-without-overwrite"
    }
    if ($PersistUserEnvironment) {
        $operations += "persist-non-secret-user-environment"
    }

    $result = [ordered]@{
        ok = $true
        action = "InitializeMtsRuntime"
        dryRun = [bool]$DryRun
        appRoot = $app
        dataRoot = $data
        modelRoot = $models
        productionConfig = $config
        workerPythonHint = Join-Path (Split-Path -Parent $config) "worker-python.path"
        configurationTemplate = $configTemplate
        workerPython = $python
        bundledModelArtifacts = $false
        operations = $operations
    }

    if (-not $DryRun) {
        [System.IO.Directory]::CreateDirectory($data) | Out-Null
        [System.IO.Directory]::CreateDirectory($models) | Out-Null
        if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
            [System.IO.Directory]::CreateDirectory((Split-Path -Parent $config)) | Out-Null
            New-BoundProductionConfig `
                -TemplatePath $configTemplate `
                -DestinationPath $config `
                -DataDirectory $data `
                -ModelsDirectory $models `
                -ApplicationRoot $app
            [System.IO.Directory]::CreateDirectory((Join-Path $data "inputs")) | Out-Null
            [System.IO.Directory]::CreateDirectory((Join-Path $data "exports")) | Out-Null
            [System.IO.Directory]::CreateDirectory((Join-Path $data "cache")) | Out-Null
        }
        if ($null -ne $python) {
            # The desktop app may be launched from Explorer with no shell PATH;
            # keep the operator-selected interpreter in a separate, replaceable
            # hint file and never alter an existing production config.
            $hintPath = Join-Path (Split-Path -Parent $config) "worker-python.path"
            [System.IO.Directory]::CreateDirectory((Split-Path -Parent $hintPath)) | Out-Null
            [System.IO.File]::WriteAllText(
                $hintPath,
                $python + [Environment]::NewLine,
                (New-Object System.Text.UTF8Encoding($false))
            )
        }
        if ($PersistUserEnvironment) {
            [Environment]::SetEnvironmentVariable("MTS_RUNTIME_ROOT", $app, "User")
            [Environment]::SetEnvironmentVariable("MTS_PRODUCTION_CONFIG", $config, "User")
            [Environment]::SetEnvironmentVariable("MTS_MODEL_ROOT", $models, "User")
            if ($null -ne $python) {
                [Environment]::SetEnvironmentVariable("MTS_WORKER_PYTHON", $python, "User")
            }
        }
    }

    [Console]::Out.WriteLine(($result | ConvertTo-Json -Depth 16))
}
catch {
    [Console]::Error.WriteLine((([ordered]@{
        ok = $false
        action = "InitializeMtsRuntime"
        error = $_.Exception.Message
    }) | ConvertTo-Json -Depth 8 -Compress))
    exit 1
}
