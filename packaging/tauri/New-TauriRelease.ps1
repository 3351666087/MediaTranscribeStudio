[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PayloadDirectory,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [Parameter(Mandatory = $true)]
    [string]$Version,

    [Parameter(Mandatory = $true)]
    [string]$EntryPoint,

    [string]$AppId = "studio.mediatranscribe.desktop",
    [string]$ProductName = "MediaTranscribe Studio",
    [ValidateSet("x64", "arm64")]
    [string]$Architecture = "x64",
    [string]$Channel = "stable",
    [int64]$SourceDateEpoch = 0,
    [string]$GitCommit = "unknown",
    [string]$TargetTriple = "x86_64-pc-windows-msvc",
    [string]$MinInstalledVersion,
    [string]$MaxInstalledVersion,
    [int]$DataSchemaReadableMin = 1,
    [int]$DataSchemaReadableMax = 1,
    [int]$DataSchemaWriteVersion = 1,
    [string]$NativeInstallerDirectory,
    [string]$PublisherThumbprint,
    [switch]$AllowUnsignedDevelopment,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$modulePath = Join-Path $PSScriptRoot "Mts.TauriPackaging.psm1"

try {
    Import-Module $modulePath -Force
    $arguments = @{
        PayloadDirectory = $PayloadDirectory
        OutputDirectory = $OutputDirectory
        Version = $Version
        EntryPoint = $EntryPoint
        AppId = $AppId
        ProductName = $ProductName
        Architecture = $Architecture
        Channel = $Channel
        SourceDateEpoch = $SourceDateEpoch
        GitCommit = $GitCommit
        TargetTriple = $TargetTriple
        DataSchemaReadableMin = $DataSchemaReadableMin
        DataSchemaReadableMax = $DataSchemaReadableMax
        DataSchemaWriteVersion = $DataSchemaWriteVersion
        AllowUnsignedDevelopment = [bool]$AllowUnsignedDevelopment
        Force = [bool]$Force
        DryRun = [bool]$DryRun
    }
    if (-not [string]::IsNullOrWhiteSpace($MinInstalledVersion)) {
        $arguments.MinInstalledVersion = $MinInstalledVersion
    }
    if (-not [string]::IsNullOrWhiteSpace($MaxInstalledVersion)) {
        $arguments.MaxInstalledVersion = $MaxInstalledVersion
    }
    if (-not [string]::IsNullOrWhiteSpace($NativeInstallerDirectory)) {
        $arguments.NativeInstallerDirectory = $NativeInstallerDirectory
    }
    if (-not [string]::IsNullOrWhiteSpace($PublisherThumbprint)) {
        $arguments.PublisherThumbprint = $PublisherThumbprint
    }

    $result = New-MtsReleaseBundle @arguments
    $result | ConvertTo-Json -Depth 32
    exit 0
}
catch {
    $errorResult = [ordered]@{
        ok = $false
        action = "CreateRelease"
        error = $_.Exception.Message
    }
    [Console]::Error.WriteLine(($errorResult | ConvertTo-Json -Depth 16 -Compress))
    exit 1
}
