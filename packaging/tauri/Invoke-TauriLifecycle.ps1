[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Validate", "Install", "Upgrade", "Rollback", "Recover", "Uninstall", "Status")]
    [string]$Action,

    [string]$ReleaseDirectory,
    [string]$InstallRoot,
    [string]$DataRoot,
    [string]$ExpectedAppId = "studio.mediatranscribe.desktop",
    [string]$ExpectedPublisherThumbprint,
    [switch]$AllowUnsignedDevelopment,
    [switch]$AllowDowngrade,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$modulePath = Join-Path $PSScriptRoot "Mts.TauriPackaging.psm1"

try {
    Import-Module $modulePath -Force
    $arguments = @{
        Action = $Action
        ExpectedAppId = $ExpectedAppId
        AllowUnsignedDevelopment = [bool]$AllowUnsignedDevelopment
        AllowDowngrade = [bool]$AllowDowngrade
        DryRun = [bool]$DryRun
    }
    if (-not [string]::IsNullOrWhiteSpace($ReleaseDirectory)) {
        $arguments.ReleaseDirectory = $ReleaseDirectory
    }
    if (-not [string]::IsNullOrWhiteSpace($InstallRoot)) {
        $arguments.InstallRoot = $InstallRoot
    }
    if (-not [string]::IsNullOrWhiteSpace($DataRoot)) {
        $arguments.DataRoot = $DataRoot
    }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedPublisherThumbprint)) {
        $arguments.ExpectedPublisherThumbprint = $ExpectedPublisherThumbprint
    }

    $result = Invoke-MtsLifecycle @arguments
    $result | ConvertTo-Json -Depth 32
    exit 0
}
catch {
    $errorResult = [ordered]@{
        ok = $false
        action = $Action
        error = $_.Exception.Message
    }
    [Console]::Error.WriteLine(($errorResult | ConvertTo-Json -Depth 16 -Compress))
    exit 1
}
