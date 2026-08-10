[CmdletBinding()]
param(
    [string]$AppRoot,
    [string]$WorkerPython,
    [string]$ModelRoot,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ModelManagerArguments
)

$ErrorActionPreference = "Stop"

try {
    if ([string]::IsNullOrWhiteSpace($AppRoot)) {
        $AppRoot = Join-Path $PSScriptRoot ".."
    }
    $app = [System.IO.Path]::GetFullPath($AppRoot)
    $manager = Join-Path $app "tools\model_manager.py"
    if (-not (Test-Path -LiteralPath $manager -PathType Leaf)) {
        throw "Model manager is missing: $manager"
    }

    $python = $null
    foreach ($candidate in @(
        $WorkerPython,
        $env:MTS_WORKER_PYTHON,
        (Join-Path $app "runtime\media-asr\python.exe")
    )) {
        if (
            $null -eq $python -and
            -not [string]::IsNullOrWhiteSpace($candidate) -and
            (Test-Path -LiteralPath $candidate -PathType Leaf)
        ) {
            $python = [System.IO.Path]::GetFullPath($candidate)
        }
    }
    if ($null -eq $python) {
        foreach ($name in @("python.exe", "python3.exe", "python")) {
            $command = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($null -ne $command) {
                $python = $command.Source
                break
            }
        }
    }
    if ($null -eq $python) {
        throw "No Python runtime is available. Set MTS_WORKER_PYTHON or pass -WorkerPython."
    }

    if ($null -eq $ModelManagerArguments -or $ModelManagerArguments.Count -eq 0) {
        $ModelManagerArguments = @("list")
    }
    if (-not [string]::IsNullOrWhiteSpace($ModelRoot)) {
        $env:MTS_MODEL_ROOT = [System.IO.Path]::GetFullPath($ModelRoot)
    }

    $commandArguments = @($manager)
    if (
        $ModelManagerArguments[0] -eq "list" -and
        $ModelManagerArguments -notcontains "--registry" -and
        $ModelManagerArguments -notcontains "-registry"
    ) {
        $catalog = Join-Path $app "configs\model-catalog.v1.json"
        if (Test-Path -LiteralPath $catalog -PathType Leaf) {
            $commandArguments += @("--registry", $catalog)
        }
    }
    $commandArguments += @($ModelManagerArguments)
    if ($DryRun) {
        [Console]::Out.WriteLine((([ordered]@{
            ok = $true
            action = "ManageMtsModels"
            dryRun = $true
            executable = $python
            arguments = $commandArguments
            modelRoot = $env:MTS_MODEL_ROOT
        }) | ConvertTo-Json -Depth 16))
        return
    }

    & $python @commandArguments
    exit $LASTEXITCODE
}
catch {
    [Console]::Error.WriteLine((([ordered]@{
        ok = $false
        action = "ManageMtsModels"
        error = $_.Exception.Message
    }) | ConvertTo-Json -Depth 8 -Compress))
    exit 1
}
