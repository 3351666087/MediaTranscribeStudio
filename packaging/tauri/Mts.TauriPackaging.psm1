Set-StrictMode -Version Latest

$script:ReleaseContract = "mts-tauri-release/v1"
$script:ReleaseSchemaVersion = "1.1.0"
$script:StateContract = "mts-tauri-install-state/v1"
$script:StateEnvelopeContract = "mts-tauri-record-envelope/v1"
$script:JournalContract = "mts-tauri-transaction/v1"
$script:ReservedMetadataDirectory = ".mts-release"
$script:ControlDirectoryName = ".mts-control"
$script:ReleaseManifestName = "release-manifest.json"
$script:ReleaseManifestChecksumName = "release-manifest.json.sha256"
$script:ReleaseManifestSignatureName = "release-manifest.json.p7s"
$script:Sha256Oid = "2.16.840.1.101.3.4.2.1"
$script:CodeSigningEkuOid = "1.3.6.1.5.5.7.3.3"

function Get-MtsUtf8NoBomEncoding {
    return New-Object System.Text.UTF8Encoding($false)
}

function ConvertTo-MtsJson {
    param(
        [Parameter(Mandatory = $true)]
        $Value,

        [switch]
        $Compressed
    )

    if ($Compressed) {
        return ($Value | ConvertTo-Json -Depth 32 -Compress)
    }

    return ($Value | ConvertTo-Json -Depth 32)
}

function Write-MtsUtf8File {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }

    [System.IO.File]::WriteAllText($Path, $Content, (Get-MtsUtf8NoBomEncoding))
}

function Write-MtsAtomicUtf8File {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $parent = Split-Path -Parent $Path
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    $temporaryPath = Join-Path $parent (".{0}.{1}.tmp" -f ([System.IO.Path]::GetFileName($Path)), [guid]::NewGuid().ToString("N"))
    $replacementBackupPath = Join-Path $parent (".{0}.{1}.replace-backup" -f ([System.IO.Path]::GetFileName($Path)), [guid]::NewGuid().ToString("N"))

    try {
        Write-MtsUtf8File -Path $temporaryPath -Content $Content
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            [System.IO.File]::Replace($temporaryPath, $Path, $replacementBackupPath, $true)
        }
        else {
            [System.IO.File]::Move($temporaryPath, $Path)
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
        if (Test-Path -LiteralPath $replacementBackupPath) {
            Remove-Item -LiteralPath $replacementBackupPath -Force
        }
    }
}

function Get-MtsSha256Bytes {
    param(
        [Parameter(Mandatory = $true)]
        [byte[]]$Bytes
    )

    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        return (($algorithm.ComputeHash($Bytes) | ForEach-Object { $_.ToString("x2") }) -join "")
    }
    finally {
        $algorithm.Dispose()
    }
}

function Get-MtsStringSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return Get-MtsSha256Bytes -Bytes (Get-MtsUtf8NoBomEncoding).GetBytes($Value)
}

function Get-MtsFileSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Cannot hash missing file: $Path"
    }

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function ConvertTo-MtsPublisherThumbprint {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Thumbprint
    )

    $normalized = $Thumbprint.Replace(" ", "").ToUpperInvariant()
    if ($normalized -notmatch "^[A-F0-9]{40}$") {
        throw "Publisher thumbprint must be a 40-character SHA-1 certificate thumbprint."
    }
    return $normalized
}

function Assert-MtsCodeSigningCertificate {
    param(
        [Parameter(Mandatory = $true)]
        [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate
    )

    $ekuOids = @(
        $Certificate.EnhancedKeyUsageList |
            ForEach-Object { [string]$_.ObjectId }
    )
    if ($ekuOids -notcontains $script:CodeSigningEkuOid) {
        throw "Publisher certificate does not declare the Code Signing enhanced key usage."
    }
}

function Import-MtsPkcsAssembly {
    if ($null -ne ("System.Security.Cryptography.Pkcs.SignedCms" -as [type])) {
        return
    }

    try {
        Add-Type -AssemblyName System.Security.Cryptography.Pkcs -ErrorAction Stop
    }
    catch {
        # Windows PowerShell 5.1 exposes SignedCms from the .NET Framework System.Security assembly.
        Add-Type -AssemblyName System.Security -ErrorAction Stop
    }

    if ($null -eq ("System.Security.Cryptography.Pkcs.SignedCms" -as [type])) {
        throw "The CMS/PKCS signing runtime is unavailable."
    }
}

function Get-MtsPublisherCertificate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Thumbprint,

        [switch]$RequirePrivateKey
    )

    $normalized = ConvertTo-MtsPublisherThumbprint $Thumbprint
    $certificatePath = "Cert:\CurrentUser\My\$normalized"
    if (-not (Test-Path -LiteralPath $certificatePath -PathType Leaf)) {
        throw "Publisher certificate is not installed in Cert:\CurrentUser\My: $normalized"
    }
    $certificate = Get-Item -LiteralPath $certificatePath
    if ($null -eq $certificate -or $certificate.Thumbprint.Replace(" ", "").ToUpperInvariant() -ne $normalized) {
        throw "Publisher certificate lookup returned an unexpected certificate."
    }
    Assert-MtsCodeSigningCertificate $certificate
    if ($RequirePrivateKey -and -not $certificate.HasPrivateKey) {
        throw "Publisher certificate does not expose the private key required to sign the release manifest."
    }
    return $certificate
}

function Write-MtsDetachedManifestSignature {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath,

        [Parameter(Mandatory = $true)]
        [string]$SignaturePath,

        [Parameter(Mandatory = $true)]
        [string]$PublisherThumbprint
    )

    Import-MtsPkcsAssembly
    $certificate = Get-MtsPublisherCertificate -Thumbprint $PublisherThumbprint -RequirePrivateKey
    $manifestBytes = [System.IO.File]::ReadAllBytes($ManifestPath)
    $content = [System.Security.Cryptography.Pkcs.ContentInfo]::new($manifestBytes)
    $signedCms = [System.Security.Cryptography.Pkcs.SignedCms]::new($content, $true)
    $signer = [System.Security.Cryptography.Pkcs.CmsSigner]::new($certificate)
    $signer.IncludeOption = [System.Security.Cryptography.X509Certificates.X509IncludeOption]::EndCertOnly
    $signer.DigestAlgorithm = [System.Security.Cryptography.Oid]::new($script:Sha256Oid)
    $signedCms.ComputeSignature($signer, $false)
    [System.IO.File]::WriteAllBytes($SignaturePath, $signedCms.Encode())
}

function Assert-MtsDetachedManifestSignature {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath,

        [Parameter(Mandatory = $true)]
        [string]$SignaturePath,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedPublisherThumbprint
    )

    if (-not (Test-Path -LiteralPath $SignaturePath -PathType Leaf)) {
        throw "Detached publisher signature is missing: $SignaturePath"
    }
    $signatureLength = (Get-Item -LiteralPath $SignaturePath).Length
    if ($signatureLength -le 0 -or $signatureLength -gt 1048576) {
        throw "Detached publisher signature has an invalid size."
    }

    Import-MtsPkcsAssembly
    $manifestBytes = [System.IO.File]::ReadAllBytes($ManifestPath)
    $content = [System.Security.Cryptography.Pkcs.ContentInfo]::new($manifestBytes)
    $signedCms = [System.Security.Cryptography.Pkcs.SignedCms]::new($content, $true)
    try {
        $signedCms.Decode([System.IO.File]::ReadAllBytes($SignaturePath))
        $signedCms.CheckSignature($true)
    }
    catch {
        throw "Detached publisher signature is invalid: $($_.Exception.Message)"
    }

    if ($signedCms.SignerInfos.Count -ne 1) {
        throw "Detached publisher signature must contain exactly one signer."
    }
    $signer = $signedCms.SignerInfos[0]
    if ($signer.DigestAlgorithm.Value -ne $script:Sha256Oid) {
        throw "Detached publisher signature must use SHA-256."
    }
    if ($null -eq $signer.Certificate) {
        throw "Detached publisher signature does not embed its signer certificate."
    }
    Assert-MtsCodeSigningCertificate $signer.Certificate
    $actualThumbprint = ConvertTo-MtsPublisherThumbprint $signer.Certificate.Thumbprint
    $expectedThumbprint = ConvertTo-MtsPublisherThumbprint $ExpectedPublisherThumbprint
    if (-not [string]::Equals($actualThumbprint, $expectedThumbprint, [System.StringComparison]::Ordinal)) {
        throw "Detached publisher signature signer does not match the configured publisher."
    }
}

function Get-MtsCanonicalFullPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Path must not be empty."
    }

    $providerPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
    return [System.IO.Path]::GetFullPath($providerPath).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
}

function Test-MtsPathEqual {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Left,

        [Parameter(Mandatory = $true)]
        [string]$Right
    )

    return [string]::Equals(
        (Get-MtsCanonicalFullPath $Left),
        (Get-MtsCanonicalFullPath $Right),
        [System.StringComparison]::OrdinalIgnoreCase
    )
}

function Test-MtsPathWithin {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Candidate,

        [Parameter(Mandatory = $true)]
        [string]$Root,

        [switch]
        $AllowEqual
    )

    $candidatePath = Get-MtsCanonicalFullPath $Candidate
    $rootPath = Get-MtsCanonicalFullPath $Root

    if ($AllowEqual -and [string]::Equals($candidatePath, $rootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }

    $prefix = $rootPath + [System.IO.Path]::DirectorySeparatorChar
    return $candidatePath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Assert-MtsSafeManagedRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $fullPath = Get-MtsCanonicalFullPath $Path
    $pathRoot = [System.IO.Path]::GetPathRoot($fullPath).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )

    if ([string]::Equals($fullPath, $pathRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must not be a filesystem root: $fullPath"
    }

    $blockedRoots = @(
        $env:WINDIR,
        $env:SystemRoot,
        $env:USERPROFILE,
        $env:ProgramData,
        $env:ProgramFiles,
        ${env:ProgramFiles(x86)}
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    foreach ($blocked in $blockedRoots) {
        if (Test-MtsPathEqual -Left $fullPath -Right $blocked) {
            throw "$Label must not be a protected anchor path: $fullPath"
        }
    }

    return $fullPath
}

function Assert-MtsInstallBoundaries {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallRoot,

        [Parameter(Mandatory = $true)]
        [string]$DataRoot
    )

    $installPath = Assert-MtsSafeManagedRoot -Path $InstallRoot -Label "InstallRoot"
    $dataPath = Assert-MtsSafeManagedRoot -Path $DataRoot -Label "DataRoot"

    if (
        (Test-MtsPathWithin -Candidate $installPath -Root $dataPath -AllowEqual) -or
        (Test-MtsPathWithin -Candidate $dataPath -Root $installPath -AllowEqual)
    ) {
        throw "InstallRoot and DataRoot must not overlap. InstallRoot=$installPath DataRoot=$dataPath"
    }

    $longestControlPath = Join-Path $installPath (
        ".mts-control\transactions\00000000000000000000000000000000\staged-app\.mts-release\release-manifest.json"
    )
    if ($longestControlPath.Length -ge 240) {
        throw "InstallRoot is too long for Windows PowerShell 5.1 transactional paths. Choose a shorter root: $installPath"
    }

    return [pscustomobject][ordered]@{
        InstallRoot = $installPath
        DataRoot = $dataPath
    }
}

function Assert-MtsRelativePayloadPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Payload path must not be empty."
    }

    if ($Path.Contains("\")) {
        throw "Payload paths must use forward slashes: $Path"
    }

    if ([System.IO.Path]::IsPathRooted($Path) -or $Path.StartsWith("/")) {
        throw "Payload path must be relative: $Path"
    }

    if ($Path.IndexOf([char]0) -ge 0 -or $Path.Contains(":")) {
        throw "Payload path contains a forbidden character: $Path"
    }

    $segments = $Path.Split("/")
    if ($segments.Count -eq 0) {
        throw "Payload path must contain a file name."
    }

    $reservedDevicePattern = "^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$"
    foreach ($segment in $segments) {
        if (
            [string]::IsNullOrWhiteSpace($segment) -or
            $segment -eq "." -or
            $segment -eq ".." -or
            $segment.EndsWith(".") -or
            $segment.EndsWith(" ") -or
            $segment -match $reservedDevicePattern
        ) {
            throw "Payload path contains an unsafe segment: $Path"
        }
    }

    if ([string]::Equals($segments[0], $script:ReservedMetadataDirectory, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Payload path uses the reserved metadata directory: $Path"
    }

    return $Path
}

function Get-MtsManifestRelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,

        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $rootUri = New-Object System.Uri((Get-MtsCanonicalFullPath $Root) + [System.IO.Path]::DirectorySeparatorChar)
    $pathUri = New-Object System.Uri((Get-MtsCanonicalFullPath $Path))
    $relative = [System.Uri]::UnescapeDataString($rootUri.MakeRelativeUri($pathUri).ToString())
    return Assert-MtsRelativePayloadPath $relative
}

function Get-MtsObjectProperty {
    param(
        [Parameter(Mandatory = $true)]
        $Object,

        [Parameter(Mandatory = $true)]
        [string]$Name,

        [switch]
        $Required
    )

    if ($null -eq $Object) {
        if ($Required) {
            throw "Required object for property '$Name' is null."
        }
        return $null
    }

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        if ($Required) {
            throw "Required property '$Name' is missing."
        }
        return $null
    }

    return $property.Value
}

function ConvertTo-MtsSemVer {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Version
    )

    $pattern = "^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
    $match = [regex]::Match($Version, $pattern)
    if (-not $match.Success) {
        throw "Version is not valid SemVer 2.0.0: $Version"
    }

    $preRelease = @()
    if ($match.Groups[4].Success) {
        $preRelease = @($match.Groups[4].Value.Split("."))
        foreach ($identifier in $preRelease) {
            if ($identifier -match "^\d+$" -and $identifier.Length -gt 1 -and $identifier.StartsWith("0")) {
                throw "Numeric prerelease identifiers must not contain leading zeroes: $Version"
            }
        }
    }

    return [pscustomobject][ordered]@{
        Raw = $Version
        Major = [uint64]$match.Groups[1].Value
        Minor = [uint64]$match.Groups[2].Value
        Patch = [uint64]$match.Groups[3].Value
        PreRelease = $preRelease
    }
}

function Compare-MtsSemVer {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Left,

        [Parameter(Mandatory = $true)]
        [string]$Right
    )

    $leftVersion = ConvertTo-MtsSemVer $Left
    $rightVersion = ConvertTo-MtsSemVer $Right

    foreach ($field in @("Major", "Minor", "Patch")) {
        if ($leftVersion.$field -lt $rightVersion.$field) {
            return -1
        }
        if ($leftVersion.$field -gt $rightVersion.$field) {
            return 1
        }
    }

    $leftPre = @($leftVersion.PreRelease)
    $rightPre = @($rightVersion.PreRelease)
    if ($leftPre.Count -eq 0 -and $rightPre.Count -eq 0) {
        return 0
    }
    if ($leftPre.Count -eq 0) {
        return 1
    }
    if ($rightPre.Count -eq 0) {
        return -1
    }

    $count = [Math]::Max($leftPre.Count, $rightPre.Count)
    for ($index = 0; $index -lt $count; $index++) {
        if ($index -ge $leftPre.Count) {
            return -1
        }
        if ($index -ge $rightPre.Count) {
            return 1
        }

        $leftIdentifier = $leftPre[$index]
        $rightIdentifier = $rightPre[$index]
        $leftNumeric = $leftIdentifier -match "^\d+$"
        $rightNumeric = $rightIdentifier -match "^\d+$"

        if ($leftNumeric -and $rightNumeric) {
            $leftNumber = [uint64]$leftIdentifier
            $rightNumber = [uint64]$rightIdentifier
            if ($leftNumber -lt $rightNumber) { return -1 }
            if ($leftNumber -gt $rightNumber) { return 1 }
            continue
        }

        if ($leftNumeric -and -not $rightNumeric) {
            return -1
        }
        if (-not $leftNumeric -and $rightNumeric) {
            return 1
        }

        $comparison = [string]::CompareOrdinal($leftIdentifier, $rightIdentifier)
        if ($comparison -lt 0) { return -1 }
        if ($comparison -gt 0) { return 1 }
    }

    return 0
}

function Assert-MtsNoReparsePoints {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root
    )

    if (-not (Test-Path -LiteralPath $Root)) {
        throw "Path does not exist: $Root"
    }

    $rootItem = Get-Item -Force -LiteralPath $Root
    if (($rootItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Reparse points are forbidden in release/install payloads: $($rootItem.FullName)"
    }

    $pendingDirectories = New-Object System.Collections.Generic.Queue[string]
    if ($rootItem.PSIsContainer) {
        $pendingDirectories.Enqueue($rootItem.FullName)
    }

    while ($pendingDirectories.Count -gt 0) {
        $directory = $pendingDirectories.Dequeue()
        foreach ($item in @(Get-ChildItem -Force -LiteralPath $directory)) {
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Reparse points are forbidden in release/install payloads: $($item.FullName)"
            }
            if ($item.PSIsContainer) {
                $pendingDirectories.Enqueue($item.FullName)
            }
        }
    }
}

function Get-MtsPayloadLedger {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PayloadRoot
    )

    if (-not (Test-Path -LiteralPath $PayloadRoot -PathType Container)) {
        throw "Payload directory does not exist: $PayloadRoot"
    }

    Assert-MtsNoReparsePoints $PayloadRoot
    $ledger = @()
    foreach ($file in @(Get-ChildItem -LiteralPath $PayloadRoot -File -Force -Recurse | Sort-Object FullName)) {
        $relativePath = Get-MtsManifestRelativePath -Root $PayloadRoot -Path $file.FullName
        $ledger += [pscustomobject][ordered]@{
            path = $relativePath
            size = [int64]$file.Length
            sha256 = Get-MtsFileSha256 $file.FullName
        }
    }

    if ($ledger.Count -eq 0) {
        throw "Payload directory must contain at least one file."
    }

    return @($ledger | Sort-Object path)
}

function Assert-MtsManifestContract {
    param(
        [Parameter(Mandatory = $true)]
        $Manifest
    )

    if ((Get-MtsObjectProperty $Manifest "schemaVersion" -Required) -ne $script:ReleaseSchemaVersion) {
        throw "Unsupported release schemaVersion."
    }
    if ((Get-MtsObjectProperty $Manifest "contract" -Required) -ne $script:ReleaseContract) {
        throw "Unsupported release contract."
    }

    $appId = [string](Get-MtsObjectProperty $Manifest "appId" -Required)
    if ($appId -notmatch "^[A-Za-z0-9](?:[A-Za-z0-9.-]{1,126}[A-Za-z0-9])?$") {
        throw "Manifest appId is invalid: $appId"
    }

    $version = [string](Get-MtsObjectProperty $Manifest "version" -Required)
    ConvertTo-MtsSemVer $version | Out-Null

    $architecture = [string](Get-MtsObjectProperty $Manifest "architecture" -Required)
    if ($architecture -notin @("x64", "arm64")) {
        throw "Unsupported architecture: $architecture"
    }

    $channel = [string](Get-MtsObjectProperty $Manifest "channel" -Required)
    if ($channel -notmatch "^[a-z][a-z0-9-]{0,31}$") {
        throw "Manifest channel is invalid: $channel"
    }

    $sourceDateEpoch = Get-MtsObjectProperty $Manifest "sourceDateEpoch" -Required
    if ([int64]$sourceDateEpoch -lt 0) {
        throw "sourceDateEpoch must be non-negative."
    }

    $entrypoint = Assert-MtsRelativePayloadPath ([string](Get-MtsObjectProperty $Manifest "entrypoint" -Required))
    $payload = Get-MtsObjectProperty $Manifest "payload" -Required
    if ((Get-MtsObjectProperty $payload "root" -Required) -ne "payload") {
        throw "Manifest payload.root must be 'payload'."
    }

    $files = @(Get-MtsObjectProperty $payload "files" -Required)
    if ($files.Count -eq 0) {
        throw "Manifest payload.files must contain at least one file."
    }

    $seen = @{}
    $totalBytes = [int64]0
    foreach ($file in $files) {
        $path = Assert-MtsRelativePayloadPath ([string](Get-MtsObjectProperty $file "path" -Required))
        $key = $path.ToLowerInvariant()
        if ($seen.ContainsKey($key)) {
            throw "Manifest payload contains a case-insensitive duplicate path: $path"
        }
        $seen[$key] = $true

        $size = [int64](Get-MtsObjectProperty $file "size" -Required)
        if ($size -lt 0) {
            throw "Manifest file size must be non-negative: $path"
        }
        $totalBytes += $size

        $sha256 = [string](Get-MtsObjectProperty $file "sha256" -Required)
        if ($sha256 -notmatch "^[a-f0-9]{64}$") {
            throw "Manifest file hash must be lowercase SHA-256: $path"
        }
    }

    if (-not $seen.ContainsKey($entrypoint.ToLowerInvariant())) {
        throw "Manifest entrypoint is not present in payload.files: $entrypoint"
    }

    if ([int](Get-MtsObjectProperty $payload "fileCount" -Required) -ne $files.Count) {
        throw "Manifest payload.fileCount does not match payload.files."
    }
    if ([int64](Get-MtsObjectProperty $payload "totalBytes" -Required) -ne $totalBytes) {
        throw "Manifest payload.totalBytes does not match payload.files."
    }

    $compatibility = Get-MtsObjectProperty $Manifest "compatibility" -Required
    foreach ($name in @("minInstalledVersion", "maxInstalledVersion")) {
        $value = Get-MtsObjectProperty $compatibility $name
        if ($null -ne $value -and -not [string]::IsNullOrWhiteSpace([string]$value)) {
            ConvertTo-MtsSemVer ([string]$value) | Out-Null
        }
    }

    $minimum = Get-MtsObjectProperty $compatibility "minInstalledVersion"
    $maximum = Get-MtsObjectProperty $compatibility "maxInstalledVersion"
    if (
        $null -ne $minimum -and
        $null -ne $maximum -and
        -not [string]::IsNullOrWhiteSpace([string]$minimum) -and
        -not [string]::IsNullOrWhiteSpace([string]$maximum) -and
        (Compare-MtsSemVer ([string]$minimum) ([string]$maximum)) -gt 0
    ) {
        throw "minInstalledVersion must not be greater than maxInstalledVersion."
    }

    $dataSchema = Get-MtsObjectProperty $compatibility "dataSchema" -Required
    $readableMin = [int](Get-MtsObjectProperty $dataSchema "readableMin" -Required)
    $readableMax = [int](Get-MtsObjectProperty $dataSchema "readableMax" -Required)
    $writeVersion = [int](Get-MtsObjectProperty $dataSchema "writeVersion" -Required)
    if ($readableMin -lt 1 -or $readableMax -lt $readableMin) {
        throw "Manifest data schema readability range is invalid."
    }
    if ($writeVersion -lt $readableMin -or $writeVersion -gt $readableMax) {
        throw "Manifest data schema writeVersion must be readable by the release."
    }

    $trust = Get-MtsObjectProperty $Manifest "trust" -Required
    $trustMode = [string](Get-MtsObjectProperty $trust "mode" -Required)
    if ($trustMode -notin @("authenticode", "development-unsigned")) {
        throw "Unsupported release trust mode: $trustMode"
    }

    if ($trustMode -eq "authenticode") {
        $thumbprint = ConvertTo-MtsPublisherThumbprint ([string](Get-MtsObjectProperty $trust "publisherThumbprint" -Required))
        $signedFiles = @(Get-MtsObjectProperty $trust "signedFiles" -Required)
        if ($signedFiles.Count -eq 0) {
            throw "Authenticode releases must declare signedFiles."
        }
        if ($signedFiles -notcontains ("payload/" + $entrypoint)) {
            throw "Authenticode releases must include the entrypoint in signedFiles."
        }
        foreach ($signedPath in $signedFiles) {
            $signedPathValue = [string]$signedPath
            if (-not ($signedPathValue.StartsWith("payload/") -or $signedPathValue.StartsWith("installers/"))) {
                throw "Signed file scope is invalid: $signedPathValue"
            }
            $separatorIndex = $signedPathValue.IndexOf("/")
            Assert-MtsRelativePayloadPath $signedPathValue.Substring($separatorIndex + 1) | Out-Null
        }

        $manifestSignature = Get-MtsObjectProperty $trust "manifestSignature" -Required
        if ((Get-MtsObjectProperty $manifestSignature "path" -Required) -ne $script:ReleaseManifestSignatureName) {
            throw "Manifest publisher signature path is invalid."
        }
        if ((Get-MtsObjectProperty $manifestSignature "format" -Required) -ne "cms-detached") {
            throw "Manifest publisher signature format is invalid."
        }
        if ((Get-MtsObjectProperty $manifestSignature "digestAlgorithm" -Required) -ne "sha256") {
            throw "Manifest publisher signature digest algorithm is invalid."
        }
    }
    else {
        if ($null -ne (Get-MtsObjectProperty $trust "publisherThumbprint" -Required)) {
            throw "Unsigned development releases must not declare a publisher thumbprint."
        }
        if (@(Get-MtsObjectProperty $trust "signedFiles" -Required).Count -ne 0) {
            throw "Unsigned development releases must not declare signed files."
        }
        if ($null -ne (Get-MtsObjectProperty $trust "manifestSignature" -Required)) {
            throw "Unsigned development releases must not declare a publisher signature."
        }
    }

    return $Manifest
}

function Assert-MtsPayloadMatchesManifest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PayloadRoot,

        [Parameter(Mandatory = $true)]
        $Manifest
    )

    $expectedFiles = @(Get-MtsObjectProperty (Get-MtsObjectProperty $Manifest "payload" -Required) "files" -Required)
    $actualFiles = @(Get-MtsPayloadLedger $PayloadRoot)
    if ($actualFiles.Count -ne $expectedFiles.Count) {
        throw "Payload file count mismatch. Expected=$($expectedFiles.Count) Actual=$($actualFiles.Count)"
    }

    $actualByPath = @{}
    foreach ($actual in $actualFiles) {
        $actualByPath[$actual.path.ToLowerInvariant()] = $actual
    }

    foreach ($expected in $expectedFiles) {
        $path = [string]$expected.path
        $key = $path.ToLowerInvariant()
        if (-not $actualByPath.ContainsKey($key)) {
            throw "Payload file is missing: $path"
        }
        $actual = $actualByPath[$key]
        if ([int64]$actual.size -ne [int64]$expected.size) {
            throw "Payload size mismatch: $path"
        }
        if (-not [string]::Equals([string]$actual.sha256, [string]$expected.sha256, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Payload SHA-256 mismatch: $path"
        }
    }
}

function Assert-MtsAuthenticodeTrust {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReleaseRoot,

        [Parameter(Mandatory = $true)]
        $Manifest,

        [string]$ExpectedPublisherThumbprint,

        [switch]$AllowUnsignedDevelopment,

        [switch]$InstalledPayloadOnly
    )

    $trust = $Manifest.trust
    $metadataRoot = if ($InstalledPayloadOnly) {
        Join-Path $ReleaseRoot $script:ReservedMetadataDirectory
    }
    else {
        $ReleaseRoot
    }
    $manifestPath = Join-Path $metadataRoot $script:ReleaseManifestName
    $signaturePath = Join-Path $metadataRoot $script:ReleaseManifestSignatureName

    if ($trust.mode -eq "development-unsigned") {
        if (-not $AllowUnsignedDevelopment) {
            throw "Unsigned development releases are rejected unless -AllowUnsignedDevelopment is explicitly supplied."
        }
        if (Test-Path -LiteralPath $signaturePath) {
            throw "Unsigned development releases must not contain a detached publisher signature."
        }
        return
    }

    if ([string]::IsNullOrWhiteSpace($ExpectedPublisherThumbprint)) {
        throw "Production release validation requires -ExpectedPublisherThumbprint as a fixed publisher trust anchor."
    }
    $manifestThumbprint = ConvertTo-MtsPublisherThumbprint ([string]$trust.publisherThumbprint)
    $expectedThumbprint = ConvertTo-MtsPublisherThumbprint $ExpectedPublisherThumbprint
    if (-not [string]::Equals($manifestThumbprint, $expectedThumbprint, [System.StringComparison]::Ordinal)) {
        throw "Manifest publisher thumbprint does not match the configured publisher."
    }
    Assert-MtsDetachedManifestSignature `
        -ManifestPath $manifestPath `
        -SignaturePath $signaturePath `
        -ExpectedPublisherThumbprint $expectedThumbprint

    foreach ($signedFile in @($trust.signedFiles)) {
        $signedFileValue = [string]$signedFile
        $separatorIndex = $signedFileValue.IndexOf("/")
        if ($separatorIndex -le 0 -or $separatorIndex -ge ($signedFileValue.Length - 1)) {
            throw "Signed file scope is invalid: $signedFileValue"
        }
        $scope = $signedFileValue.Substring(0, $separatorIndex)
        $relative = $signedFileValue.Substring($separatorIndex + 1)
        if ($InstalledPayloadOnly -and $scope -eq "installers") {
            continue
        }

        $basePath = if ($InstalledPayloadOnly) {
            $ReleaseRoot
        }
        else {
            Join-Path $ReleaseRoot $scope
        }
        $fullPath = Join-Path $basePath ($relative.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
        if (-not (Test-MtsPathWithin -Candidate $fullPath -Root $basePath)) {
            throw "Signed file escapes its declared scope: $signedFile"
        }
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
            throw "Signed file is missing: $signedFile"
        }

        $signature = Get-AuthenticodeSignature -LiteralPath $fullPath
        $allowedStatuses = @(
            [System.Management.Automation.SignatureStatus]::Valid,
            [System.Management.Automation.SignatureStatus]::UnknownError
        )
        if ($signature.Status -notin $allowedStatuses) {
            throw "Authenticode signature is not valid for $signedFile. Status=$($signature.Status)"
        }
        if ($null -eq $signature.SignerCertificate) {
            throw "Authenticode signer certificate is missing for $signedFile."
        }
        Assert-MtsCodeSigningCertificate $signature.SignerCertificate

        $actualThumbprint = $signature.SignerCertificate.Thumbprint.Replace(" ", "").ToUpperInvariant()
        if (-not [string]::Equals($actualThumbprint, $manifestThumbprint, [System.StringComparison]::Ordinal)) {
            throw "Authenticode signer thumbprint mismatch for $signedFile. Expected=$manifestThumbprint Actual=$actualThumbprint"
        }
        if ($signature.Status -eq [System.Management.Automation.SignatureStatus]::UnknownError) {
            $chain = [System.Security.Cryptography.X509Certificates.X509Chain]::new()
            try {
                $chain.ChainPolicy.RevocationMode = [System.Security.Cryptography.X509Certificates.X509RevocationMode]::NoCheck
                $chain.ChainPolicy.VerificationFlags = [System.Security.Cryptography.X509Certificates.X509VerificationFlags]::AllowUnknownCertificateAuthority
                if (-not $chain.Build($signature.SignerCertificate)) {
                    $chainStatuses = @($chain.ChainStatus | ForEach-Object { [string]$_.Status }) -join ","
                    throw "Authenticode signer chain is invalid for $signedFile. Status=$chainStatuses"
                }
            }
            finally {
                $chain.Dispose()
            }
        }
    }
}

function Read-MtsManifestFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath,

        [Parameter(Mandatory = $true)]
        [string]$ChecksumPath
    )

    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Release manifest is missing: $ManifestPath"
    }
    if (-not (Test-Path -LiteralPath $ChecksumPath -PathType Leaf)) {
        throw "Release manifest checksum is missing: $ChecksumPath"
    }

    $expectedHash = ([System.IO.File]::ReadAllText($ChecksumPath)).Trim().ToLowerInvariant()
    if ($expectedHash -notmatch "^[a-f0-9]{64}$") {
        throw "Release manifest checksum file is invalid."
    }

    $actualHash = Get-MtsFileSha256 $ManifestPath
    if ($actualHash -ne $expectedHash) {
        throw "Release manifest checksum mismatch."
    }

    try {
        $manifest = [System.IO.File]::ReadAllText($ManifestPath, (Get-MtsUtf8NoBomEncoding)) | ConvertFrom-Json
    }
    catch {
        throw "Release manifest is not valid JSON: $($_.Exception.Message)"
    }

    Assert-MtsManifestContract $manifest | Out-Null
    return [pscustomobject][ordered]@{
        Manifest = $manifest
        ManifestSha256 = $actualHash
    }
}

function Test-MtsReleaseBundle {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReleaseDirectory,

        [string]$ExpectedAppId,

        [string]$ExpectedPublisherThumbprint,

        [switch]$AllowUnsignedDevelopment
    )

    $releaseRoot = Get-MtsCanonicalFullPath $ReleaseDirectory
    if (-not (Test-Path -LiteralPath $releaseRoot -PathType Container)) {
        throw "Release directory does not exist: $releaseRoot"
    }

    Assert-MtsNoReparsePoints $releaseRoot
    $allowedRootEntries = @(
        $script:ReleaseManifestName,
        $script:ReleaseManifestChecksumName,
        $script:ReleaseManifestSignatureName,
        "payload",
        "installers"
    )
    foreach ($entry in @(Get-ChildItem -LiteralPath $releaseRoot -Force)) {
        if ($entry.Name -notin $allowedRootEntries) {
            throw "Release root contains an undeclared entry: $($entry.Name)"
        }
    }

    $manifestRecord = Read-MtsManifestFile `
        -ManifestPath (Join-Path $releaseRoot $script:ReleaseManifestName) `
        -ChecksumPath (Join-Path $releaseRoot $script:ReleaseManifestChecksumName)
    $manifest = $manifestRecord.Manifest

    if (
        -not [string]::IsNullOrWhiteSpace($ExpectedAppId) -and
        -not [string]::Equals([string]$manifest.appId, $ExpectedAppId, [System.StringComparison]::Ordinal)
    ) {
        throw "Release appId does not match the expected appId."
    }

    $payloadRoot = Join-Path $releaseRoot "payload"
    Assert-MtsPayloadMatchesManifest -PayloadRoot $payloadRoot -Manifest $manifest

    $installerDirectory = Join-Path $releaseRoot "installers"
    $declaredInstallers = @()
    $nativeInstallersProperty = $manifest.PSObject.Properties["nativeInstallers"]
    if ($null -ne $nativeInstallersProperty) {
        $declaredInstallers = @($nativeInstallersProperty.Value)
    }

    if ($declaredInstallers.Count -gt 0) {
        if (-not (Test-Path -LiteralPath $installerDirectory -PathType Container)) {
            throw "Manifest declares native installers, but installers/ is missing."
        }
        $actualInstallers = @(Get-MtsPayloadLedger $installerDirectory)
        if ($actualInstallers.Count -ne $declaredInstallers.Count) {
            throw "Native installer ledger count mismatch."
        }
        $actualInstallerMap = @{}
        foreach ($installer in $actualInstallers) {
            $actualInstallerMap[$installer.path.ToLowerInvariant()] = $installer
        }
        foreach ($declared in $declaredInstallers) {
            $path = Assert-MtsRelativePayloadPath ([string]$declared.path)
            if (-not $actualInstallerMap.ContainsKey($path.ToLowerInvariant())) {
                throw "Native installer is missing: $path"
            }
            $actual = $actualInstallerMap[$path.ToLowerInvariant()]
            if ([int64]$actual.size -ne [int64]$declared.size -or $actual.sha256 -ne [string]$declared.sha256) {
                throw "Native installer hash or size mismatch: $path"
            }
        }
    }
    elseif (Test-Path -LiteralPath $installerDirectory) {
        $extraInstallers = @(Get-ChildItem -LiteralPath $installerDirectory -File -Recurse -Force)
        if ($extraInstallers.Count -gt 0) {
            throw "Release contains undeclared native installer files."
        }
    }

    Assert-MtsAuthenticodeTrust `
        -ReleaseRoot $releaseRoot `
        -Manifest $manifest `
        -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
        -AllowUnsignedDevelopment:$AllowUnsignedDevelopment

    return [pscustomobject][ordered]@{
        ReleaseRoot = $releaseRoot
        Manifest = $manifest
        ManifestSha256 = $manifestRecord.ManifestSha256
        PayloadRoot = $payloadRoot
    }
}

function New-MtsRecordEnvelope {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RecordContract,

        [Parameter(Mandatory = $true)]
        $Record
    )

    $recordJson = ConvertTo-MtsJson -Value $Record -Compressed
    return [pscustomobject][ordered]@{
        contract = $script:StateEnvelopeContract
        recordContract = $RecordContract
        recordSha256 = Get-MtsStringSha256 $recordJson
        recordJson = $recordJson
    }
}

function Write-MtsRecordEnvelope {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$RecordContract,

        [Parameter(Mandatory = $true)]
        $Record
    )

    $envelope = New-MtsRecordEnvelope -RecordContract $RecordContract -Record $Record
    $content = (ConvertTo-MtsJson -Value $envelope) + "`n"
    Write-MtsAtomicUtf8File -Path $Path -Content $content
}

function Read-MtsRecordEnvelope {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedRecordContract
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Record file is missing: $Path"
    }

    try {
        $envelope = [System.IO.File]::ReadAllText($Path, (Get-MtsUtf8NoBomEncoding)) | ConvertFrom-Json
    }
    catch {
        throw "Record envelope is not valid JSON: $Path"
    }

    if ($envelope.contract -ne $script:StateEnvelopeContract) {
        throw "Record envelope contract is invalid: $Path"
    }
    if ($envelope.recordContract -ne $ExpectedRecordContract) {
        throw "Record contract is invalid: $Path"
    }
    if ([string]$envelope.recordSha256 -notmatch "^[a-f0-9]{64}$") {
        throw "Record checksum is invalid: $Path"
    }

    $actualHash = Get-MtsStringSha256 ([string]$envelope.recordJson)
    if ($actualHash -ne [string]$envelope.recordSha256) {
        throw "Record checksum mismatch: $Path"
    }

    try {
        return ([string]$envelope.recordJson | ConvertFrom-Json)
    }
    catch {
        throw "Embedded record JSON is invalid: $Path"
    }
}

function Get-MtsControlPaths {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallRoot
    )

    $controlRoot = Join-Path $InstallRoot $script:ControlDirectoryName
    return [pscustomobject][ordered]@{
        App = Join-Path $InstallRoot "app"
        Backups = Join-Path $InstallRoot "backups"
        Control = $controlRoot
        Transactions = Join-Path $controlRoot "transactions"
        State = Join-Path $controlRoot "install-state.json"
        Journal = Join-Path $controlRoot "transaction.json"
    }
}

function Read-MtsInstallState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallRoot,

        [Parameter(Mandatory = $true)]
        [string]$DataRoot,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedAppId
    )

    $paths = Get-MtsControlPaths $InstallRoot
    $state = Read-MtsRecordEnvelope -Path $paths.State -ExpectedRecordContract $script:StateContract
    if ($state.schemaVersion -ne "1.0.0" -or $state.contract -ne $script:StateContract) {
        throw "Install state contract is unsupported."
    }
    if ($state.status -ne "committed") {
        throw "Install state is not committed."
    }
    if ($state.appId -ne $ExpectedAppId) {
        throw "Install state appId does not match."
    }
    if (-not (Test-MtsPathEqual $state.installRoot $InstallRoot)) {
        throw "Install state root does not match the requested InstallRoot."
    }
    if (-not (Test-MtsPathEqual $state.dataRoot $DataRoot)) {
        throw "Install state data root does not match the requested DataRoot."
    }

    ConvertTo-MtsSemVer ([string]$state.current.version) | Out-Null
    if ([string]$state.current.manifestSha256 -notmatch "^[a-f0-9]{64}$") {
        throw "Install state manifest hash is invalid."
    }
    if ([int]$state.current.dataSchemaVersion -lt 1) {
        throw "Install state data schema version is invalid."
    }

    return $state
}

function Read-MtsInstalledManifest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AppPath,

        [string]$ExpectedPublisherThumbprint,

        [switch]$AllowUnsignedDevelopment
    )

    $metadataRoot = Join-Path $AppPath $script:ReservedMetadataDirectory
    $metadataEntries = @(Get-ChildItem -LiteralPath $metadataRoot -Force)
    foreach ($entry in $metadataEntries) {
        if (
            $entry.PSIsContainer -or
            $entry.Name -notin @(
                $script:ReleaseManifestName,
                $script:ReleaseManifestChecksumName,
                $script:ReleaseManifestSignatureName
            )
        ) {
            throw "Installed release metadata contains an undeclared entry: $($entry.Name)"
        }
    }

    $manifestRecord = Read-MtsManifestFile `
        -ManifestPath (Join-Path $metadataRoot $script:ReleaseManifestName) `
        -ChecksumPath (Join-Path $metadataRoot $script:ReleaseManifestChecksumName)

    $manifest = $manifestRecord.Manifest
    $expectedMetadataCount = if ($manifest.trust.mode -eq "authenticode") { 3 } else { 2 }
    if ($metadataEntries.Count -ne $expectedMetadataCount) {
        throw "Installed release metadata entry count does not match its trust mode."
    }
    # Validate the exact installed ledger without copying. Get-MtsPayloadLedger cannot exclude metadata,
    # so compare the manifest against a direct ledger of app files excluding the reserved directory.
    Assert-MtsNoReparsePoints $AppPath
    $actualFiles = @()
    foreach ($file in @(Get-ChildItem -LiteralPath $AppPath -File -Force -Recurse | Sort-Object FullName)) {
        if (Test-MtsPathWithin -Candidate $file.FullName -Root $metadataRoot -AllowEqual) {
            continue
        }
        $relative = Get-MtsManifestRelativePath -Root $AppPath -Path $file.FullName
        $actualFiles += [pscustomobject][ordered]@{
            path = $relative
            size = [int64]$file.Length
            sha256 = Get-MtsFileSha256 $file.FullName
        }
    }

    $expectedFiles = @($manifest.payload.files)
    if ($actualFiles.Count -ne $expectedFiles.Count) {
        throw "Installed payload file count mismatch."
    }
    $actualMap = @{}
    foreach ($actual in $actualFiles) {
        $actualMap[$actual.path.ToLowerInvariant()] = $actual
    }
    foreach ($expected in $expectedFiles) {
        $key = ([string]$expected.path).ToLowerInvariant()
        if (-not $actualMap.ContainsKey($key)) {
            throw "Installed payload file is missing: $($expected.path)"
        }
        $actual = $actualMap[$key]
        if ([int64]$actual.size -ne [int64]$expected.size -or $actual.sha256 -ne [string]$expected.sha256) {
            throw "Installed payload hash or size mismatch: $($expected.path)"
        }
    }

    Assert-MtsAuthenticodeTrust `
        -ReleaseRoot $AppPath `
        -Manifest $manifest `
        -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
        -AllowUnsignedDevelopment:$AllowUnsignedDevelopment `
        -InstalledPayloadOnly

    return [pscustomobject][ordered]@{
        Manifest = $manifest
        ManifestSha256 = $manifestRecord.ManifestSha256
        AppPath = (Get-MtsCanonicalFullPath $AppPath)
    }
}

function Copy-MtsReleaseToStage {
    param(
        [Parameter(Mandatory = $true)]
        $Release,

        [Parameter(Mandatory = $true)]
        [string]$StagePath
    )

    if (Test-Path -LiteralPath $StagePath) {
        throw "Stage path already exists: $StagePath"
    }
    [System.IO.Directory]::CreateDirectory($StagePath) | Out-Null

    foreach ($file in @($Release.Manifest.payload.files)) {
        $relative = [string]$file.path
        $source = Join-Path $Release.PayloadRoot ($relative.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
        $destination = Join-Path $StagePath ($relative.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
        if (-not (Test-MtsPathWithin -Candidate $destination -Root $StagePath)) {
            throw "Stage destination escapes the transaction directory: $relative"
        }

        [System.IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
        if ((Get-MtsFileSha256 $destination) -ne [string]$file.sha256) {
            throw "Staged payload hash mismatch: $relative"
        }
    }

    $metadataRoot = Join-Path $StagePath $script:ReservedMetadataDirectory
    [System.IO.Directory]::CreateDirectory($metadataRoot) | Out-Null
    Copy-Item -LiteralPath (Join-Path $Release.ReleaseRoot $script:ReleaseManifestName) -Destination $metadataRoot
    Copy-Item -LiteralPath (Join-Path $Release.ReleaseRoot $script:ReleaseManifestChecksumName) -Destination $metadataRoot
    $signaturePath = Join-Path $Release.ReleaseRoot $script:ReleaseManifestSignatureName
    if (Test-Path -LiteralPath $signaturePath -PathType Leaf) {
        Copy-Item -LiteralPath $signaturePath -Destination $metadataRoot
    }
}

function Assert-MtsUpgradeCompatibility {
    param(
        [Parameter(Mandatory = $true)]
        $CurrentState,

        [Parameter(Mandatory = $true)]
        $TargetManifest,

        [switch]$AllowDowngrade
    )

    $currentVersion = [string]$CurrentState.current.version
    $targetVersion = [string]$TargetManifest.version
    $comparison = Compare-MtsSemVer $targetVersion $currentVersion

    if ($comparison -lt 0 -and -not $AllowDowngrade) {
        throw "Version downgrade is rejected by default. Current=$currentVersion Target=$targetVersion"
    }

    if ($comparison -eq 0) {
        throw "Same-version replacement is forbidden. Use an incremented immutable release version."
    }

    $minimum = $TargetManifest.compatibility.minInstalledVersion
    $maximum = $TargetManifest.compatibility.maxInstalledVersion
    if (
        $null -ne $minimum -and
        -not [string]::IsNullOrWhiteSpace([string]$minimum) -and
        (Compare-MtsSemVer $currentVersion ([string]$minimum)) -lt 0
    ) {
        throw "Installed version is below the release upgrade floor. Current=$currentVersion Minimum=$minimum"
    }
    if (
        $null -ne $maximum -and
        -not [string]::IsNullOrWhiteSpace([string]$maximum) -and
        (Compare-MtsSemVer $currentVersion ([string]$maximum)) -gt 0
    ) {
        throw "Installed version is above the release upgrade ceiling. Current=$currentVersion Maximum=$maximum"
    }

    $dataSchemaVersion = [int]$CurrentState.current.dataSchemaVersion
    $readableMin = [int]$TargetManifest.compatibility.dataSchema.readableMin
    $readableMax = [int]$TargetManifest.compatibility.dataSchema.readableMax
    if ($dataSchemaVersion -lt $readableMin -or $dataSchemaVersion -gt $readableMax) {
        throw "Target release cannot read the installed user-data schema. Data=$dataSchemaVersion Readable=$readableMin..$readableMax"
    }
}

function New-MtsTransactionJournal {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TransactionId,

        [Parameter(Mandatory = $true)]
        [string]$Action,

        [Parameter(Mandatory = $true)]
        [string]$InstallRoot,

        [Parameter(Mandatory = $true)]
        [string]$DataRoot,

        [Parameter(Mandatory = $true)]
        [string]$StagePath,

        [string]$BackupPath,

        $PreviousState,

        [Parameter(Mandatory = $true)]
        [string]$TargetManifestSha256,

        [Parameter(Mandatory = $true)]
        [string]$Phase
    )

    return [pscustomobject][ordered]@{
        schemaVersion = "1.0.0"
        contract = $script:JournalContract
        transactionId = $TransactionId
        action = $Action
        phase = $Phase
        installRoot = $InstallRoot
        dataRoot = $DataRoot
        stagePath = $StagePath
        backupPath = $BackupPath
        targetManifestSha256 = $TargetManifestSha256
        previousState = $PreviousState
    }
}

function Set-MtsJournalPhase {
    param(
        [Parameter(Mandatory = $true)]
        [string]$JournalPath,

        [Parameter(Mandatory = $true)]
        $Journal,

        [Parameter(Mandatory = $true)]
        [string]$Phase
    )

    $Journal.phase = $Phase
    Write-MtsRecordEnvelope -Path $JournalPath -RecordContract $script:JournalContract -Record $Journal
}

function New-MtsCommittedState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AppId,

        [Parameter(Mandatory = $true)]
        [string]$InstallRoot,

        [Parameter(Mandatory = $true)]
        [string]$DataRoot,

        [Parameter(Mandatory = $true)]
        $CurrentRelease,

        [Parameter(Mandatory = $true)]
        [int]$DataSchemaVersion,

        [string]$RollbackPath,

        $RollbackRelease,

        [Parameter(Mandatory = $true)]
        [string]$TransactionId
    )

    $rollback = $null
    if (-not [string]::IsNullOrWhiteSpace($RollbackPath) -and $null -ne $RollbackRelease) {
        $rollback = [pscustomobject][ordered]@{
            path = $RollbackPath
            version = [string]$RollbackRelease.Manifest.version
            releaseId = [string]$RollbackRelease.Manifest.releaseId
            manifestSha256 = [string]$RollbackRelease.ManifestSha256
        }
    }

    return [pscustomobject][ordered]@{
        schemaVersion = "1.0.0"
        contract = $script:StateContract
        status = "committed"
        appId = $AppId
        installRoot = $InstallRoot
        dataRoot = $DataRoot
        current = [pscustomobject][ordered]@{
            version = [string]$CurrentRelease.Manifest.version
            releaseId = [string]$CurrentRelease.Manifest.releaseId
            manifestSha256 = [string]$CurrentRelease.ManifestSha256
            entrypoint = [string]$CurrentRelease.Manifest.entrypoint
            dataSchemaVersion = $DataSchemaVersion
        }
        rollback = $rollback
        lastTransactionId = $TransactionId
    }
}

function Remove-MtsTransactionArtifacts {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallRoot,

        [Parameter(Mandatory = $true)]
        [string]$TransactionRoot,

        [Parameter(Mandatory = $true)]
        [string]$JournalPath
    )

    if (-not (Test-MtsPathWithin -Candidate $TransactionRoot -Root $InstallRoot)) {
        throw "Refusing to clean transaction path outside InstallRoot."
    }

    if (Test-Path -LiteralPath $TransactionRoot) {
        Remove-Item -LiteralPath $TransactionRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $JournalPath) {
        Remove-Item -LiteralPath $JournalPath -Force
    }
}

function Invoke-MtsDeploy {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("Install", "Upgrade")]
        [string]$Action,

        [Parameter(Mandatory = $true)]
        [string]$InstallRoot,

        [Parameter(Mandatory = $true)]
        [string]$DataRoot,

        [Parameter(Mandatory = $true)]
        $Release,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedAppId,

        [string]$ExpectedPublisherThumbprint,

        [switch]$AllowUnsignedDevelopment,

        [switch]$AllowDowngrade,

        [switch]$DryRun
    )

    $paths = Get-MtsControlPaths $InstallRoot
    $hasState = Test-Path -LiteralPath $paths.State -PathType Leaf
    $previousState = $null
    $previousRelease = $null

    if ($Action -eq "Install" -and $hasState) {
        throw "Install requires an empty managed installation. Use Upgrade for an existing installation."
    }
    if ($Action -eq "Upgrade" -and -not $hasState) {
        throw "Upgrade requires a committed existing installation."
    }
    if ((Test-Path -LiteralPath $paths.Journal) -and -not $DryRun) {
        throw "An unfinished transaction exists. Run Recover before deployment."
    }

    if ($hasState) {
        $previousState = Read-MtsInstallState -InstallRoot $InstallRoot -DataRoot $DataRoot -ExpectedAppId $ExpectedAppId
        $previousRelease = Read-MtsInstalledManifest `
            -AppPath $paths.App `
            -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
            -AllowUnsignedDevelopment:$AllowUnsignedDevelopment

        if ($previousState.current.manifestSha256 -ne $previousRelease.ManifestSha256) {
            throw "Install state and installed manifest hash do not reconcile."
        }
        Assert-MtsUpgradeCompatibility `
            -CurrentState $previousState `
            -TargetManifest $Release.Manifest `
            -AllowDowngrade:$AllowDowngrade
    }
    elseif (Test-Path -LiteralPath $paths.App) {
        throw "Unmanaged app directory already exists. Refusing to overwrite it."
    }

    $transactionId = [guid]::NewGuid().ToString("N")
    $transactionRoot = Join-Path $paths.Transactions $transactionId
    $stagePath = Join-Path $transactionRoot "staged-app"
    $backupPath = $null
    if ($null -ne $previousRelease) {
        $backupName = "{0}--{1}--{2}" -f (
            ([string]$previousRelease.Manifest.version -replace "[^A-Za-z0-9._-]", "_"),
            $previousRelease.ManifestSha256.Substring(0, 12),
            $transactionId
        )
        $backupPath = Join-Path $paths.Backups $backupName
    }

    $operations = @(
        "verify-release-ledger",
        "stage-payload-inside-install-volume",
        "verify-staged-payload",
        "persist-transaction-journal"
    )
    if ($null -ne $previousRelease) {
        $operations += "rename-current-to-versioned-backup"
    }
    $operations += @(
        "rename-stage-to-current",
        "verify-current-ledger-and-entrypoint",
        "atomically-commit-install-state",
        "read-back-state-and-reconcile",
        "clear-transaction-journal"
    )

    if ($DryRun) {
        return [pscustomobject][ordered]@{
            ok = $true
            action = $Action
            dryRun = $true
            installRoot = $InstallRoot
            dataRoot = $DataRoot
            currentVersion = if ($null -eq $previousState) { $null } else { [string]$previousState.current.version }
            targetVersion = [string]$Release.Manifest.version
            userDataPolicy = "preserve"
            operations = $operations
        }
    }

    [System.IO.Directory]::CreateDirectory($paths.Transactions) | Out-Null
    [System.IO.Directory]::CreateDirectory($paths.Backups) | Out-Null
    [System.IO.Directory]::CreateDirectory($DataRoot) | Out-Null

    $journal = New-MtsTransactionJournal `
        -TransactionId $transactionId `
        -Action $Action `
        -InstallRoot $InstallRoot `
        -DataRoot $DataRoot `
        -StagePath $stagePath `
        -BackupPath $backupPath `
        -PreviousState $previousState `
        -TargetManifestSha256 $Release.ManifestSha256 `
        -Phase "initializing"

    $movedPrevious = $false
    $activatedNew = $false
    try {
        Copy-MtsReleaseToStage -Release $Release -StagePath $stagePath
        $stagedRelease = Read-MtsInstalledManifest `
            -AppPath $stagePath `
            -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
            -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
        if ($stagedRelease.ManifestSha256 -ne $Release.ManifestSha256) {
            throw "Staged release manifest does not reconcile with the source release."
        }

        Set-MtsJournalPhase -JournalPath $paths.Journal -Journal $journal -Phase "staged"

        if ($null -ne $previousRelease) {
            if (Test-Path -LiteralPath $backupPath) {
                throw "Backup path already exists: $backupPath"
            }
            Move-Item -LiteralPath $paths.App -Destination $backupPath
            $movedPrevious = $true
            Set-MtsJournalPhase -JournalPath $paths.Journal -Journal $journal -Phase "current-backed-up"
        }

        Move-Item -LiteralPath $stagePath -Destination $paths.App
        $activatedNew = $true
        Set-MtsJournalPhase -JournalPath $paths.Journal -Journal $journal -Phase "new-current-active"

        $currentRelease = Read-MtsInstalledManifest `
            -AppPath $paths.App `
            -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
            -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
        if ($currentRelease.ManifestSha256 -ne $Release.ManifestSha256) {
            throw "Activated release manifest does not reconcile with the source release."
        }

        $dataSchemaVersion = if ($null -eq $previousState) {
            [int]$Release.Manifest.compatibility.dataSchema.writeVersion
        }
        else {
            [int]$previousState.current.dataSchemaVersion
        }

        $state = New-MtsCommittedState `
            -AppId $ExpectedAppId `
            -InstallRoot $InstallRoot `
            -DataRoot $DataRoot `
            -CurrentRelease $currentRelease `
            -DataSchemaVersion $dataSchemaVersion `
            -RollbackPath $backupPath `
            -RollbackRelease $previousRelease `
            -TransactionId $transactionId
        Write-MtsRecordEnvelope -Path $paths.State -RecordContract $script:StateContract -Record $state
        Set-MtsJournalPhase -JournalPath $paths.Journal -Journal $journal -Phase "state-committed"

        $readBackState = Read-MtsInstallState -InstallRoot $InstallRoot -DataRoot $DataRoot -ExpectedAppId $ExpectedAppId
        if (
            $readBackState.current.manifestSha256 -ne $currentRelease.ManifestSha256 -or
            $readBackState.current.version -ne $currentRelease.Manifest.version
        ) {
            throw "Committed install state did not survive read-back reconciliation."
        }

        $readBackRelease = Read-MtsInstalledManifest `
            -AppPath $paths.App `
            -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
            -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
        if ($readBackRelease.ManifestSha256 -ne $readBackState.current.manifestSha256) {
            throw "Post-commit installed payload does not reconcile with install state."
        }

        Remove-MtsTransactionArtifacts -InstallRoot $InstallRoot -TransactionRoot $transactionRoot -JournalPath $paths.Journal
        return [pscustomobject][ordered]@{
            ok = $true
            action = $Action
            dryRun = $false
            status = "committed-and-verified"
            installRoot = $InstallRoot
            dataRoot = $DataRoot
            version = [string]$readBackState.current.version
            manifestSha256 = [string]$readBackState.current.manifestSha256
            rollbackAvailable = ($null -ne $readBackState.rollback)
            userDataPolicy = "preserved"
        }
    }
    catch {
        $deploymentError = $_
        try {
            if ($activatedNew -and (Test-Path -LiteralPath $paths.App)) {
                $failedPath = Join-Path $transactionRoot "failed-new-current"
                if (Test-Path -LiteralPath $failedPath) {
                    Remove-Item -LiteralPath $failedPath -Recurse -Force
                }
                Move-Item -LiteralPath $paths.App -Destination $failedPath
                $activatedNew = $false
            }
            if ($movedPrevious -and (Test-Path -LiteralPath $backupPath) -and -not (Test-Path -LiteralPath $paths.App)) {
                Move-Item -LiteralPath $backupPath -Destination $paths.App
                $movedPrevious = $false
            }

            if ($null -eq $previousState) {
                if (Test-Path -LiteralPath $paths.State) {
                    Remove-Item -LiteralPath $paths.State -Force
                }
            }
            else {
                Write-MtsRecordEnvelope -Path $paths.State -RecordContract $script:StateContract -Record $previousState
            }

            if (-not $movedPrevious -and -not $activatedNew) {
                Remove-MtsTransactionArtifacts -InstallRoot $InstallRoot -TransactionRoot $transactionRoot -JournalPath $paths.Journal
            }
        }
        catch {
            throw "Deployment failed and automatic rollback also failed. Run Recover. Deployment=$($deploymentError.Exception.Message) Rollback=$($_.Exception.Message)"
        }

        throw "Deployment failed; previous installation was restored. $($deploymentError.Exception.Message)"
    }
}

function Invoke-MtsRollback {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallRoot,

        [Parameter(Mandatory = $true)]
        [string]$DataRoot,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedAppId,

        [string]$ExpectedPublisherThumbprint,

        [switch]$AllowUnsignedDevelopment,

        [switch]$AllowDowngrade,

        [switch]$DryRun
    )

    if (-not $AllowDowngrade) {
        throw "Rollback is a version downgrade and requires explicit -AllowDowngrade consent."
    }

    $paths = Get-MtsControlPaths $InstallRoot
    if (Test-Path -LiteralPath $paths.Journal) {
        throw "An unfinished transaction exists. Run Recover before rollback."
    }

    $state = Read-MtsInstallState -InstallRoot $InstallRoot -DataRoot $DataRoot -ExpectedAppId $ExpectedAppId
    if ($null -eq $state.rollback) {
        throw "No verified rollback release is recorded."
    }

    $currentRelease = Read-MtsInstalledManifest `
        -AppPath $paths.App `
        -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
        -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
    if ($currentRelease.ManifestSha256 -ne $state.current.manifestSha256) {
        throw "Current application does not reconcile with install state."
    }

    $rollbackPath = Get-MtsCanonicalFullPath ([string]$state.rollback.path)
    if (-not (Test-MtsPathWithin -Candidate $rollbackPath -Root $paths.Backups)) {
        throw "Recorded rollback path escapes the managed backups directory."
    }
    $rollbackRelease = Read-MtsInstalledManifest `
        -AppPath $rollbackPath `
        -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
        -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
    if ($rollbackRelease.ManifestSha256 -ne $state.rollback.manifestSha256) {
        throw "Rollback application does not reconcile with install state."
    }

    $dataSchemaVersion = [int]$state.current.dataSchemaVersion
    $readableMin = [int]$rollbackRelease.Manifest.compatibility.dataSchema.readableMin
    $readableMax = [int]$rollbackRelease.Manifest.compatibility.dataSchema.readableMax
    if ($dataSchemaVersion -lt $readableMin -or $dataSchemaVersion -gt $readableMax) {
        throw "Rollback release cannot read the current user-data schema."
    }

    if ($DryRun) {
        return [pscustomobject][ordered]@{
            ok = $true
            action = "Rollback"
            dryRun = $true
            currentVersion = [string]$currentRelease.Manifest.version
            targetVersion = [string]$rollbackRelease.Manifest.version
            userDataPolicy = "preserve"
            operations = @(
                "verify-current-and-rollback-ledgers",
                "rename-current-to-swap-backup",
                "rename-rollback-to-current",
                "verify-current-ledger-and-entrypoint",
                "atomically-commit-install-state",
                "read-back-state-and-reconcile"
            )
        }
    }

    $transactionId = [guid]::NewGuid().ToString("N")
    $transactionRoot = Join-Path $paths.Transactions $transactionId
    $swapPath = Join-Path $paths.Backups (
        "{0}--{1}--{2}" -f (
            ([string]$currentRelease.Manifest.version -replace "[^A-Za-z0-9._-]", "_"),
            $currentRelease.ManifestSha256.Substring(0, 12),
            $transactionId
        )
    )
    [System.IO.Directory]::CreateDirectory($transactionRoot) | Out-Null

    $journal = New-MtsTransactionJournal `
        -TransactionId $transactionId `
        -Action "Rollback" `
        -InstallRoot $InstallRoot `
        -DataRoot $DataRoot `
        -StagePath $rollbackPath `
        -BackupPath $swapPath `
        -PreviousState $state `
        -TargetManifestSha256 $rollbackRelease.ManifestSha256 `
        -Phase "staged"
    Write-MtsRecordEnvelope -Path $paths.Journal -RecordContract $script:JournalContract -Record $journal

    try {
        Move-Item -LiteralPath $paths.App -Destination $swapPath
        Set-MtsJournalPhase -JournalPath $paths.Journal -Journal $journal -Phase "current-backed-up"
        Move-Item -LiteralPath $rollbackPath -Destination $paths.App
        Set-MtsJournalPhase -JournalPath $paths.Journal -Journal $journal -Phase "new-current-active"

        $activatedRelease = Read-MtsInstalledManifest `
            -AppPath $paths.App `
            -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
            -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
        $newState = New-MtsCommittedState `
            -AppId $ExpectedAppId `
            -InstallRoot $InstallRoot `
            -DataRoot $DataRoot `
            -CurrentRelease $activatedRelease `
            -DataSchemaVersion $dataSchemaVersion `
            -RollbackPath $swapPath `
            -RollbackRelease $currentRelease `
            -TransactionId $transactionId
        Write-MtsRecordEnvelope -Path $paths.State -RecordContract $script:StateContract -Record $newState
        Set-MtsJournalPhase -JournalPath $paths.Journal -Journal $journal -Phase "state-committed"

        $readBackState = Read-MtsInstallState -InstallRoot $InstallRoot -DataRoot $DataRoot -ExpectedAppId $ExpectedAppId
        $readBackRelease = Read-MtsInstalledManifest `
            -AppPath $paths.App `
            -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
            -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
        if ($readBackState.current.manifestSha256 -ne $readBackRelease.ManifestSha256) {
            throw "Rollback state did not reconcile after commit."
        }

        Remove-MtsTransactionArtifacts -InstallRoot $InstallRoot -TransactionRoot $transactionRoot -JournalPath $paths.Journal
        return [pscustomobject][ordered]@{
            ok = $true
            action = "Rollback"
            dryRun = $false
            status = "committed-and-verified"
            version = [string]$readBackState.current.version
            rollbackAvailable = $true
            userDataPolicy = "preserved"
        }
    }
    catch {
        $rollbackError = $_
        try {
            if (Test-Path -LiteralPath $paths.App) {
                $failedPath = Join-Path $transactionRoot "failed-rollback-current"
                Move-Item -LiteralPath $paths.App -Destination $failedPath
            }
            if (Test-Path -LiteralPath $swapPath) {
                Move-Item -LiteralPath $swapPath -Destination $paths.App
            }
            Write-MtsRecordEnvelope -Path $paths.State -RecordContract $script:StateContract -Record $state
            Remove-MtsTransactionArtifacts -InstallRoot $InstallRoot -TransactionRoot $transactionRoot -JournalPath $paths.Journal
        }
        catch {
            throw "Rollback failed and restoration failed. Run Recover. Rollback=$($rollbackError.Exception.Message) Restoration=$($_.Exception.Message)"
        }
        throw "Rollback failed; the prior current release was restored. $($rollbackError.Exception.Message)"
    }
}

function Invoke-MtsRecover {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallRoot,

        [Parameter(Mandatory = $true)]
        [string]$DataRoot,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedAppId,

        [string]$ExpectedPublisherThumbprint,

        [switch]$AllowUnsignedDevelopment,

        [switch]$DryRun
    )

    $paths = Get-MtsControlPaths $InstallRoot
    if (-not (Test-Path -LiteralPath $paths.Journal -PathType Leaf)) {
        if (Test-Path -LiteralPath $paths.State -PathType Leaf) {
            $state = Read-MtsInstallState -InstallRoot $InstallRoot -DataRoot $DataRoot -ExpectedAppId $ExpectedAppId
            $release = Read-MtsInstalledManifest `
                -AppPath $paths.App `
                -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
                -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
            if ($state.current.manifestSha256 -ne $release.ManifestSha256) {
                throw "No transaction journal exists, but current state and payload do not reconcile."
            }
        }
        return [pscustomobject][ordered]@{
            ok = $true
            action = "Recover"
            dryRun = [bool]$DryRun
            status = "healthy-no-recovery-needed"
        }
    }

    $journal = Read-MtsRecordEnvelope -Path $paths.Journal -ExpectedRecordContract $script:JournalContract
    if ($journal.contract -ne $script:JournalContract) {
        throw "Transaction journal contract is unsupported."
    }
    if (-not (Test-MtsPathEqual $journal.installRoot $InstallRoot) -or -not (Test-MtsPathEqual $journal.dataRoot $DataRoot)) {
        throw "Transaction journal roots do not match the requested roots."
    }

    $transactionRoot = Split-Path -Parent ([string]$journal.stagePath)
    if (-not (Test-MtsPathWithin -Candidate $transactionRoot -Root $paths.Transactions)) {
        throw "Transaction journal stage path escapes the managed transaction directory."
    }
    if (
        $null -ne $journal.backupPath -and
        -not [string]::IsNullOrWhiteSpace([string]$journal.backupPath) -and
        -not (Test-MtsPathWithin -Candidate ([string]$journal.backupPath) -Root $paths.Backups)
    ) {
        throw "Transaction journal backup path escapes the managed backups directory."
    }

    $phase = [string]$journal.phase
    $restorePrevious = $phase -in @("current-backed-up", "new-current-active")
    $finalizeCommitted = $phase -eq "state-committed"
    $discardStage = $phase -in @("initializing", "staged")
    if (-not ($restorePrevious -or $finalizeCommitted -or $discardStage)) {
        throw "Transaction journal phase is unsupported: $phase"
    }

    if ($DryRun) {
        return [pscustomobject][ordered]@{
            ok = $true
            action = "Recover"
            dryRun = $true
            phase = $phase
            recovery = if ($restorePrevious) { "restore-previous" } elseif ($finalizeCommitted) { "verify-and-finalize" } else { "discard-uncommitted-stage" }
        }
    }

    if ($discardStage) {
        Remove-MtsTransactionArtifacts -InstallRoot $InstallRoot -TransactionRoot $transactionRoot -JournalPath $paths.Journal
        return [pscustomobject][ordered]@{
            ok = $true
            action = "Recover"
            dryRun = $false
            status = "uncommitted-stage-discarded"
        }
    }

    if ($finalizeCommitted) {
        $state = Read-MtsInstallState -InstallRoot $InstallRoot -DataRoot $DataRoot -ExpectedAppId $ExpectedAppId
        $release = Read-MtsInstalledManifest `
            -AppPath $paths.App `
            -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
            -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
        if ($state.current.manifestSha256 -ne $release.ManifestSha256) {
            throw "Committed transaction cannot be finalized because state and payload do not reconcile."
        }
        Remove-MtsTransactionArtifacts -InstallRoot $InstallRoot -TransactionRoot $transactionRoot -JournalPath $paths.Journal
        return [pscustomobject][ordered]@{
            ok = $true
            action = "Recover"
            dryRun = $false
            status = "committed-transaction-finalized"
        }
    }

    $previousState = $journal.previousState
    if ($null -eq $previousState) {
        if (Test-Path -LiteralPath $paths.App) {
            $failedPath = Join-Path $transactionRoot "discarded-uncommitted-current"
            Move-Item -LiteralPath $paths.App -Destination $failedPath
        }
        if (Test-Path -LiteralPath $paths.State) {
            Remove-Item -LiteralPath $paths.State -Force
        }
        Remove-MtsTransactionArtifacts -InstallRoot $InstallRoot -TransactionRoot $transactionRoot -JournalPath $paths.Journal
        return [pscustomobject][ordered]@{
            ok = $true
            action = "Recover"
            dryRun = $false
            status = "uncommitted-install-removed"
        }
    }

    $backupPath = [string]$journal.backupPath
    if (-not (Test-Path -LiteralPath $backupPath -PathType Container)) {
        throw "Cannot recover previous release because the verified backup is missing."
    }
    $backupRelease = Read-MtsInstalledManifest `
        -AppPath $backupPath `
        -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
        -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
    if ($backupRelease.ManifestSha256 -ne $previousState.current.manifestSha256) {
        throw "Recovery backup does not reconcile with previous install state."
    }

    if (Test-Path -LiteralPath $paths.App) {
        $failedCurrentPath = Join-Path $transactionRoot "discarded-uncommitted-current"
        Move-Item -LiteralPath $paths.App -Destination $failedCurrentPath
    }
    Move-Item -LiteralPath $backupPath -Destination $paths.App
    Write-MtsRecordEnvelope -Path $paths.State -RecordContract $script:StateContract -Record $previousState

    $stateReadBack = Read-MtsInstallState -InstallRoot $InstallRoot -DataRoot $DataRoot -ExpectedAppId $ExpectedAppId
    $releaseReadBack = Read-MtsInstalledManifest `
        -AppPath $paths.App `
        -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
        -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
    if ($stateReadBack.current.manifestSha256 -ne $releaseReadBack.ManifestSha256) {
        throw "Recovered state and payload do not reconcile."
    }

    Remove-MtsTransactionArtifacts -InstallRoot $InstallRoot -TransactionRoot $transactionRoot -JournalPath $paths.Journal
    return [pscustomobject][ordered]@{
        ok = $true
        action = "Recover"
        dryRun = $false
        status = "previous-release-restored"
        version = [string]$stateReadBack.current.version
        userDataPolicy = "preserved"
    }
}

function Invoke-MtsUninstall {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallRoot,

        [Parameter(Mandatory = $true)]
        [string]$DataRoot,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedAppId,

        [string]$ExpectedPublisherThumbprint,

        [switch]$AllowUnsignedDevelopment,

        [switch]$DryRun
    )

    $paths = Get-MtsControlPaths $InstallRoot
    if (Test-Path -LiteralPath $paths.Journal) {
        throw "An unfinished transaction exists. Run Recover before uninstall."
    }

    $state = Read-MtsInstallState -InstallRoot $InstallRoot -DataRoot $DataRoot -ExpectedAppId $ExpectedAppId
    $release = Read-MtsInstalledManifest `
        -AppPath $paths.App `
        -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
        -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
    if ($state.current.manifestSha256 -ne $release.ManifestSha256) {
        throw "Refusing uninstall because current state and payload do not reconcile."
    }

    if ($DryRun) {
        return [pscustomobject][ordered]@{
            ok = $true
            action = "Uninstall"
            dryRun = $true
            version = [string]$state.current.version
            remove = @($paths.App, $paths.Backups, $paths.Control)
            preserve = @($DataRoot)
            userDataPolicy = "preserve"
        }
    }

    foreach ($managedPath in @($paths.App, $paths.Backups, $paths.Control)) {
        if (-not (Test-MtsPathWithin -Candidate $managedPath -Root $InstallRoot)) {
            throw "Refusing to remove managed path outside InstallRoot: $managedPath"
        }
        if (Test-Path -LiteralPath $managedPath) {
            Remove-Item -LiteralPath $managedPath -Recurse -Force
        }
    }

    if (
        (Test-Path -LiteralPath $paths.App) -or
        (Test-Path -LiteralPath $paths.State) -or
        -not (Test-Path -LiteralPath $DataRoot -PathType Container)
    ) {
        throw "Uninstall post-condition failed. Application state was not fully removed or user data was not preserved."
    }

    return [pscustomobject][ordered]@{
        ok = $true
        action = "Uninstall"
        dryRun = $false
        status = "application-removed-user-data-preserved"
        dataRoot = $DataRoot
        userDataPolicy = "preserved"
    }
}

function Invoke-MtsStatus {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallRoot,

        [Parameter(Mandatory = $true)]
        [string]$DataRoot,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedAppId,

        [string]$ExpectedPublisherThumbprint,

        [switch]$AllowUnsignedDevelopment
    )

    $paths = Get-MtsControlPaths $InstallRoot
    if (-not (Test-Path -LiteralPath $paths.State -PathType Leaf)) {
        return [pscustomobject][ordered]@{
            ok = $true
            action = "Status"
            installed = $false
            recoveryRequired = (Test-Path -LiteralPath $paths.Journal -PathType Leaf)
            installRoot = $InstallRoot
            dataRoot = $DataRoot
        }
    }

    $state = Read-MtsInstallState -InstallRoot $InstallRoot -DataRoot $DataRoot -ExpectedAppId $ExpectedAppId
    $release = Read-MtsInstalledManifest `
        -AppPath $paths.App `
        -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
        -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
    if ($state.current.manifestSha256 -ne $release.ManifestSha256) {
        throw "Installed state and payload do not reconcile."
    }

    return [pscustomobject][ordered]@{
        ok = $true
        action = "Status"
        installed = $true
        verified = $true
        recoveryRequired = (Test-Path -LiteralPath $paths.Journal -PathType Leaf)
        version = [string]$state.current.version
        manifestSha256 = [string]$state.current.manifestSha256
        rollbackAvailable = ($null -ne $state.rollback)
        dataSchemaVersion = [int]$state.current.dataSchemaVersion
        installRoot = $InstallRoot
        dataRoot = $DataRoot
    }
}

function New-MtsReleaseBundle {
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

    ConvertTo-MtsSemVer $Version | Out-Null
    $entrypointPath = Assert-MtsRelativePayloadPath $EntryPoint
    $payloadRoot = Get-MtsCanonicalFullPath $PayloadDirectory
    $outputRoot = Assert-MtsSafeManagedRoot -Path $OutputDirectory -Label "OutputDirectory"
    if (-not (Test-Path -LiteralPath $payloadRoot -PathType Container)) {
        throw "Payload directory does not exist: $payloadRoot"
    }
    if (
        (Test-MtsPathWithin -Candidate $outputRoot -Root $payloadRoot -AllowEqual) -or
        (Test-MtsPathWithin -Candidate $payloadRoot -Root $outputRoot -AllowEqual)
    ) {
        throw "OutputDirectory and PayloadDirectory must not overlap."
    }
    if ($SourceDateEpoch -lt 0) {
        throw "SourceDateEpoch must be non-negative."
    }
    if ($Channel -notmatch "^[a-z][a-z0-9-]{0,31}$") {
        throw "Channel is invalid: $Channel"
    }

    $ledger = @(Get-MtsPayloadLedger $payloadRoot)
    if ($ledger.path -notcontains $entrypointPath) {
        throw "Entrypoint is not present in the payload: $entrypointPath"
    }

    $installerLedger = @()
    $installerRoot = $null
    if (-not [string]::IsNullOrWhiteSpace($NativeInstallerDirectory)) {
        $installerRoot = Get-MtsCanonicalFullPath $NativeInstallerDirectory
        $installerLedger = @(Get-MtsPayloadLedger $installerRoot)
    }

    if ([string]::IsNullOrWhiteSpace($PublisherThumbprint) -and -not $AllowUnsignedDevelopment) {
        throw "Production release creation requires -PublisherThumbprint. Use -AllowUnsignedDevelopment only for local fixtures."
    }
    if ($AllowUnsignedDevelopment -and -not [string]::IsNullOrWhiteSpace($PublisherThumbprint)) {
        throw "-AllowUnsignedDevelopment cannot be combined with -PublisherThumbprint."
    }

    $signedFiles = @()
    $trust = if ($AllowUnsignedDevelopment) {
        [pscustomobject][ordered]@{
            mode = "development-unsigned"
            publisherThumbprint = $null
            signedFiles = @()
            manifestSignature = $null
        }
    }
    else {
        $normalizedThumbprint = ConvertTo-MtsPublisherThumbprint $PublisherThumbprint

        foreach ($file in $ledger) {
            if ([System.IO.Path]::GetExtension($file.path) -match "^(?i:\.exe|\.dll)$") {
                $signedFiles += "payload/$($file.path)"
            }
        }
        foreach ($file in $installerLedger) {
            if ([System.IO.Path]::GetExtension($file.path) -match "^(?i:\.exe|\.msi|\.dll)$") {
                $signedFiles += "installers/$($file.path)"
            }
        }
        if ($signedFiles -notcontains ("payload/" + $entrypointPath)) {
            throw "Production entrypoint must be a signed PE file."
        }

        [pscustomobject][ordered]@{
            mode = "authenticode"
            publisherThumbprint = $normalizedThumbprint
            signedFiles = @($signedFiles | Sort-Object)
            manifestSignature = [pscustomobject][ordered]@{
                path = $script:ReleaseManifestSignatureName
                format = "cms-detached"
                digestAlgorithm = "sha256"
            }
        }
    }

    $releaseId = "{0}-{1}-{2}-{3}" -f (
        ($AppId -replace "[^A-Za-z0-9.-]", "-"),
        $Version,
        $Architecture,
        $Channel
    )
    $manifest = [pscustomobject][ordered]@{
        schemaVersion = $script:ReleaseSchemaVersion
        contract = $script:ReleaseContract
        appId = $AppId
        productName = $ProductName
        releaseId = $releaseId
        version = $Version
        architecture = $Architecture
        channel = $Channel
        sourceDateEpoch = $SourceDateEpoch
        entrypoint = $entrypointPath
        build = [pscustomobject][ordered]@{
            targetTriple = $TargetTriple
            gitCommit = $GitCommit
        }
        compatibility = [pscustomobject][ordered]@{
            minInstalledVersion = if ([string]::IsNullOrWhiteSpace($MinInstalledVersion)) { $null } else { $MinInstalledVersion }
            maxInstalledVersion = if ([string]::IsNullOrWhiteSpace($MaxInstalledVersion)) { $null } else { $MaxInstalledVersion }
            dataSchema = [pscustomobject][ordered]@{
                readableMin = $DataSchemaReadableMin
                readableMax = $DataSchemaReadableMax
                writeVersion = $DataSchemaWriteVersion
            }
        }
        trust = $trust
        payload = [pscustomobject][ordered]@{
            root = "payload"
            fileCount = $ledger.Count
            totalBytes = [int64](($ledger | Measure-Object -Property size -Sum).Sum)
            files = $ledger
        }
        nativeInstallers = $installerLedger
    }
    Assert-MtsManifestContract $manifest | Out-Null

    if ($DryRun) {
        return [pscustomobject][ordered]@{
            ok = $true
            action = "CreateRelease"
            dryRun = $true
            outputDirectory = $outputRoot
            version = $Version
            trustMode = $trust.mode
            payloadFileCount = $ledger.Count
            nativeInstallerCount = $installerLedger.Count
        }
    }

    if (Test-Path -LiteralPath $outputRoot) {
        if (-not $Force) {
            throw "OutputDirectory already exists. Use -Force only when replacing a local, unpublished bundle."
        }
        Test-MtsReleaseBundle `
            -ReleaseDirectory $outputRoot `
            -ExpectedAppId $AppId `
            -ExpectedPublisherThumbprint $PublisherThumbprint `
            -AllowUnsignedDevelopment:$AllowUnsignedDevelopment | Out-Null
        Remove-Item -LiteralPath $outputRoot -Recurse -Force
    }

    $outputParent = Split-Path -Parent $outputRoot
    [System.IO.Directory]::CreateDirectory($outputParent) | Out-Null
    $stageRoot = Join-Path $outputParent (".mts-release-stage-" + [guid]::NewGuid().ToString("N"))
    try {
        $stagePayload = Join-Path $stageRoot "payload"
        [System.IO.Directory]::CreateDirectory($stagePayload) | Out-Null
        foreach ($file in $ledger) {
            $source = Join-Path $payloadRoot ($file.path.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
            $destination = Join-Path $stagePayload ($file.path.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
            [System.IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
            Copy-Item -LiteralPath $source -Destination $destination
        }

        if ($installerLedger.Count -gt 0) {
            $stageInstallers = Join-Path $stageRoot "installers"
            [System.IO.Directory]::CreateDirectory($stageInstallers) | Out-Null
            foreach ($file in $installerLedger) {
                $source = Join-Path $installerRoot ($file.path.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
                $destination = Join-Path $stageInstallers ($file.path.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
                [System.IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
                Copy-Item -LiteralPath $source -Destination $destination
            }
        }

        $manifestContent = (ConvertTo-MtsJson -Value $manifest) + "`n"
        $manifestPath = Join-Path $stageRoot $script:ReleaseManifestName
        Write-MtsUtf8File -Path $manifestPath -Content $manifestContent
        Write-MtsUtf8File `
            -Path (Join-Path $stageRoot $script:ReleaseManifestChecksumName) `
            -Content ((Get-MtsFileSha256 $manifestPath) + "`n")
        if (-not $AllowUnsignedDevelopment) {
            Write-MtsDetachedManifestSignature `
                -ManifestPath $manifestPath `
                -SignaturePath (Join-Path $stageRoot $script:ReleaseManifestSignatureName) `
                -PublisherThumbprint $PublisherThumbprint
        }

        $verified = Test-MtsReleaseBundle `
            -ReleaseDirectory $stageRoot `
            -ExpectedAppId $AppId `
            -ExpectedPublisherThumbprint $PublisherThumbprint `
            -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
        Move-Item -LiteralPath $stageRoot -Destination $outputRoot

        return [pscustomobject][ordered]@{
            ok = $true
            action = "CreateRelease"
            dryRun = $false
            outputDirectory = $outputRoot
            version = $Version
            manifestSha256 = $verified.ManifestSha256
            trustMode = $trust.mode
            payloadFileCount = $ledger.Count
            nativeInstallerCount = $installerLedger.Count
        }
    }
    finally {
        if (Test-Path -LiteralPath $stageRoot) {
            Remove-Item -LiteralPath $stageRoot -Recurse -Force
        }
    }
}

function Invoke-MtsLifecycle {
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

    if ($Action -eq "Validate") {
        if ([string]::IsNullOrWhiteSpace($ReleaseDirectory)) {
            throw "Validate requires -ReleaseDirectory."
        }
        $release = Test-MtsReleaseBundle `
            -ReleaseDirectory $ReleaseDirectory `
            -ExpectedAppId $ExpectedAppId `
            -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
            -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
        return [pscustomobject][ordered]@{
            ok = $true
            action = "Validate"
            releaseDirectory = $release.ReleaseRoot
            version = [string]$release.Manifest.version
            manifestSha256 = $release.ManifestSha256
            trustMode = [string]$release.Manifest.trust.mode
        }
    }

    if ([string]::IsNullOrWhiteSpace($InstallRoot) -or [string]::IsNullOrWhiteSpace($DataRoot)) {
        throw "$Action requires both -InstallRoot and -DataRoot."
    }
    $boundaries = Assert-MtsInstallBoundaries -InstallRoot $InstallRoot -DataRoot $DataRoot
    $installPath = $boundaries.InstallRoot
    $dataPath = $boundaries.DataRoot

    switch ($Action) {
        "Install" {
            if ([string]::IsNullOrWhiteSpace($ReleaseDirectory)) {
                throw "Install requires -ReleaseDirectory."
            }
            $release = Test-MtsReleaseBundle `
                -ReleaseDirectory $ReleaseDirectory `
                -ExpectedAppId $ExpectedAppId `
                -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
                -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
            return Invoke-MtsDeploy `
                -Action "Install" `
                -InstallRoot $installPath `
                -DataRoot $dataPath `
                -Release $release `
                -ExpectedAppId $ExpectedAppId `
                -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
                -AllowUnsignedDevelopment:$AllowUnsignedDevelopment `
                -AllowDowngrade:$AllowDowngrade `
                -DryRun:$DryRun
        }
        "Upgrade" {
            if ([string]::IsNullOrWhiteSpace($ReleaseDirectory)) {
                throw "Upgrade requires -ReleaseDirectory."
            }
            $release = Test-MtsReleaseBundle `
                -ReleaseDirectory $ReleaseDirectory `
                -ExpectedAppId $ExpectedAppId `
                -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
                -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
            return Invoke-MtsDeploy `
                -Action "Upgrade" `
                -InstallRoot $installPath `
                -DataRoot $dataPath `
                -Release $release `
                -ExpectedAppId $ExpectedAppId `
                -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
                -AllowUnsignedDevelopment:$AllowUnsignedDevelopment `
                -AllowDowngrade:$AllowDowngrade `
                -DryRun:$DryRun
        }
        "Rollback" {
            return Invoke-MtsRollback `
                -InstallRoot $installPath `
                -DataRoot $dataPath `
                -ExpectedAppId $ExpectedAppId `
                -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
                -AllowUnsignedDevelopment:$AllowUnsignedDevelopment `
                -AllowDowngrade:$AllowDowngrade `
                -DryRun:$DryRun
        }
        "Recover" {
            return Invoke-MtsRecover `
                -InstallRoot $installPath `
                -DataRoot $dataPath `
                -ExpectedAppId $ExpectedAppId `
                -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
                -AllowUnsignedDevelopment:$AllowUnsignedDevelopment `
                -DryRun:$DryRun
        }
        "Uninstall" {
            return Invoke-MtsUninstall `
                -InstallRoot $installPath `
                -DataRoot $dataPath `
                -ExpectedAppId $ExpectedAppId `
                -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
                -AllowUnsignedDevelopment:$AllowUnsignedDevelopment `
                -DryRun:$DryRun
        }
        "Status" {
            return Invoke-MtsStatus `
                -InstallRoot $installPath `
                -DataRoot $dataPath `
                -ExpectedAppId $ExpectedAppId `
                -ExpectedPublisherThumbprint $ExpectedPublisherThumbprint `
                -AllowUnsignedDevelopment:$AllowUnsignedDevelopment
        }
    }
}

Export-ModuleMember -Function @(
    "Compare-MtsSemVer",
    "Get-MtsCanonicalFullPath",
    "Get-MtsFileSha256",
    "Invoke-MtsLifecycle",
    "New-MtsReleaseBundle",
    "Test-MtsReleaseBundle"
)
