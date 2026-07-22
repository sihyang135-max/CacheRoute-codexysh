[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Server,

    [Parameter(Mandatory = $true)]
    [string]$RemoteArchive,

    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = 'Stop'
$destinationPath = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
$archiveName = [IO.Path]::GetFileName($RemoteArchive)
$localArchive = Join-Path $destinationPath $archiveName
$localChecksum = "$localArchive.sha256"

& scp "${Server}:$RemoteArchive" $localArchive
if ($LASTEXITCODE -ne 0) { throw "scp failed for $RemoteArchive" }
& scp "${Server}:$RemoteArchive.sha256" $localChecksum
if ($LASTEXITCODE -ne 0) { throw "scp failed for $RemoteArchive.sha256" }

$expected = ((Get-Content -LiteralPath $localChecksum -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
$actual = (Get-FileHash -LiteralPath $localArchive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) {
    throw "SHA-256 mismatch: expected=$expected actual=$actual"
}

Write-Output "[OK] $localArchive"
Write-Output "sha256=$actual"
