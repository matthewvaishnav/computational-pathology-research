# PowerShell 5.1+ / PowerShell 7; metadata-only. Never runs PA-NF models.
[CmdletBinding()]
param(
    [string]$DataDir = (Join-Path $env:USERPROFILE 'Downloads\hmi'),
    [string]$RegistrationReceipt = (Join-Path $env:USERPROFILE 'Documents\PA_NF_ScanComp_Intake\hmi825_registration_v2\SCANCOMP_REGISTRATION_DIAGNOSTIC.json'),
    [string]$LegacySites = '',
    [string]$PixelGallery = '',
    [string]$OutputPath = (Join-Path $env:USERPROFILE 'Documents\PA_NF_ScanComp_Intake\hmi825_panf_001r_preflight.json'),
    [switch]$VerifyHashes
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$arguments = @(
    '-m', 'scripts.external_multiscanner.scancomp_panf_001r_preflight',
    '--data-dir', $DataDir,
    '--registration-receipt', $RegistrationReceipt,
    '--out', $OutputPath
)
if ($LegacySites) { $arguments += @('--legacy-sites', $LegacySites) }
if ($PixelGallery) { $arguments += @('--pixel-gallery', $PixelGallery) }
if ($VerifyHashes) { $arguments += '--verify-hashes' }

Push-Location $repoRoot
try {
    & python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Read-only preflight failed (exit $LASTEXITCODE). No inference was run."
    }
    Write-Host "Preflight receipt: $OutputPath"
    Write-Host 'Physical 800-um evaluation remains BLOCKED pending authenticated registered-export scale.'
}
finally {
    Pop-Location
}
