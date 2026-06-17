# TrueScan — Self-Signed SSL Certificate Generator (Windows PowerShell)
# =======================================================================
# Generates a self-signed TLS cert for local HTTPS development.
# Run in an elevated (Administrator) PowerShell.
#
# Usage:  .\scripts\generate_ssl.ps1
# Or with custom domain:  .\scripts\generate_ssl.ps1 -Domain my.local

param(
    [string]$Domain = "truescan.local",
    [int]$ValidDays = 3650
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " TrueScan — Self-Signed SSL Certificate Generator"     -ForegroundColor Cyan
Write-Host " Domain  : $Domain"
Write-Host " Valid   : $ValidDays days"
Write-Host "======================================================" -ForegroundColor Cyan

# ── Output directory ──────────────────────────────────────────────────────────
$OutDir = Join-Path $PSScriptRoot "..\nginx\ssl"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$CertPath = Join-Path $OutDir "truescan.crt"
$KeyPath  = Join-Path $OutDir "truescan.key"
$PfxPath  = Join-Path $OutDir "truescan.pfx"

# ── Generate self-signed cert via PowerShell ──────────────────────────────────
Write-Host "`nGenerating certificate..." -ForegroundColor Yellow

$cert = New-SelfSignedCertificate `
    -Subject          "CN=$Domain, O=TrueScan, C=US" `
    -DnsName          @($Domain, "localhost") `
    -IPAddress        "127.0.0.1" `
    -KeyAlgorithm     RSA `
    -KeyLength        4096 `
    -HashAlgorithm    SHA256 `
    -NotAfter         (Get-Date).AddDays($ValidDays) `
    -KeyExportPolicy  Exportable `
    -CertStoreLocation "Cert:\LocalMachine\My"

Write-Host "✅ Certificate created in Windows cert store: $($cert.Thumbprint)"

# ── Export PFX (for openssl conversion) ───────────────────────────────────────
$PfxPassword = ConvertTo-SecureString -String "truescan-dev" -Force -AsPlainText
Export-PfxCertificate -Cert $cert -FilePath $PfxPath -Password $PfxPassword | Out-Null
Write-Host "✅ PFX exported: $PfxPath"

# ── Check if openssl is available for PEM extraction ─────────────────────────
if (Get-Command openssl -ErrorAction SilentlyContinue) {
    Write-Host "`nExtracting PEM files with openssl..."
    openssl pkcs12 -in $PfxPath -nokeys -out $CertPath -passin "pass:truescan-dev"
    openssl pkcs12 -in $PfxPath -nocerts -nodes -out $KeyPath -passin "pass:truescan-dev"
    Write-Host "✅ CRT: $CertPath"
    Write-Host "✅ KEY: $KeyPath"
} else {
    Write-Host "`n⚠️  openssl not found in PATH." -ForegroundColor Yellow
    Write-Host "   Install Git for Windows (includes openssl) or use WSL:"
    Write-Host "   winget install Git.Git"
    Write-Host "   Then re-run this script."
    Write-Host ""
    Write-Host "   Alternatively, copy $PfxPath to WSL and run:"
    Write-Host "   openssl pkcs12 -in truescan.pfx -nokeys -out truescan.crt -passin pass:truescan-dev"
    Write-Host "   openssl pkcs12 -in truescan.pfx -nocerts -nodes -out truescan.key -passin pass:truescan-dev"
}

# ── Trust the cert in Windows ─────────────────────────────────────────────────
Write-Host "`nTrusting certificate in Windows Root CA store..."
try {
    $rootStore = New-Object System.Security.Cryptography.X509Certificates.X509Store("Root", "LocalMachine")
    $rootStore.Open("ReadWrite")
    $rootStore.Add($cert)
    $rootStore.Close()
    Write-Host "✅ Certificate trusted by Windows (Chrome/Edge will accept it)"
} catch {
    Write-Host "⚠️  Could not auto-trust (run as Administrator). Trust manually:" -ForegroundColor Yellow
    Write-Host "   certutil -addstore Root `"$PfxPath`""
}

# ── /etc/hosts reminder ───────────────────────────────────────────────────────
Write-Host "`n------------------------------------------------------"
Write-Host "Add to C:\Windows\System32\drivers\etc\hosts:"
Write-Host "  127.0.0.1  $Domain"
Write-Host ""
Write-Host "Then copy nginx\truescan.conf to your nginx sites-available"
Write-Host "and update ssl_certificate paths to point to the generated files."
Write-Host "------------------------------------------------------" -ForegroundColor Cyan
