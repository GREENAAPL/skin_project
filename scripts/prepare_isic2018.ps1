[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [int]$Seed = 42,
    [int]$DevFold = 0
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

Push-Location $ProjectRoot
try {
    $python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        $python = "python"
    }
    & $python -m skin_project.isic2018 `
        --root "data/isic2018" `
        --n-splits 5 `
        --dev-fold $DevFold `
        --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "ISIC 2018 preparation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
