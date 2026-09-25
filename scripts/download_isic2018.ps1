[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [ValidateRange(1, 16)]
    [int]$Connections = 8
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

$datasetRoot = Join-Path $ProjectRoot "data\isic2018"
$downloadRoot = Join-Path $datasetRoot "downloads"
$rawRoot = Join-Path $datasetRoot "raw"
$imageArchive = Join-Path $downloadRoot "ISIC2018_Task3_Training_Input.zip"
$groundTruthArchive = Join-Path $downloadRoot "ISIC2018_Task3_Training_GroundTruth.zip"
$groupingsFile = Join-Path $rawRoot "ISIC2018_Task3_Training_LesionGroupings.csv"

$imageUrl = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_Input.zip"
$groundTruthUrl = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_GroundTruth.zip"
$groupingsUrl = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_LesionGroupings.csv"

$expectedImageArchiveBytes = 2771732744L
$expectedGroundTruthMd5 = "8302427e4ce0c107559531b9f444abe9"
$expectedGroupingsMd5 = "9517b6aa6902d6c27e234bac8279a64a"

New-Item -ItemType Directory -Force -Path $downloadRoot, $rawRoot | Out-Null

function Invoke-SmallDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$ExpectedMd5
    )

    if (Test-Path -LiteralPath $Destination) {
        $actualMd5 = (Get-FileHash -LiteralPath $Destination -Algorithm MD5).Hash.ToLowerInvariant()
        if ($actualMd5 -eq $ExpectedMd5) {
            Write-Host "Verified existing file: $Destination"
            return
        }
    }

    $temporaryPath = "$Destination.download"
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Force
    }

    & curl.exe -sS -L --fail --retry 8 --retry-delay 2 --retry-all-errors `
        --connect-timeout 30 --output $temporaryPath $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Url"
    }

    $actualMd5 = (Get-FileHash -LiteralPath $temporaryPath -Algorithm MD5).Hash.ToLowerInvariant()
    if ($actualMd5 -ne $ExpectedMd5) {
        throw "MD5 mismatch for $temporaryPath (expected $ExpectedMd5, got $actualMd5)"
    }

    Move-Item -LiteralPath $temporaryPath -Destination $Destination -Force
    Write-Host "Downloaded and verified: $Destination"
}

function Invoke-ParallelRangeDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][long]$ExpectedBytes,
        [Parameter(Mandatory = $true)][int]$ConnectionCount
    )

    if ((Test-Path -LiteralPath $Destination) -and
        ((Get-Item -LiteralPath $Destination).Length -eq $ExpectedBytes)) {
        Write-Host "Verified existing image archive: $Destination"
        return
    }

    if (Test-Path -LiteralPath $Destination) {
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        Move-Item -LiteralPath $Destination -Destination "$Destination.incomplete-$timestamp"
    }

    $segmentBytes = [long][math]::Ceiling($ExpectedBytes / [double]$ConnectionCount)
    $partPaths = New-Object System.Collections.Generic.List[string]
    $jobs = New-Object System.Collections.Generic.List[object]

    for ($index = 0; $index -lt $ConnectionCount; $index++) {
        $rangeStart = [long]$index * $segmentBytes
        if ($rangeStart -ge $ExpectedBytes) {
            break
        }

        $rangeEnd = [math]::Min($ExpectedBytes - 1, $rangeStart + $segmentBytes - 1)
        $expectedPartBytes = $rangeEnd - $rangeStart + 1
        $partPath = "$Destination.part{0:D2}" -f $index
        $partPaths.Add($partPath)

        if ((Test-Path -LiteralPath $partPath) -and
            ((Get-Item -LiteralPath $partPath).Length -eq $expectedPartBytes)) {
            Write-Host ("Reusing complete segment {0}/{1}" -f ($index + 1), $ConnectionCount)
            continue
        }

        if (Test-Path -LiteralPath $partPath) {
            Remove-Item -LiteralPath $partPath -Force
        }

        $job = Start-Job -Name ("isic-part-{0:D2}" -f $index) -ScriptBlock {
            param($SourceUrl, $OutputPath, $StartByte, $EndByte)
            & curl.exe -sS -L --fail --retry 8 --retry-delay 2 --retry-all-errors `
                --connect-timeout 30 --range "$StartByte-$EndByte" `
                --output $OutputPath $SourceUrl
            if ($LASTEXITCODE -ne 0) {
                throw "curl failed with exit code $LASTEXITCODE for bytes $StartByte-$EndByte"
            }
        } -ArgumentList $Url, $partPath, $rangeStart, $rangeEnd
        $jobs.Add($job)
    }

    if ($jobs.Count -gt 0) {
        Write-Host ("Downloading {0:N2} GB with {1} parallel connections..." -f ($ExpectedBytes / 1GB), $jobs.Count)
        do {
            $pendingJobs = @($jobs | Where-Object { $_.State -in @("Running", "NotStarted") })
            if ($pendingJobs.Count -gt 0) {
                Wait-Job -Job $pendingJobs -Any -Timeout 10 | Out-Null
            }
            $finished = @($jobs | Where-Object { $_.State -eq "Completed" }).Count
            $failed = @($jobs | Where-Object { $_.State -eq "Failed" }).Count
            $downloadedBytes = 0L
            foreach ($partPath in $partPaths) {
                if (Test-Path -LiteralPath $partPath) {
                    $downloadedBytes += (Get-Item -LiteralPath $partPath).Length
                }
            }
            Write-Host ("  segments complete: {0}/{1}; downloaded: {2:N0} MB; failed: {3}" -f `
                $finished, $jobs.Count, ($downloadedBytes / 1MB), $failed)
        } while ($pendingJobs.Count -gt 0)

        $jobs | Receive-Job
        $failedJobs = @($jobs | Where-Object { $_.State -ne "Completed" })
        $jobs | Remove-Job -Force
        if ($failedJobs.Count -gt 0) {
            throw "$($failedJobs.Count) download segment(s) failed. Run this script again to retry."
        }
    }

    for ($index = 0; $index -lt $partPaths.Count; $index++) {
        $rangeStart = [long]$index * $segmentBytes
        $rangeEnd = [math]::Min($ExpectedBytes - 1, $rangeStart + $segmentBytes - 1)
        $expectedPartBytes = $rangeEnd - $rangeStart + 1
        $actualPartBytes = (Get-Item -LiteralPath $partPaths[$index]).Length
        if ($actualPartBytes -ne $expectedPartBytes) {
            throw "Segment size mismatch: $($partPaths[$index]) (expected $expectedPartBytes, got $actualPartBytes)"
        }
    }

    $assemblingPath = "$Destination.assembling"
    if (Test-Path -LiteralPath $assemblingPath) {
        Remove-Item -LiteralPath $assemblingPath -Force
    }

    $outputStream = [System.IO.File]::Open($assemblingPath, [System.IO.FileMode]::CreateNew)
    try {
        foreach ($partPath in $partPaths) {
            $inputStream = [System.IO.File]::OpenRead($partPath)
            try {
                $inputStream.CopyTo($outputStream)
            }
            finally {
                $inputStream.Dispose()
            }
        }
    }
    finally {
        $outputStream.Dispose()
    }

    $assembledBytes = (Get-Item -LiteralPath $assemblingPath).Length
    if ($assembledBytes -ne $ExpectedBytes) {
        throw "Assembled archive size mismatch (expected $ExpectedBytes, got $assembledBytes)"
    }

    Move-Item -LiteralPath $assemblingPath -Destination $Destination
    foreach ($partPath in $partPaths) {
        Remove-Item -LiteralPath $partPath -Force
    }
    Write-Host "Assembled and verified image archive: $Destination"
}

Invoke-ParallelRangeDownload `
    -Url $imageUrl `
    -Destination $imageArchive `
    -ExpectedBytes $expectedImageArchiveBytes `
    -ConnectionCount $Connections

Invoke-SmallDownload `
    -Url $groundTruthUrl `
    -Destination $groundTruthArchive `
    -ExpectedMd5 $expectedGroundTruthMd5

Invoke-SmallDownload `
    -Url $groupingsUrl `
    -Destination $groupingsFile `
    -ExpectedMd5 $expectedGroupingsMd5

$existingImages = @(Get-ChildItem -LiteralPath $rawRoot -Recurse -File -Filter "*.jpg" -ErrorAction SilentlyContinue)
if ($existingImages.Count -ne 10015) {
    Write-Host "Extracting image archive..."
    & tar.exe -xf $imageArchive -C $rawRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Image archive extraction failed."
    }
}

Write-Host "Extracting ground-truth labels..."
& tar.exe -xf $groundTruthArchive -C $rawRoot
if ($LASTEXITCODE -ne 0) {
    throw "Ground-truth archive extraction failed."
}

$images = @(Get-ChildItem -LiteralPath $rawRoot -Recurse -File -Filter "*.jpg")
$groundTruthFile = Get-ChildItem -LiteralPath $rawRoot -Recurse -File `
    -Filter "ISIC2018_Task3_Training_GroundTruth.csv" | Select-Object -First 1

if ($images.Count -ne 10015) {
    throw "Expected 10015 JPEG images, found $($images.Count)."
}
if ($null -eq $groundTruthFile) {
    throw "Ground-truth CSV was not found after extraction."
}

$groundTruthRows = @(Import-Csv -LiteralPath $groundTruthFile.FullName)
$groupingRows = @(Import-Csv -LiteralPath $groupingsFile)
if ($groundTruthRows.Count -ne 10015) {
    throw "Expected 10015 ground-truth rows, found $($groundTruthRows.Count)."
}
if ($groupingRows.Count -ne 10015) {
    throw "Expected 10015 lesion-grouping rows, found $($groupingRows.Count)."
}

Write-Host ""
Write-Host "ISIC 2018 Task 3 download is complete."
Write-Host "  dataset root: $datasetRoot"
Write-Host "  JPEG images:  $($images.Count)"
Write-Host "  label rows:    $($groundTruthRows.Count)"
Write-Host "  grouping rows: $($groupingRows.Count)"
Write-Host "  license:       CC BY-NC 4.0"
