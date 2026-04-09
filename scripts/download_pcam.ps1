# Download real PatchCamelyon dataset from Zenodo
# Total size: ~7GB compressed, ~7GB extracted

$ErrorActionPreference = "Stop"

# Create data directory
New-Item -ItemType Directory -Force -Path "data/pcam_real" | Out-Null
Set-Location "data/pcam_real"

Write-Host "Downloading PatchCamelyon dataset from Zenodo..." -ForegroundColor Green
Write-Host "This will download ~7GB of data. Please be patient." -ForegroundColor Yellow
Write-Host ""

$files = @(
    @{Name="training images"; Url="https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz"},
    @{Name="training labels"; Url="https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz"},
    @{Name="validation images"; Url="https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz"},
    @{Name="validation labels"; Url="https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz"},
    @{Name="test images"; Url="https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz"},
    @{Name="test labels"; Url="https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz"}
)

$i = 1
foreach ($file in $files) {
    $filename = Split-Path $file.Url -Leaf
    Write-Host "[$i/6] Downloading $($file.Name)..." -ForegroundColor Cyan
    
    if (Test-Path $filename) {
        Write-Host "  File already exists, skipping..." -ForegroundColor Yellow
    } else {
        Invoke-WebRequest -Uri $file.Url -OutFile $filename -UseBasicParsing
    }
    $i++
}

Write-Host ""
Write-Host "Download complete! Extracting files..." -ForegroundColor Green
Write-Host ""

# Extract all .gz files
Get-ChildItem -Filter "*.gz" | ForEach-Object {
    $outFile = $_.Name -replace '\.gz$', ''
    Write-Host "Extracting $($_.Name)..." -ForegroundColor Cyan
    
    # Use .NET to decompress
    $inStream = New-Object System.IO.FileStream($_.FullName, [System.IO.FileMode]::Open)
    $gzipStream = New-Object System.IO.Compression.GZipStream($inStream, [System.IO.Compression.CompressionMode]::Decompress)
    $outStream = New-Object System.IO.FileStream($outFile, [System.IO.FileMode]::Create)
    
    $gzipStream.CopyTo($outStream)
    
    $outStream.Close()
    $gzipStream.Close()
    $inStream.Close()
    
    # Remove .gz file after extraction
    Remove-Item $_.FullName
}

Write-Host ""
Write-Host "Extraction complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Dataset statistics:" -ForegroundColor Cyan
Write-Host "  Train: 262,144 images"
Write-Host "  Valid: 32,768 images"
Write-Host "  Test:  32,768 images"
Write-Host "  Total: 327,680 images"
Write-Host ""
Write-Host "Files are located in: data/pcam_real/" -ForegroundColor Green
Write-Host ""
Write-Host "Next step: Run training with:" -ForegroundColor Yellow
Write-Host "  python experiments/train_pcam.py --config experiments/configs/pcam_rtx4070_laptop.yaml"
