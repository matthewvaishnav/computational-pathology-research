<#
.SYNOPSIS
Scaffold a standalone PathoAlign external canine SCC validation repository.

.DESCRIPTION
This script copies only reproducible code, documentation, manifests, and compact
result tables from the main computational-pathology-research repository. It does
not copy raw TIFFs, extracted JPEG patches, NPZ feature archives, model
checkpoints, or generated run directories that are too large or regenerable.

Run from any PowerShell session after the external canine five-fold results have
been generated in the main repository.
#>

param(
    [string]$SourceRoot = "C:\Users\matth\computational-pathology-research",
    [string]$DestinationRoot = "C:\Users\matth\pathoalign-external-caninescc",
    [switch]$InitializeGit
)

$ErrorActionPreference = "Stop"

function Copy-RequiredFile {
    param(
        [string]$RelativePath,
        [string]$TargetRelativePath = $RelativePath
    )
    $source = Join-Path $SourceRoot $RelativePath
    $target = Join-Path $DestinationRoot $TargetRelativePath
    if (!(Test-Path $source)) {
        throw "Required file not found: $source"
    }
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
    Copy-Item $source $target -Force
}

function Copy-OptionalFile {
    param(
        [string]$RelativePath,
        [string]$TargetRelativePath = $RelativePath
    )
    $source = Join-Path $SourceRoot $RelativePath
    $target = Join-Path $DestinationRoot $TargetRelativePath
    if (Test-Path $source) {
        New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
        Copy-Item $source $target -Force
    } else {
        Write-Warning "Skipping optional missing file: $source"
    }
}

function Copy-RequiredDirectory {
    param(
        [string]$RelativePath,
        [string]$TargetRelativePath = $RelativePath
    )
    $source = Join-Path $SourceRoot $RelativePath
    $target = Join-Path $DestinationRoot $TargetRelativePath
    if (!(Test-Path $source)) {
        throw "Required directory not found: $source"
    }
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
    Copy-Item $source $target -Recurse -Force
}

function Copy-OptionalDirectory {
    param(
        [string]$RelativePath,
        [string]$TargetRelativePath = $RelativePath
    )
    $source = Join-Path $SourceRoot $RelativePath
    $target = Join-Path $DestinationRoot $TargetRelativePath
    if (Test-Path $source) {
        New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
        Copy-Item $source $target -Recurse -Force
    } else {
        Write-Warning "Skipping optional missing directory: $source"
    }
}

function New-TextFile {
    param(
        [string]$RelativePath,
        [string]$Content
    )
    $target = Join-Path $DestinationRoot $RelativePath
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
    Set-Content -Path $target -Value $Content -Encoding UTF8
}

if (!(Test-Path $SourceRoot)) {
    throw "SourceRoot does not exist: $SourceRoot"
}

New-Item -ItemType Directory -Force $DestinationRoot | Out-Null

# Core reproducible code.
Copy-RequiredDirectory "scripts\external_multiscanner" "scripts\external_multiscanner"
Copy-RequiredDirectory "experiments\external_multiscanner" "experiments\external_multiscanner"
Copy-RequiredDirectory "src\models" "src\models"
Copy-RequiredFile "experiments\scorpion\run_pathoalign_projection.py" "experiments\scorpion\run_pathoalign_projection.py"
Copy-RequiredFile "scripts\scorpion\extract_scorpion_vit_features.py" "scripts\scorpion\extract_scorpion_vit_features.py"
Copy-RequiredFile "scripts\scorpion\extract_scorpion_resnet50_features.py" "scripts\scorpion\extract_scorpion_resnet50_features.py"

# Protocol and frozen result notes.
Copy-RequiredFile "docs\research\pathoalign-external-caninescc-results.md" "docs\results.md"
Copy-OptionalFile "docs\research\pathoalign-external-multiscanner-caninescc-protocol.md" "docs\protocol.md"

# Small manifests and compact result artifacts.
Copy-RequiredDirectory "data\external_multiscanner_caninescc\patch_manifests" "data\external_multiscanner_caninescc\patch_manifests"
Copy-OptionalFile "data\external_multiscanner_caninescc\manifest.csv" "data\external_multiscanner_caninescc\manifest.csv"
Copy-RequiredFile "results\external_multiscanner_caninescc\geometry_qualified\geometry_qualified_summary.json" "results\geometry_qualified_summary.json"
Copy-RequiredFile "data\external_multiscanner_caninescc\patch_manifests\patch_extraction_summary.json" "results\patch_extraction_summary.json"
Copy-RequiredDirectory "results\external_multiscanner_caninescc\pathoalign_dinov2_crossfold_analysis" "results\pathoalign_dinov2_crossfold_analysis"
Copy-OptionalDirectory "results\external_multiscanner_caninescc\frozen_dinov2_base_fold0_val" "results\frozen_encoder_baselines\dinov2_base_fold0_val"
Copy-OptionalDirectory "results\external_multiscanner_caninescc\frozen_resnet50_fold0_val" "results\frozen_encoder_baselines\resnet50_fold0_val"
Copy-OptionalDirectory "results\external_multiscanner_caninescc\frozen_phikon_fold0_val" "results\frozen_encoder_baselines\phikon_fold0_val"

# Create top-level repository files.
$readme = @"
# PathoAlign External Canine SCC Validation

Standalone reproducibility package for the external paired-scanner validation of
PathoAlign on the Multi-Scanner Canine Cutaneous Squamous Cell Carcinoma
histopathology dataset.

## Headline result

PathoAlign reduced scanner identifiability on a locked five-fold external test
while preserving same-region retrieval.

| Metric | Paired reference | PathoAlign dep20 | Difference |
|---|---:|---:|---:|
| Scanner probe accuracy | 0.752868 | 0.361408 | -0.380347 sample-blocked contrast |
| Pair cosine average | 0.696022 | 0.729961 | +0.033256 sample-blocked contrast |
| Pair cosine worst | 0.627300 | 0.656736 | +0.033104 sample-blocked contrast |
| Retrieval top-1 average | 0.930637 | 0.933392 | +0.002326 sample-blocked contrast |
| Retrieval top-1 worst | 0.881242 | 0.884431 | +0.001731 sample-blocked contrast |

All predefined success criteria passed over 44 biological sample blocks.

## What is included

- Dataset inspection and annotation correspondence scripts.
- Geometry qualification and P1000 orientation-normalization scripts.
- Patch extraction manifest generation.
- Frozen encoder analysis scripts.
- Locked PathoAlign validation and five-fold test runners.
- Compact result tables and JSON summaries.

## What is not included

The repository intentionally excludes raw TIFFs, extracted JPEG patches, NPZ
feature archives, model checkpoints, and full run directories. These artifacts
are large or regenerable from the public dataset and scripts.

## Reproduction outline

1. Download the public Multi-Scanner Canine SCC dataset.
2. Build or verify the geometry-qualified patch manifests.
3. Extract orientation-normalized patches locally.
4. Extract DINOv2 frozen features.
5. Run `experiments/external_multiscanner/run_canine_pathoalign_crossfold.py`.
6. Run `scripts/external_multiscanner/analyze_canine_pathoalign_crossfold.py`.

See `docs/results.md` for the frozen result statement and claim boundary.

## Claim boundary

This is a representation-identifiability and paired-acquisition validation
study. It is research only and is not clinical, diagnostic, or patient-care
software.
"@
New-TextFile "README.md" $readme

$gitignore = @"
# Raw data and generated image/model artifacts
*.tif
*.tiff
*.svs
*.ndpi
*.mrxs
*.jpg
*.jpeg
*.png
*.npz
*.pt
*.pth
*.ckpt

# Regenerable local outputs
data/external_multiscanner_caninescc/patches/
results/**/runs/
results/**/fold_*/

# Python / OS
__pycache__/
*.pyc
.venv/
.env
.DS_Store
Thumbs.db
"@
New-TextFile ".gitignore" $gitignore

$requirements = @"
numpy
pandas
pillow
torch
torchvision
tifffile
zarr
scikit-learn
transformers
"@
New-TextFile "requirements.txt" $requirements

$citation = @"
cff-version: 1.2.0
title: PathoAlign External Canine SCC Validation
message: If you use this reproducibility package, please cite the repository and the public Multi-Scanner Canine SCC dataset.
authors:
  - family-names: Chorne
    given-names: Larry
repository-code: https://github.com/matthewvaishnav/pathoalign-external-caninescc
license: MIT
"@
New-TextFile "CITATION.cff" $citation

$license = @"
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"@
New-TextFile "LICENSE" $license

# Remove accidental heavy artifacts if the user had local generated files under copied paths.
Get-ChildItem $DestinationRoot -Recurse -Include *.npz,*.pt,*.pth,*.ckpt,*.tif,*.tiff,*.jpg,*.jpeg,*.png -File |
    ForEach-Object {
        Write-Warning "Removing generated/heavy artifact from scaffold: $($_.FullName)"
        Remove-Item $_.FullName -Force
    }

if ($InitializeGit) {
    Push-Location $DestinationRoot
    try {
        if (!(Test-Path ".git")) {
            git init | Out-Host
        }
        git add . | Out-Host
        git status --short | Out-Host
    } finally {
        Pop-Location
    }
}

Write-Host "Standalone PathoAlign canine SCC repo scaffold complete: $DestinationRoot"
Write-Host "Next: inspect the directory, then create the GitHub repository pathoalign-external-caninescc and push it."
