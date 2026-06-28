<#
.SYNOPSIS
Scaffold a standalone Paired-Acquisition Neural Factorization SCORPION core repository.

.DESCRIPTION
This script copies the core paired-scanner method code, SCORPION experiment
runners, analysis scripts, public result notes, compact result artifacts, and
paper source from the main computational-pathology-research repository into a
focused child-repository directory.

It intentionally excludes raw images, feature archives, checkpoints, and large
generated run directories.
#>

param(
    [string]$SourceRoot = "C:\Users\matth\computational-pathology-research",
    [string]$DestinationRoot = "C:\Users\matth\paired-acquisition-factorization-scorpion",
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
    New-Item -ItemType Directory -Force $target | Out-Null
    Get-ChildItem -LiteralPath $source -Force | Copy-Item -Destination $target -Recurse -Force
}

function Copy-OptionalDirectory {
    param(
        [string]$RelativePath,
        [string]$TargetRelativePath = $RelativePath
    )
    $source = Join-Path $SourceRoot $RelativePath
    $target = Join-Path $DestinationRoot $TargetRelativePath
    if (Test-Path $source) {
        New-Item -ItemType Directory -Force $target | Out-Null
        Get-ChildItem -LiteralPath $source -Force | Copy-Item -Destination $target -Recurse -Force
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

function Convert-LegacyPublicNames {
    $textExtensions = @(
        ".bib", ".cff", ".csv", ".gitignore", ".json", ".md", ".ps1", ".py", ".tex", ".txt", ".yml", ".yaml"
    )
    $literalTextFiles = @("LICENSE", "requirements.txt")
    $replacements = @(
        @("scorpion_pathoalign", "scorpion_paired_acquisition"),
        @("run_pathoalign", "run_paired_acquisition"),
        @("analyze_pathoalign", "analyze_paired_acquisition"),
        @("pathoalign_dep", "paired_acquisition_dep"),
        @("pathoalign_", "paired_acquisition_"),
        @("https://github.com/matthewvaishnav/computational-pathology-research", "https://github.com/matthewvaishnav/paired-acquisition-factorization-scorpion"),
        @("Repository target: \texttt{paired-acquisition-factorization-scorpion}", "\href{https://github.com/matthewvaishnav/paired-acquisition-factorization-scorpion}{repository}; \href{https://matthewvaishnav.github.io/paired-acquisition-factorization-scorpion/paired-acquisition-factorization-scorpion.pdf}{PDF}"),
        @("PATHOALIGN", "PAIRED-ACQUISITION NEURAL FACTORIZATION"),
        @("PathoAlign", "Paired-Acquisition Neural Factorization"),
        @("pathoalign", "paired_acquisition")
    )

    Get-ChildItem -LiteralPath $DestinationRoot -Recurse -Force -File |
        Where-Object {
            $_.FullName -notmatch "\\.git\\" -and
            ($textExtensions -contains $_.Extension -or $literalTextFiles -contains $_.Name)
        } |
        ForEach-Object {
            $content = Get-Content -LiteralPath $_.FullName -Raw
            $updated = $content
            foreach ($entry in $replacements) {
                $updated = $updated.Replace($entry[0], $entry[1])
            }
            if ($updated -ne $content) {
                Set-Content -LiteralPath $_.FullName -Value $updated -Encoding UTF8
            }
        }

    Get-ChildItem -LiteralPath $DestinationRoot -Recurse -Force |
        Where-Object { $_.FullName -notmatch "\\.git\\" -and $_.Name -match "(?i)pathoalign" } |
        Sort-Object { $_.FullName.Length } -Descending |
        ForEach-Object {
            $newName = $_.Name -replace "(?i)pathoalign", "paired_acquisition"
            Rename-Item -LiteralPath $_.FullName -NewName $newName
        }
}

if (!(Test-Path $SourceRoot)) {
    throw "SourceRoot does not exist: $SourceRoot"
}

New-Item -ItemType Directory -Force $DestinationRoot | Out-Null
$resolvedDestination = (Resolve-Path $DestinationRoot).Path
if ($resolvedDestination -ne "C:\Users\matth\paired-acquisition-factorization-scorpion") {
    throw "Refusing to clean unexpected DestinationRoot: $resolvedDestination"
}
Get-ChildItem -LiteralPath $DestinationRoot -Force |
    Where-Object { $_.Name -ne ".git" } |
    Remove-Item -Recurse -Force

# Core method code and experiment runners.
Copy-RequiredDirectory "src\models" "src\models"
Copy-RequiredDirectory "experiments\scorpion" "experiments\scorpion"
Copy-RequiredDirectory "scripts\scorpion" "scripts\scorpion"

# Public result and positioning notes.
Copy-RequiredFile "docs\research\paired-acquisition-neural-factorization-positioning.md" "docs\positioning.md"
Copy-RequiredFile "docs\research\paired-acquisition-factorization-scorpion-results.md" "docs\results.md"
Copy-OptionalFile "docs\research\scorpion-pathoalign-crossbackbone-protocol.md" "docs\crossbackbone_protocol.md"
Copy-OptionalFile "docs\research\scorpion-pathoalign-plan.md" "docs\historical_development_plan.md"

# Compact result artifacts. These are intentionally optional because some local
# generated result directories may not be tracked on every checkout.
Copy-OptionalDirectory "results\scorpion\pathoalign_crossbackbone_transfer" "results\crossbackbone_transfer"
Copy-OptionalDirectory "results\scorpion\pathoalign_crossbackbone_transfer_analysis" "results\crossbackbone_transfer_analysis"
Copy-OptionalDirectory "results\scorpion\pathoalign_confirmation" "results\confirmation"
Copy-OptionalDirectory "results\scorpion\pathoalign_calibration" "results\calibration"

# Paper source snapshot. This keeps the exact public-program PDF source available
# at split time. A later child-repo pass can trim the paper down to SCORPION only.
Copy-RequiredFile "paper\arxiv\main.tex" "paper\arxiv\main.tex"
Copy-RequiredFile "paper\arxiv\paired_acquisition_model_math.tex" "paper\arxiv\paired_acquisition_model_math.tex"
Copy-RequiredFile "paper\arxiv\paired_acquisition_figure1_benchmark_table.tex" "paper\arxiv\paired_acquisition_figure1_benchmark_table.tex"
Copy-RequiredFile "paper\arxiv\paired_acquisition_resource_allocation_figure.tex" "paper\arxiv\paired_acquisition_resource_allocation_figure.tex"
Copy-RequiredFile "paper\arxiv\study_specific_packages.tex" "paper\arxiv\study_specific_packages.tex"
Copy-RequiredFile "paper\arxiv\broader_research_program.tex" "paper\arxiv\broader_research_program.tex"
Copy-RequiredFile "paper\arxiv\identifiability_calculations.tex" "paper\arxiv\identifiability_calculations.tex"
Copy-RequiredFile "paper\arxiv\identifiability_calculations_part1.tex" "paper\arxiv\identifiability_calculations_part1.tex"
Copy-RequiredFile "paper\arxiv\identifiability_calculations_part2a.tex" "paper\arxiv\identifiability_calculations_part2a.tex"
Copy-RequiredFile "paper\arxiv\identifiability_calculations_part2b.tex" "paper\arxiv\identifiability_calculations_part2b.tex"
Copy-RequiredFile "paper\arxiv\identifiability_calculations_part3a.tex" "paper\arxiv\identifiability_calculations_part3a.tex"
Copy-RequiredFile "paper\arxiv\build_arxiv_package.py" "paper\arxiv\build_arxiv_package.py"
Copy-OptionalFile "paper\arxiv\references.bib" "paper\arxiv\references.bib"

$readme = @"
# Paired-Acquisition Neural Factorization on SCORPION

Standalone reproducibility package for the core paired-scanner method study:
**Paired-Acquisition Neural Factorization** on the SCORPION histopathology
benchmark.

## Dataset and unit of analysis

- Dataset: SCORPION paired-scanner human H&E benchmark.
- Biological material: 48 original slides.
- Paired regions: 480 aligned tissue regions.
- Scanner views: five scanners per region, 2,400 total real-human-tissue patches.
- Statistical unit: original slide.

## Headline result

The frozen paired-acquisition neural factorization objective reduced linearly
recoverable scanner identity in the biological branch while preserving or
improving same-tissue cross-scanner retrieval across DINOv2, Phikon, and
ImageNet ResNet50 feature substrates.

Transfer-backbone repeated-measures summary over Phikon and ResNet50:

| Metric | Mean difference | 95% slide-bootstrap interval | Favorable slides |
|---|---:|---:|---:|
| Scanner-probe accuracy | -0.401875 | [-0.414458, -0.389458] | 48/48 |
| Mean paired cosine | +0.057762 | [+0.053949, +0.061526] | 48/48 |
| Worst paired cosine | +0.062832 | [+0.058452, +0.067158] | 48/48 |
| Mean top-1 retrieval | +0.004344 | [+0.003313, +0.005427] | 41/48 |
| Worst top-1 retrieval | +0.010833 | [+0.007083, +0.014583] | 36/48 |

## What is included

- Core model code under `src/models`.
- SCORPION experiment runners under `experiments/scorpion`.
- SCORPION feature extraction and analysis scripts under `scripts/scorpion`.
- Result and positioning notes under `docs`.
- Compact result artifacts when available locally.
- Paper source snapshot under `paper/arxiv`.

## What is not included

The repository intentionally excludes raw images, extracted patches, feature
archives, model checkpoints, and large generated run directories.

## Claim boundary

This is a representation-identifiability study. It does not establish clinical
safety, diagnostic equivalence, patient-outcome improvement, external-center
clinical deployment, or perfect biological/acquisition disentanglement.
"@
New-TextFile "README.md" $readme

$gitignore = @"
# Raw image and slide artifacts
*.svs
*.ndpi
*.mrxs
*.tif
*.tiff
*.jpg
*.jpeg
*.png

# Feature/model artifacts
*.npz
*.pt
*.pth
*.ckpt
*.safetensors

# Regenerable local outputs
results/**/runs/
results/**/fold_*/
paper/paired-acquisition-factorization-scorpion.pdf
paper/arxiv/*.aux
paper/arxiv/*.bbl
paper/arxiv/*.blg
paper/arxiv/*.log
paper/arxiv/*.out
paper/arxiv/main.pdf
paper/arxiv/build/
paper/arxiv/*.zip

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
scikit-learn
torch
torchvision
transformers
Pillow
matplotlib
"@
New-TextFile "requirements.txt" $requirements

$paperBuild = @'
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "arxiv")
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Copy-Item main.pdf ..\paired-acquisition-factorization-scorpion.pdf -Force
Copy-Item main.pdf ..\..\paired-acquisition-factorization-scorpion.pdf -Force
Write-Host "Built paired-acquisition-factorization-scorpion.pdf"
'@
New-TextFile "paper\build.ps1" $paperBuild

$citation = @"
cff-version: 1.2.0
title: Paired-Acquisition Neural Factorization on SCORPION
message: If you use this reproducibility package, please cite the repository and the SCORPION benchmark.
authors:
  - family-names: Chorne
    given-names: Larry
repository-code: https://github.com/matthewvaishnav/paired-acquisition-factorization-scorpion
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
Get-ChildItem $DestinationRoot -Recurse -Include *.npz,*.pt,*.pth,*.ckpt,*.safetensors,*.tif,*.tiff,*.jpg,*.jpeg,*.png,*.svs,*.ndpi,*.mrxs -File |
    ForEach-Object {
        Write-Warning "Removing generated/heavy artifact from scaffold: $($_.FullName)"
        Remove-Item $_.FullName -Force
    }

Convert-LegacyPublicNames

if ($InitializeGit) {
    Push-Location $DestinationRoot
    try {
        if (!(Test-Path ".git")) {
            git init | Out-Host
        }
        git add -A | Out-Host
        git status --short | Out-Host
    } finally {
        Pop-Location
    }
}

Write-Host "Standalone SCORPION Paired-Acquisition Neural Factorization repo scaffold complete: $DestinationRoot"
Write-Host "Next: inspect the directory, then create/push the GitHub repository paired-acquisition-factorization-scorpion."
