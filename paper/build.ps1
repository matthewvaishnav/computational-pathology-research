$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

Write-Host "Built paper/main.pdf"
