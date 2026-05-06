#!/bin/bash
# Security scan script for HistoCore dependencies
# Run this after installing dependencies in your virtual environment

set -e

echo "=== HistoCore Security Scan ==="
echo ""

# Check if safety is installed
if ! command -v safety &> /dev/null; then
    echo "Installing safety..."
    pip install safety
fi

echo "Scanning core dependencies..."
safety check --file requirements-core.txt --output text

echo ""
echo "Scanning dev dependencies..."
safety check --file requirements-dev.txt --output text

echo ""
echo "Scanning optional dependencies..."
safety check --file requirements-optional.txt --output text

echo ""
echo "=== Scan Complete ==="
echo ""
echo "To fix vulnerabilities:"
echo "1. Review the output above"
echo "2. Update affected packages in requirements files"
echo "3. Test thoroughly after updates"
echo "4. Re-run this script to verify fixes"
