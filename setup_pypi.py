#!/usr/bin/env python3
"""
Setup script for HistoCore PyPI package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read core requirements
with open("requirements-core.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="histocore",
    version="0.1.0",
    author="Matthew Vaishnav",
    author_email="matthew.vaishnav@example.com",  # Replace with real email
    description="Production-grade computational pathology framework with federated learning and PACS integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matthewvaishnav/computational-pathology-research",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy", "hypothesis"],
        "full": ["pydicom", "openslide-python", "tensorboard", "fastapi", "uvicorn"],
        "federated": ["cryptography", "pycryptodome"],
    },
    entry_points={
        "console_scripts": [
            "histocore=src.cli:main",
        ],
    },
    keywords="pathology, machine learning, federated learning, medical imaging, DICOM, WSI",
    project_urls={
        "Bug Reports": "https://github.com/matthewvaishnav/computational-pathology-research/issues",
        "Source": "https://github.com/matthewvaishnav/computational-pathology-research",
        "Documentation": "https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/",
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
)