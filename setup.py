#!/usr/bin/env python
"""Setup script for Sentinel - Computational Pathology Research Framework."""

from setuptools import setup, find_packages
import os

# Read version from version file
version = {}
with open("src/sentinel/version.py") as f:
    exec(f.read(), version)

# Read long description from README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="sentinel-pathology",
    version=version["__version__"],
    author="Matthew Vaishnav",
    author_email="",
    description="Computational Pathology Research Framework for WSI analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matthewvaishnav/computational-pathology-research",
    project_urls={
        "Bug Tracker": "https://github.com/matthewvaishnav/computational-pathology-research/issues",
        "Documentation": "https://matthewvaishnav.github.io/computational-pathology-research/",
        "Source Code": "https://github.com/matthewvaishnav/computational-pathology-research",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "gui": [
            "PyQt6>=6.4.0",
            "pyqtgraph>=0.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sentinel=sentinel.cli:main",
            "sentinel-train=sentinel.cli:train_command",
            "sentinel-eval=sentinel.cli:eval_command",
            "sentinel-gui=sentinel.gui.app:main [gui]",
        ],
    },
    include_package_data=True,
    package_data={
        "sentinel": ["configs/*.yaml", "assets/*"],
    },
    zip_safe=False,
)
