# Model Cards

This page summarizes the model-card layer for the project.

## Purpose

Model cards separate model capability claims from implementation details.

Each model card should include:

- model name and version
- task and dataset context
- training data summary
- evaluation metrics
- intended use
- limitations
- validation status

## Current model families

- TransnnMIL v2.0
- AttentionMIL
- CLAM-style MIL
- TransMIL-style baselines
- foundation encoder pipelines

## Evidence standard

A result should specify:

- dataset
- split
- seed or cross-validation protocol
- metric definition
- hardware / training configuration
- whether the result is smoke, benchmark, or validation

Clinical claims require separate validation and are not implied by benchmark results.
