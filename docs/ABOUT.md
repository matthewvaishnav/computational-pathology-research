# About Computational Pathology Research

## Project Overview

**Computational Pathology Research** is a PyTorch-based research engineering framework for digital pathology, whole-slide image analysis, multiple-instance learning (MIL), and federated oncology validation. The project includes end-to-end infrastructure for pathology feature extraction, manifest generation, HDF5 feature validation, slide-level MIL training, benchmark reporting, and research-only clinical workflow prototypes.

The current validated research stack includes PCam patch-level benchmarking and PANDA prostate cancer slide-level experiments using Phikon feature embeddings. On the current PANDA held-out split, mean-pooled Phikon features achieved **QWK 0.7274**, gated AttentionMIL achieved **QWK 0.8100**, and tuned TransnnMIL achieved **QWK 0.8155** after HDF5 readability verification over **10,611 readable slides**.

The framework also includes **FAIR-WEIGHTS-H**, a mathematical research framework for auditable institutional weighting in federated oncology learning, and PathologyFL-oriented infrastructure for privacy-preserving multi-site experimentation.

**Clinical status:** research-only. This repository is not clinically validated, not FDA-cleared, not Health Canada-authorized, and not diagnostic software.

---

## Current Statistics

- **Source Code**: 141 Python modules in `src/`
- **Test Suite**: 186 test files with 3,006+ total tests
- **Code Coverage**: 55% with property-based testing coverage
- **Development Activity**: 286+ commits since January 2026
- **PCam Benchmark**: 95.37% validation AUC, 85.26% test accuracy
- **PANDA Slide-Level Baselines**:
  - Mean-pooled Phikon + MLP: QWK 0.7274
  - Gated AttentionMIL: QWK 0.8100
  - Tuned TransnnMIL: QWK 0.8155
- **PANDA Feature Integrity**: 10,611 readable slide-level Phikon feature files after manifest filtering and read verification
- **Federated Oncology Direction**: FAIR-WEIGHTS-H and PathologyFL research infrastructure

---

## Key Capabilities

### Computational Pathology Modeling
- Patch-level PCam benchmarking
- Slide-level PANDA prostate cancer grading experiments
- Phikon feature extraction and downstream MIL training
- AttentionMIL, CLAM, TransMIL, and TransnnMIL model support
- Quadratic weighted kappa (QWK), accuracy, macro F1, and confusion-matrix reporting

### Slide-Level MIL Pipelines
- Manifest-driven training over extracted HDF5 feature files
- Variable-length feature-bag loading with masked padding
- HDF5 read-integrity verification before training
- Safe artifact tracking that excludes raw slides, feature stores, and checkpoints
- Reproducible result artifacts for mean pooling, AttentionMIL, and TransnnMIL baselines

### Federated Oncology Research
- FAIR-WEIGHTS-H institutional weighting framework
- Difficulty-adjusted quality, contribution, uniqueness, uncertainty, entropy, and subgroup-constraint concepts
- PathologyFL-oriented privacy-preserving multi-site learning direction
- Research prototypes for differential privacy, secure aggregation, and federated evaluation

### WSI and Feature Processing Infrastructure
- Whole-slide image processing direction with OpenSlide-based workflows
- Multi-format pathology image support where dependencies are available
- Patch extraction, feature extraction, and HDF5 caching workflows
- GPU-accelerated training pipelines on consumer hardware

### Model Interpretability
- Attention weight extraction for MIL models
- Grad-CAM support for CNN feature extractors
- Failure-case analysis and visualization utilities
- Research-oriented tools for pathologist-facing explanation workflows

### Testing and Engineering Hygiene
- PyTest and Hypothesis-based testing
- Data hygiene rules for medical imaging artifacts
- `.gitignore` protection for `.tif`, `.h5`, `.pt`, checkpoints, feature stores, and raw datasets
- Documentation-first experiment tracking and claim-boundary discipline

---

## Author

**Matthew Vaishnav** is building computational pathology AI research infrastructure in Kitchener-Waterloo, Ontario. His work focuses on digital pathology, multiple-instance learning, prostate cancer slide-level prediction, federated oncology learning, and mathematical validation infrastructure.

Recent work includes a PANDA prostate cancer slide-level pipeline using Phikon features, a validated manifest over 10,611 readable slides, and MIL baselines where tuned TransnnMIL reached QWK 0.8155 on the current held-out split. He is also developing FAIR-WEIGHTS-H, a framework direction for auditable institutional weighting in federated oncology learning.

### Contact

- **GitHub**: [matthewvaishnav](https://github.com/matthewvaishnav)
- **Email**: matthew.vaishnav@gmail.com
- **Location**: Kitchener-Waterloo, Ontario, Canada

---

## Project History

### 2026: Core Development
- Initial framework architecture and PyTorch implementation
- PatchCamelyon benchmark pipeline
- Attention-based MIL models: AttentionMIL, CLAM, TransMIL, and TransnnMIL
- PANDA Phikon feature extraction and slide-level manifest generation
- PANDA mean-pooling, AttentionMIL, and tuned TransnnMIL baselines
- HDF5 read-integrity verification for extracted feature files
- FAIR-WEIGHTS-H theory and documentation
- PathologyFL federated learning research direction
- WSI processing pipeline with OpenSlide-oriented integration
- Clinical workflow and PACS integration prototypes
- Comprehensive testing infrastructure
- Model interpretability tools

### Validated Benchmarks

#### PatchCamelyon (PCam)
- **Validation AUC**: 95.37% (primary metric)
- **Test Accuracy**: 85.26% ± 0.40% (95% CI: 84.83%-85.63%)
- **Test F1**: 0.8507 ± 0.0040 (95% CI: 0.8464-0.8543)
- **Dataset**: 262,144 train, 32,768 val, 32,768 test (96×96 RGB patches)
- **Hardware**: RTX 4070 Laptop (8GB VRAM)
- **Training Time**: 2-3 hours (15 epochs)

#### PANDA Prostate Cancer Slide-Level Baselines
- **Dataset**: PANDA prostate cancer histopathology
- **Feature Source**: Phikon patch embeddings
- **Readable Slides Used**: 10,611
- **Train/Validation Split**: 8,488 train / 2,123 validation
- **Mean-Pooled Phikon + MLP**: validation QWK 0.7274
- **Gated AttentionMIL**: validation QWK 0.8100
- **Tuned TransnnMIL**: validation QWK 0.8155
- **Status**: research-only single-split result; repeated-seed validation and ablations planned

---

## Research Applications

Computational Pathology Research is designed for:

1. **Academic Research**: Reproducible computational pathology experiments
2. **Algorithm Development**: Prototyping and comparing MIL architectures
3. **Slide-Level Benchmarking**: PANDA, PCam, and future Camelyon-style validation
4. **Federated Oncology Research**: Privacy-preserving multi-site validation and institutional weighting
5. **Interpretability Research**: Attention and feature-attribution workflows for pathology AI
6. **Engineering Education**: Teaching robust medical AI data handling, artifact hygiene, and claim boundaries

---

## Technology Stack

- **Deep Learning**: PyTorch, torchvision, timm
- **Medical Imaging**: OpenSlide, pydicom, python-gdcm where available
- **Clinical Standards Prototypes**: DICOM, FHIR, PACS-oriented interfaces
- **Data Processing**: NumPy, pandas, h5py, Pillow
- **Visualization**: matplotlib, seaborn, Grad-CAM
- **Testing**: pytest, Hypothesis
- **Deployment Prototypes**: Docker, Kubernetes, ONNX
- **CI/CD**: GitHub Actions, codecov

---

## License

This project is released under the MIT License. See [LICENSE](../LICENSE) for details.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{vaishnav2026computational_pathology_research,
  title = {Computational Pathology Research: PyTorch Infrastructure for Digital Pathology, MIL, and Federated Oncology Validation},
  author = {Vaishnav, Matthew},
  year = {2026},
  url = {https://github.com/matthewvaishnav/computational-pathology-research},
  note = {Research framework with PCam benchmarking, PANDA Phikon slide-level baselines, FAIR-WEIGHTS-H, and PathologyFL-oriented infrastructure.}
}
```

---

## Acknowledgments

This framework builds upon foundational work in computational pathology:

- **CAMELYON Dataset**: Ehteshami Bejnordi et al. (2018) - GigaScience
- **PatchCamelyon**: Veeling et al. (2018) - Medical Image Analysis
- **Attention MIL**: Ilse et al. (2018) - ICML
- **CLAM**: Lu et al. (2021) - Nature Biomedical Engineering
- **TransMIL**: Shao et al. (2021) - NeurIPS
- **PANDA Challenge**: Prostate cANcer graDe Assessment dataset

---

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

For questions or collaboration opportunities, please open an issue on [GitHub](https://github.com/matthewvaishnav/computational-pathology-research/issues).

---

*Last updated: May 2026*
