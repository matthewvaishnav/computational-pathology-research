---
license: other
library_name: pytorch
tags:
  - computational-pathology
  - representation-learning
  - paired-acquisition
  - research
---

# PA-NF — Paired-Acquisition Neural Factorization

PA-NF is a paired-acquisition representation-learning model that separates
frozen pathology features into biological and acquisition branches under
scanner-paired supervision. This repository is the trained model artifact; its
supported result is a bounded, dataset- and comparator-specific structured-
separation result, not evidence of pure factors, complete scanner invariance,
clinical utility, or deployment readiness.

The separate
[`MatthewVaishnav/paired-acquisition-factorization-evidence`](https://huggingface.co/datasets/MatthewVaishnav/paired-acquisition-factorization-evidence)
repository contains registered metrics, analyses, manifests, and negative
results. This model repository contains the trained PA-NF family and the
preprocessing state required to use it.

## Method summary

The release object is the complete registered `pathoalign_dep20` family from the
SCORPION capacity-matched campaign: five slide-blocked folds × seeds 801–805 =
25 fixed-final-epoch checkpoints. The campaign had no model-selection rule and
records `checkpoint_selection: none; fixed final epoch only`. No checkpoint is
called “best,” and published test results must not be used to rank the 25
instances post hoc.

## Architecture

Each PyTorch checkpoint contains a 1,550,026-parameter `ScorpionProjection` with:

- a 768 → 512 → 256 biological projection;
- a 768 → 512 → 64 acquisition projection;
- a 320 → 512 → 768 joint reconstruction decoder;
- five-class scanner heads on both branches;
- the registered contrastive, reconstruction, variance, covariance, scanner-
  adversarial, acquisition-scanner, scanner-dependence, and cross-covariance
  objectives.

The exact architecture is distributed as `scorpion_pathoalign.py`. Biological
and acquisition latent dimensions are 256 and 64; the hidden dimension is 512.

## Intended use

- reproduce the registered SCORPION PA-NF projections with a matching fold and
  seed;
- audit checkpoint-to-campaign provenance and state-dict identity;
- study paired-acquisition representation behavior on frozen features under a
  prespecified research protocol;
- apply all 25 co-equal family members, or choose a fold/seed by a rule specified
  independently of the published test outcomes.

## Out-of-scope use

Do not use this release for diagnosis, treatment, patient care, clinical or
regulatory decisions, production deployment, or claims of universal scanner
harmonization. Do not select a checkpoint using the registered held-out test
results. The model does not accept raw pathology images.

## Training data

The registered campaign used SCORPION v1: 2,400 spatially aligned H&E patches
from 480 regions on 48 original slides, each acquired on five scanners. Folds
are blocked by original slide. Each fold's preprocessing statistics were fit on
all non-test slides in that registered fold.

Inputs were frozen `facebook/dinov2-base` 768-D features at immutable Hub
revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`. The DINOv2 feature archive,
SCORPION images, and image paths are not redistributed here.

## Evaluation

On the registered SCORPION structured-separation objective, the full model's
primary scanner-suppression contrast against the equal-capacity two-branch
no-scanner-objectives control was:

`full - capacity-matched no-scanner-objectives = -0.3108333333`

with fold-aware 95% interval
`[-0.3346122449, -0.2857872340]`. Average same-region retrieval and worst-pair
retrieval were preserved within their registered noninferiority margins, while
the acquisition branch retained strong scanner information.

This is distinct from the corrected canine fixed five-category estimand, where
the neural feature-space representations did not establish an additional
improvement over the strongest simple scanner-removal baselines. The SCORPION
result does not establish universal superiority; the canine result does not
erase the registered SCORPION controlled advantage.

## Supported claims

The release supports partial structured separation under the tested paired-
acquisition protocol and a controlled comparative advantage over the registered
equal-capacity two-branch neural control on the SCORPION objective. It supports
reduced linear scanner recoverability in the tissue-oriented branch, preserved
registered same-region retrieval noninferiority, and retained scanner
information in the acquisition branch.

## Unsupported claims

This release does not establish pure biological factors, complete scanner
invariance, complete disentanglement, information-theoretic independence,
universal superiority, diagnostic improvement, clinical utility, patient
benefit, deployment readiness, or regulatory readiness.

## Limitations

The model was trained on one small scanner-paired dataset and one frozen feature
contract. Its outputs depend on the exact DINOv2 representation and fold-specific
standardization. Scanner labels were used during training for the scanner
objectives, but are not required by the forward pass at inference. Broader human-
tissue, site, device, backbone, and task validation is not supplied by these
checkpoints.

## Provenance

- training source commit:
  `0adea50f1ef22865969109f1834a3c175e3f8b43`;
- model definition: `src/models/scorpion_pathoalign.py` at SHA256
  `528b653459bbdb759c5ca414cf434cf0ad54fba1ef6cbc98e6968c235a3c799f`;
- campaign config hash:
  `59ba04450adf738aa2983dc40ecb4ff09ffa495f13d43c56edde5ec57707a2e1`;
- evidence dataset revision:
  `a9853bd32e3b446a97608002f7e5ea12f68f88e1`;
- source repository:
  <https://github.com/matthewvaishnav/computational-pathology-research>.

`model-manifest.json` binds every fold, seed, run ID, checkpoint SHA256, source
cell-manifest SHA256, preprocessing SHA256, config, and state-dict validation.
`checksums.sha256` covers the complete released folder. The immutable model Hub
revision is recorded only after upload, redownload, and checksum verification.

## Citation

Please cite the repository `CITATION.cff`, the PA-NF foundations manuscript, the
[SCORPION dataset](https://doi.org/10.5281/zenodo.16517924), and
[DINOv2](https://arxiv.org/abs/2304.07193). The evidence repository carries the
registered result and analysis provenance.

## License

The researcher-authored source and inference code are MIT-licensed. The frozen
`facebook/dinov2-base` source model is identified as Apache-2.0 by its upstream
model card and repository; its weights and frozen feature archive are not
included here.

The SCORPION Zenodo v1 record currently does not state an explicit license.
Public redistribution of the derived PA-NF checkpoints therefore remains
blocked until documented permission or an explicit compatible dataset license
is recorded. The `license: other` metadata is intentional and must not be changed
to MIT merely because the surrounding source repository is MIT-licensed.

## Reproducibility

The checkpoint alone is not sufficient. Use the matching
`preprocessing/fold_X_standardization.npz`, which contains 1 × 768 float32
`mean` and `std` arrays fit on that fold's non-test slides.

```python
from pathlib import Path
import numpy as np

from inference import load_panf, project_features

root = Path(".")
# Example coordinate only; fold 0 / seed 801 is not designated as the best model.
model, mean, std, identity = load_panf(root, fold=0, seed=801)
raw_dinov2_features = np.load("my_compatible_features.npy")
outputs = project_features(model, raw_dinov2_features, mean, std)
biological = outputs["biological"]  # [n, 256]
acquisition = outputs["acquisition"]  # [n, 64]
```

Before use, verify the package and all 25 checkpoint contents:

```powershell
python tools/huggingface/panf_model_bundle.py verify --bundle build/huggingface/paired-acquisition-neural-factorization
python tools/huggingface/release.py verify-local build/huggingface/paired-acquisition-neural-factorization
```

Scanner labels are required to reproduce the registered training objectives;
they are not inputs to the trained projection at inference.
