# Citation Audit

**Date:** 2026-08-05
**Manuscript:** `manuscripts/computational-pathology-foundations-v1/`

## Rule

Only verifiable primary sources are cited. No fabricated citations. No
absolute-priority wording. Prior-art citations are used only to attribute
building blocks, never to claim novelty.

## Bibliography entries (references.bib)

| Key | Work | Verification status |
| --- | --- | --- |
| `veeling2018rotation` | Veeling et al., MICCAI 2018, Rotation Equivariant CNNs | verified (peer-reviewed) |
| `pcam2018kaggle` | PCam benchmark | verified (public benchmark) |
| `shao2021transmil` | TransMIL, NeurIPS 2021 | verified (peer-reviewed) |
| `ilse2018attention` | ABMIL/Attention-based Deep MIL, ICML 2018 | verified (peer-reviewed) |
| `lu2021clam` | CLAM, Nature Biomedical Engineering 2021 | verified (peer-reviewed) |
| `bulten2022panda` | PANDA challenge, Nature Medicine 2022 | verified (peer-reviewed) |
| `bandi2018camelyon17` | CAMELYON17, IEEE TMI 2018 | verified (peer-reviewed) |
| `koh2021wilds` | WILDS, ICML 2021 | verified (peer-reviewed) |
| `filiot2023phikon` | Phikon, medRxiv 2023 | verified (preprint) |
| `mcmahan2017fedavg` | FedAvg, AISTATS 2017 | verified (peer-reviewed) |
| `li2020fedprox` | FedProx, MLSys 2020 | verified (peer-reviewed) |
| `blanchard2017krum` | Krum, NeurIPS 2017 | verified (peer-reviewed) |
| `abadi2016dp` | DP-SGD, CCS 2016 | verified (peer-reviewed) |
| `bonawitz2017secure` | Secure aggregation, CCS 2017 | verified (peer-reviewed) |
| `shapley1953` | Shapley value, 1953 | verified (classic) |
| `ryu2023scorpion` | SCORPION dataset, MICCAI UNSURE 2023 / arXiv:2507.20907 | verified via prior-art review |
| `lu2022histofl` | HistoFL, MedIA 2022 | verified via prior-art review |
| `hou2022h2mil` | H2-MIL, AAAI 2022 | verified via prior-art review |

## Entries removed for unverifiability

Two entries from the initial prior-art pass were removed from `references.bib`
because their author fields could not be verified from the searched primary
sources: "Mind the Gap" (arXiv:2211.16141) and "ScanGen" (arXiv:2507.22092).
These remain referenced by URL in
`docs/research/full-program-prior-art-review-20260804.md` but are not in the
manuscript bibliography.

## Placeholder check

`grep -ciE "placeholder|TODO|FIXME|unverified" references.bib` returns 0.

## Build check

The manuscript builds with **0 unresolved citations** and **0 undefined
references** (`latexmk` log).
