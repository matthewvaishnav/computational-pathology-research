# arXiv submission metadata

## Title

Paired-Acquisition Neural Factorization: An End-to-End Computational Pathology Pipeline

## Authors

Matthew Vaishnav

## Primary category

`cs.CV`

## Secondary categories

`cs.LG`, `q-bio.QM`

## Comments

Program-level PA-NF preprint spanning paired-scanner representation learning, whole-slide modeling, and multi-institutional learning across SCORPION, multi-scanner canine SCC, PANDA, CAMELYON17/WILDS, and PCam. Supplement contains implementation, testing, result-pointer, and reproducibility details.

## Abstract

Paired-Acquisition Neural Factorization (PA-NF) is an end-to-end computational pathology research pipeline spanning scanner-aware representation learning, whole-slide neural aggregation, spatial modeling, and multi-institutional learning. The registered SCORPION campaign uses 48 human H&E slides, 480 aligned regions, five scanners, and 2,400 images. Across 175 capacity-matched fits, PA-NF reduced tissue-branch scanner balanced accuracy by 0.3108 relative to an equal-capacity two-branch neural control, with fold/slide bootstrap interval [-0.3346,-0.2858], while same-region retrieval remained within a preregistered 0.02 noninferiority margin and scanner information remained concentrated in an explicit acquisition branch. The frozen objective transferred across DINOv2, Phikon, and ImageNet ResNet50 feature families, while an independent five-scanner canine SCC study established a harder boundary in which strong simple scanner-removal baselines remain competitive. On PANDA, 10,611 readable Phikon slide bags support TransnnMIL and institutional experiments. Stabilized TransnnMIL reached mean best validation QWK 0.8257 across three seeds at learning rate 1e-4, with a best recorded run of 0.8455. A dominance-aware detector-switch policy improved global QWK over FedAvg at 25% and 35% dominant-site corruption and transferred without retuning from label-noise stress to systematic conservative ordinal threshold shift; at 45% shift, global QWK, macro-F1, and worst-site QWK improved by 0.01053, 0.01512, and 0.01290, respectively, with positive 95% intervals. CAMELYON17/WILDS adds 455,954 examples across five centers: equal-client weighting improved held-out-center accuracy from 0.8312 to 0.9132 on frozen ImageNet features, and dominant-source downweighting reached 0.9322 with source-trained features versus 0.9052 for sample-proportional weighting. PCam supplies a complete patch-level benchmark with a documented 0.9394 ROC AUC on the official 32,768-patch test split. The strongest conclusions are comparator-specific and do not constitute clinical validation or a universal state-of-the-art claim.
