# Camelyon17 privacy-noise stress test

## Purpose

This is the first Pillar 3 stress test.

It does not prove formal differential privacy. Instead, it adds Gaussian noise to the feature/head classifier coefficients to mimic privacy/noisy-update degradation pressure, then asks whether weighting-policy gains survive.

## Setup

- Feature extractor: Camelyon17 source-trained ResNet18
- Classifier: logistic head
- Sample size: 8,000
- Noise levels: [0.0, 0.01, 0.03, 0.05, 0.1, 0.2]
- Noise repeats per setting: 5

## Summary

| policy                     |   noise_std | split   |   accuracy_mean |   accuracy_std |   balanced_accuracy_mean |   balanced_accuracy_std |   macro_f1_mean |   macro_f1_std |   auc_mean |   auc_std |
|:---------------------------|------------:|:--------|----------------:|---------------:|-------------------------:|------------------------:|----------------:|---------------:|-----------:|----------:|
| downweight_dominant_center |        0    | id_val  |        0.968333 |       0        |                 0.968333 |                0        |        0.968331 |       0        |   0.995082 |  0        |
| downweight_dominant_center |        0    | test    |        0.934    |       0        |                 0.934    |                0        |        0.933948 |       0        |   0.978872 |  0        |
| downweight_dominant_center |        0    | val     |        0.925    |       0        |                 0.925    |                0        |        0.924996 |       0        |   0.979928 |  0        |
| downweight_dominant_center |        0.01 | id_val  |        0.968267 |       0.000149 |                 0.968267 |                0.000149 |        0.968264 |       0.000149 |   0.995082 |  1e-06    |
| downweight_dominant_center |        0.01 | test    |        0.934    |       0        |                 0.934    |                0        |        0.933948 |       0        |   0.978871 |  1.3e-05  |
| downweight_dominant_center |        0.01 | val     |        0.9254   |       0.000548 |                 0.9254   |                0.000548 |        0.925396 |       0.000547 |   0.979926 |  1e-05    |
| downweight_dominant_center |        0.03 | id_val  |        0.968133 |       0.000183 |                 0.968133 |                0.000183 |        0.968131 |       0.000183 |   0.995076 |  7e-06    |
| downweight_dominant_center |        0.03 | test    |        0.9334   |       0.000548 |                 0.9334   |                0.000548 |        0.933345 |       0.00055  |   0.978854 |  7.7e-05  |
| downweight_dominant_center |        0.03 | val     |        0.9258   |       0.000447 |                 0.9258   |                0.000447 |        0.925796 |       0.000447 |   0.979942 |  3e-05    |
| downweight_dominant_center |        0.05 | id_val  |        0.968067 |       0.000279 |                 0.968067 |                0.000279 |        0.968064 |       0.000279 |   0.995073 |  1.1e-05  |
| downweight_dominant_center |        0.05 | test    |        0.9336   |       0.000548 |                 0.9336   |                0.000548 |        0.933546 |       0.00055  |   0.978907 |  4.2e-05  |
| downweight_dominant_center |        0.05 | val     |        0.9254   |       0.000894 |                 0.9254   |                0.000894 |        0.925394 |       0.000896 |   0.979862 |  8.5e-05  |
| downweight_dominant_center |        0.1  | id_val  |        0.968267 |       0.000279 |                 0.968267 |                0.000279 |        0.968264 |       0.000279 |   0.995059 |  1.2e-05  |
| downweight_dominant_center |        0.1  | test    |        0.9342   |       0.001095 |                 0.9342   |                0.001095 |        0.934149 |       0.0011   |   0.978874 |  0.000173 |
| downweight_dominant_center |        0.1  | val     |        0.9262   |       0.001304 |                 0.9262   |                0.001304 |        0.926197 |       0.001304 |   0.97983  |  0.000227 |
| downweight_dominant_center |        0.2  | id_val  |        0.968333 |       0.000745 |                 0.968333 |                0.000745 |        0.968331 |       0.000746 |   0.995045 |  2.3e-05  |
| downweight_dominant_center |        0.2  | test    |        0.935    |       0.001    |                 0.935    |                0.001    |        0.934952 |       0.001004 |   0.978936 |  0.000322 |
| downweight_dominant_center |        0.2  | val     |        0.926    |       0.002    |                 0.926    |                0.002    |        0.925996 |       0.002002 |   0.97968  |  0.000396 |
| equal_client               |        0    | id_val  |        0.967667 |       0        |                 0.967667 |                0        |        0.967663 |       0        |   0.995162 |  0        |
| equal_client               |        0    | test    |        0.934    |       0        |                 0.934    |                0        |        0.933948 |       0        |   0.97956  |  0        |
| equal_client               |        0    | val     |        0.926    |       0        |                 0.926    |                0        |        0.925995 |       0        |   0.98058  |  0        |
| equal_client               |        0.01 | id_val  |        0.967667 |       0        |                 0.967667 |                0        |        0.967663 |       0        |   0.995164 |  1e-06    |
| equal_client               |        0.01 | test    |        0.934    |       0        |                 0.934    |                0        |        0.933948 |       0        |   0.97955  |  2e-05    |
| equal_client               |        0.01 | val     |        0.926    |       0        |                 0.926    |                0        |        0.925995 |       0        |   0.980582 |  1.7e-05  |
| equal_client               |        0.03 | id_val  |        0.967733 |       0.000149 |                 0.967733 |                0.000149 |        0.96773  |       0.000149 |   0.995163 |  5e-06    |
| equal_client               |        0.03 | test    |        0.934    |       0        |                 0.934    |                0        |        0.933948 |       0        |   0.979565 |  5.8e-05  |
| equal_client               |        0.03 | val     |        0.927    |       0.001    |                 0.927    |                0.001    |        0.926996 |       0.001001 |   0.980604 |  3.1e-05  |
| equal_client               |        0.05 | id_val  |        0.967667 |       0        |                 0.967667 |                0        |        0.967663 |       0        |   0.99516  |  1.5e-05  |
| equal_client               |        0.05 | test    |        0.934    |       0        |                 0.934    |                0        |        0.933948 |       0        |   0.979614 |  6.4e-05  |
| equal_client               |        0.05 | val     |        0.9262   |       0.000447 |                 0.9262   |                0.000447 |        0.926195 |       0.000448 |   0.980517 |  7.1e-05  |
| equal_client               |        0.1  | id_val  |        0.968067 |       0.000279 |                 0.968067 |                0.000279 |        0.968064 |       0.000279 |   0.995143 |  1.3e-05  |
| equal_client               |        0.1  | test    |        0.9348   |       0.000447 |                 0.9348   |                0.000447 |        0.934752 |       0.000449 |   0.979601 |  0.000189 |
| equal_client               |        0.1  | val     |        0.9266   |       0.001673 |                 0.9266   |                0.001673 |        0.926597 |       0.001674 |   0.980524 |  0.000222 |
| equal_client               |        0.2  | id_val  |        0.969    |       0.000745 |                 0.969    |                0.000745 |        0.968997 |       0.000746 |   0.995121 |  3.6e-05  |
| equal_client               |        0.2  | test    |        0.9358   |       0.001643 |                 0.9358   |                0.001643 |        0.935754 |       0.001651 |   0.979631 |  0.000322 |
| equal_client               |        0.2  | val     |        0.9262   |       0.001924 |                 0.9262   |                0.001924 |        0.926196 |       0.001925 |   0.980366 |  0.000393 |
| fedavg_equal_patch         |        0    | id_val  |        0.970667 |       0        |                 0.970667 |                0        |        0.970666 |       0        |   0.994578 |  0        |
| fedavg_equal_patch         |        0    | test    |        0.91     |       0        |                 0.91     |                0        |        0.909982 |       0        |   0.967168 |  0        |
| fedavg_equal_patch         |        0    | val     |        0.916    |       0        |                 0.916    |                0        |        0.915959 |       0        |   0.969804 |  0        |
| fedavg_equal_patch         |        0.01 | id_val  |        0.970533 |       0.000183 |                 0.970533 |                0.000183 |        0.970533 |       0.000183 |   0.99457  |  2e-05    |
| fedavg_equal_patch         |        0.01 | test    |        0.9088   |       0.000837 |                 0.9088   |                0.000837 |        0.908784 |       0.000835 |   0.967089 |  0.000101 |
| fedavg_equal_patch         |        0.01 | val     |        0.9146   |       0.001342 |                 0.9146   |                0.001342 |        0.914556 |       0.001349 |   0.969677 |  9.3e-05  |
| fedavg_equal_patch         |        0.03 | id_val  |        0.9708   |       0.00073  |                 0.9708   |                0.00073  |        0.970799 |       0.000731 |   0.994565 |  2.9e-05  |
| fedavg_equal_patch         |        0.03 | test    |        0.9088   |       0.000837 |                 0.9088   |                0.000837 |        0.908785 |       0.000839 |   0.967103 |  0.000367 |
| fedavg_equal_patch         |        0.03 | val     |        0.9144   |       0.00313  |                 0.9144   |                0.00313  |        0.914354 |       0.003148 |   0.969841 |  0.000365 |
| fedavg_equal_patch         |        0.05 | id_val  |        0.971733 |       0.001011 |                 0.971733 |                0.001011 |        0.971733 |       0.001011 |   0.994577 |  3.1e-05  |
| fedavg_equal_patch         |        0.05 | test    |        0.9094   |       0.001949 |                 0.9094   |                0.001949 |        0.909354 |       0.001941 |   0.967105 |  0.00057  |
| fedavg_equal_patch         |        0.05 | val     |        0.9172   |       0.004817 |                 0.9172   |                0.004817 |        0.917174 |       0.004834 |   0.970251 |  0.000928 |
| fedavg_equal_patch         |        0.1  | id_val  |        0.970667 |       0.002186 |                 0.970667 |                0.002186 |        0.970666 |       0.002187 |   0.994635 |  0.000141 |
| fedavg_equal_patch         |        0.1  | test    |        0.9128   |       0.000837 |                 0.9128   |                0.000837 |        0.912789 |       0.000826 |   0.968706 |  0.000646 |
| fedavg_equal_patch         |        0.1  | val     |        0.917    |       0.001225 |                 0.917    |                0.001225 |        0.916963 |       0.001228 |   0.970895 |  0.00154  |
| fedavg_equal_patch         |        0.2  | id_val  |        0.967733 |       0.003897 |                 0.967733 |                0.003897 |        0.96773  |       0.003901 |   0.994575 |  0.000271 |
| fedavg_equal_patch         |        0.2  | test    |        0.916    |       0.00495  |                 0.916    |                0.00495  |        0.915979 |       0.00495  |   0.96915  |  0.001999 |
| fedavg_equal_patch         |        0.2  | val     |        0.9166   |       0.004393 |                 0.9166   |                0.004393 |        0.916563 |       0.004405 |   0.970764 |  0.003784 |

## Test-set deltas versus FedAvg-style weighting

Positive values mean the alternative policy improved held-out test performance over FedAvg-style equal-patch weighting under the same noise level.

| policy                     |   noise_std | split   |   accuracy_delta_vs_fedavg |   macro_f1_delta_vs_fedavg |   auc_delta_vs_fedavg |
|:---------------------------|------------:|:--------|---------------------------:|---------------------------:|----------------------:|
| equal_client               |        0    | test    |                     0.024  |                   0.023966 |              0.012392 |
| downweight_dominant_center |        0    | test    |                     0.024  |                   0.023966 |              0.011704 |
| equal_client               |        0.01 | test    |                     0.0252 |                   0.025164 |              0.012462 |
| downweight_dominant_center |        0.01 | test    |                     0.0252 |                   0.025164 |              0.011782 |
| equal_client               |        0.03 | test    |                     0.0252 |                   0.025163 |              0.012462 |
| downweight_dominant_center |        0.03 | test    |                     0.0246 |                   0.02456  |              0.011751 |
| equal_client               |        0.05 | test    |                     0.0246 |                   0.024594 |              0.01251  |
| downweight_dominant_center |        0.05 | test    |                     0.0242 |                   0.024193 |              0.011802 |
| equal_client               |        0.1  | test    |                     0.022  |                   0.021963 |              0.010895 |
| downweight_dominant_center |        0.1  | test    |                     0.0214 |                   0.02136  |              0.010169 |
| equal_client               |        0.2  | test    |                     0.0198 |                   0.019775 |              0.010482 |
| downweight_dominant_center |        0.2  | test    |                     0.019  |                   0.018973 |              0.009786 |

## Conservative interpretation

This is not a formal privacy guarantee and should not be described as DP validation.

It is a privacy-noise robustness probe. If equal-client or downweight-dominant weighting remains better than FedAvg-style weighting under increasing coefficient noise, then the site-signal alignment result is less fragile under privacy-like perturbation.
