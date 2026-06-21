# CAMELYON17 PathoAlign v3 stronger adversary smoke

Result: negative diagnostic.

The nuisance-center loss wiring fix succeeded: nuisance representations became strongly center-decodable.

However, the stronger adversary grid did not produce the desired PathoAlign separation pattern.

Observed:
- original center mean: about 0.917
- teacher diagnostic center mean: about 0.857
- cleaned feature center mean: about 0.915 to 0.923
- diagnostic representation center mean: about 0.879 to 0.906
- nuisance tumor AUC remained high, usually about 0.98 to 0.99

Interpretation:
- nuisance branch now learns center/source identity
- cleaned features do not lose center/source identity
- nuisance branch remains tumor-predictive
- this is not a candidate for full confirmatory scaling

Next hypothesis:
The decoded nuisance component is not aligned with the removable center direction in original feature space, even when the low-dimensional nuisance code predicts center.
