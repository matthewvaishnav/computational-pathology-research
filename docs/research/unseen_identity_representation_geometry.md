# Unseen-identity representation-geometry diagnostic

## Scientific scope

The original unseen-identity generalization gate remains closed. All eight
crossed-target prototype runs passed operational two-axis counterfactual
transfer, but only three passed full factor allocation because the frozen linear
biological-latent recovery threshold was not met consistently. This post-hoc
diagnostic asks what geometry is present in those representations; it is not a
replacement confirmatory campaign and cannot retroactively change the primary
result.

The diagnostic remains synthetic evidence. It is not pathology-domain
validation, and it does not establish behavior on tissue images, clinical
cohorts, sites, scanners, or endpoints.

## Frozen inputs and replication

The diagnostic requires the original
`unseen_identity_generalization_result.json`. Before training it verifies the
unseen-identity v1 schema, dataset seeds 4301 and 5301, model seeds 2201 and
2202, both renderers, eight crossed-target runs, successful two-axis transfer in
each crossed-target run, and a closed primary gate. The input file SHA-256 is
recorded and checked again after all fits.

Only `crossed_target_prototype` and `oracle_supervised` are rerun, giving the
fixed 16-fit smoke grid. The dataset, identity partition, architecture, epochs,
loss weights, and determinism settings are loaded from the reference result.
For crossed-target fits, biology-retention delta, acquisition-transfer delta,
two-axis identity success, biological-to-biological Ridge R², and
biological-to-acquisition Ridge R² must match the reference within
`1e-6 + 1e-5 * abs(reference)`. A mismatch fails closed before an interpretation
artifact is written.

## Probe protocol

For each dataset seed and renderer, training identities are deterministically
split 80/20 into probe-training and probe-validation identities. Every scanner
observation from one identity remains in one partition. Scalers are fit only on
probe-training observations, validation is used only for early stopping, and
final metrics are computed only on the entirely unseen test identities. Exact
identities, observation indices, and SHA-256 hashes are recorded.

The frozen linear baseline is standardized Ridge with `alpha=1e-3`; its original
`R² >= 0.80` interpretation remains unchanged. The nonlinear regressors and
scanner classifier use two GELU hidden layers of width 32, AdamW, at most 300
full-batch epochs, and validation early stopping with patience 30. There is no
hyperparameter sweep. A nonlinear-probe success shows transferable nonlinear
coordinate preservation; it does not show recovery of the exact canonical
latent basis.

## Predeclared interpretation criteria

- Nonlinear biological recovery: unseen-test variance-weighted `R² >= 0.80`.
- Material nonlinear improvement: at least `+0.05 R²` over frozen Ridge.
- Linear and nonlinear scanner exclusion: balanced accuracy no greater than
  scanner chance plus `0.10`.
- Acquisition biological exclusion: nonlinear acquisition-to-biology
  `R² <= 0.10`.
- Cross-scanner retrieval: unseen-identity top-1 at least `0.90`.
- Independent diagnostic decoding: biological-code-plus-scanner-prototype MSE,
  normalized by unseen target observation mean-square, must be no greater than
  `0.50` and no more than `0.10` above the corresponding decoder supplied with
  the true biological latent.

The known synthetic acquisition latent, indexed by the true scanner label, is
the true scanner prototype. A learned scanner prototype is computed separately
from the model acquisition branch on probe-training identities. For the
crossed-target family the latter reproduces its scanner-indexed embedding; for
the oracle it is the probe-training scanner centroid. This prevents
identity-specific acquisition observations from entering an ordered transfer.

The diagnostic decoder is separate from the trained factorizer. It never
updates the original encoder, acquisition branch, prototype embedding, or model
decoder. Three fixed shallow decoders receive (1) learned biological code plus
the known true scanner prototype, (2) biological code alone, or (3) true
biological latent plus the learned scanner prototype. They are trained only on
probe-training identities and evaluated on all ordered scanner transfers of
unseen identities.

Strong nonlinear biology recovery together with scanner exclusion, acquisition
exclusion, retrieval, and independent decoding supports a nonlinear but
transferable representation. Strong original-decoder transfer with weak probes
and weak independent decoding suggests decoder-dependent coding. Nonlinear
scanner recovery above the exclusion bound indicates hidden scanner leakage.
A nonlinear biology probe does not count as support when nonlinear scanner
leakage is also present.

The global nonlinear-transferable status requires every crossed-target run to
meet all of those criteria and to replicate its reference metrics. The oracle
must pass the same representation checks as a positive control. Otherwise the
result is reported as decoder-dependent, hidden-leakage, mixed, or
`diagnostic_failed`; no status reopens the primary campaign.

## Planned smoke invocation

```powershell
py -m experiments.paired_acquisition.run_unseen_identity_representation_geometry `
  --mode smoke `
  --device cuda `
  --reference-result "C:\Users\matth\computational-pathology-research\results\unseen_identity_crossed_generalization_smoke_20260802T101904\unseen_identity_generalization_result.json" `
  --output-root "C:\Users\matth\computational-pathology-research\results\unseen_identity_representation_geometry_smoke_<timestamp>"
```

The output directory must not already exist. The runner writes the result JSON,
heterogeneous summary CSV, and manifest atomically and records a canonical
internal result hash.
