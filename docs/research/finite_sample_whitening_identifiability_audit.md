# Finite-sample whitening and identifiability audit

## Scope

This is a deterministic, no-factorizer audit of the frozen 32-fit minimality × whitening result. It initializes, trains, and evaluates zero neural models. The original `complete_canonicalization_tradeoff_detected` status and its fixed 0.10 covariance criterion remain immutable. Finite-sample calibration is a new post-hoc analysis and cannot replace that criterion.

The audit separates two questions: whether observed covariance is plausible under a finite sample from a population-white code, and whether minimal dimensionality plus whitening makes the generator latent linearly identifiable. Relative covariance improvement is also kept separate from absolute finite-sample-null consistency.

## Parametric white-population calibration

For every observed `(identity count, code dimension)` pair, the audit generates 50,000 deterministic Monte Carlo samples from `N(0, I_d)`. Each sample is centered by its own sample mean and uses the same covariance estimator `XᵀX / (n - 1)` as the frozen factorial runner. Generation is chunked without changing the random stream. The predeclared pairs are `(32,32)`, `(8,32)`, `(20,32)`, `(32,8)`, `(8,8)`, and `(20,8)`.

The analytic approximation for mean absolute off-diagonal covariance is `sqrt(2 / (pi (n - 1)))`. It depends on identity count, demonstrating directly why a fixed 0.10 cutoff is not sample-size invariant.

A run/split is finite-sample-white-consistent only if its mean absolute off-diagonal covariance is below the matched 97.5th percentile, its minimum and maximum diagonals lie within their matched 2.5–97.5% limits, its numerical rank is `min(d, n - 1)`, and every reported metric is finite. The frozen fixed flag is always reported separately.

## Direct nonlinear counterexample

Let `b ~ N(0, I_8)` and define each learned coordinate as `z_j = b_j^3 / sqrt(15)`. This representation is exactly eight-dimensional. Odd symmetry gives `E[z_j] = 0`; the sixth Gaussian moment gives `Var(z_j) = E[b_j^6] / 15 = 15/15 = 1`; and coordinate independence gives identity covariance. The cube is coordinatewise bijective, with inverse `b_j = cbrt(sqrt(15) z_j)`, so biological information is preserved.

Nevertheless, `Cov(b_j,z_j) = E[b_j^4]/sqrt(15) = 3/sqrt(15)`. The population linear coefficient of determination is therefore `(3/sqrt(15))² = 9/15 = 0.60`, while the nonlinear inverse recovers `b` exactly. Minimality, identity covariance, biological information, retrieval, and nonlinear decodability can therefore coexist with failed canonical Ridge recovery.

## Objective-invariance argument

For any biological-code bijection `h`, begin with encoder `E_b(x)` and decoder `D(z,a)`. Define

`E'_b(x) = h(E_b(x))`

and

`D'(z',a) = D(h^{-1}(z'),a)`.

Substitution shows directly that self reconstruction and crossed-target reconstruction are unchanged. Scanner prototypes are untouched. Whenever same-identity views had identical biological codes, applying the same function `h` preserves that equality and hence preserves zero consistency loss. A bijection can preserve identity retrieval, and a sufficiently capable independent decoder can compose with its inverse. None of these facts requires the transformed coordinates to remain linearly related to the synthetic generator latent.

Dimensional minimality removes redundant coordinates but does not remove nonlinear bijections. Covariance whitening constrains first and second moments but does not constrain higher-order nonlinear coordinate transformations. Thus the present unsupervised objective cannot guarantee canonical linear biological coordinates. This is a direct invariance argument for this objective, not an externally attributed theorem.

## Claim boundary

The frozen factorial result remains unchanged, including failure of the original fixed whitening cutoff. Perfect retrieval and informative independent decoding remain compatible with failed Ridge recovery. Under the present objective, canonical generator-latent recovery is not an identifiable target; privileging one coordinate system would require additional structural assumptions or supervision. These synthetic conclusions do not establish a pathology-facing architecture or clinical validity.
