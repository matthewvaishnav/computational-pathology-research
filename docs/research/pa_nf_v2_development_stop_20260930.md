# PA-NF v2 development stop — 2026-09-30

## Status

The crossed-intervention v2 development line is **stopped** after the frozen R5 smoke evaluation. The predeclared R5 decision rule was followed: if R5 failed the unchanged promotion gate, no R6 or further tuning would be performed on this synthetic development set.

This is a development-negative result. It does not invalidate the earlier PA-NF v1 evidence or the positive mechanism signals observed during v2 development.

## R5 frozen architecture

R5 replaced the unconstrained concatenative decoder with a canonical-biology plus bounded acquisition-operator decoder while retaining R4 donor-invariant same-scanner pooling for crossed acquisition state.

The frozen candidate/control comparison used identical architecture and exact parameter count; the only candidate-specific training terms were crossed reconstruction plus biological/acquisition cycle losses.

Validated parameter count per model: **221,920**.

Fresh R5 smoke seeds: **3501, 3502, 3503** across linear and nonlinear synthetic renderers.

## R5 result

Promotion summary:

- candidate all-seed crossed-factorization success: **true**
- candidate mean acquisition-transfer delta greater than control: **true**
- candidate mean biology-retention delta greater than control: **false**
- candidate mean variance-normalized cycle error lower than control: **true**
- parameter counts equal: **true**
- development promotion pass: **false**

Paired candidate-minus-control contrasts:

- acquisition-transfer delta: **+0.01833755026261012**
- biology-retention delta: **-0.00794970989227295**
- control-minus-candidate variance-normalized cycle error: **+0.11060189704100291**

## Development trajectory

The successive development iterations localized the failure:

- **R1:** crossed factorization worked and acquisition transfer improved, but biology retention was worse than the matched control.
- **R2:** same-scanner acquisition consistency and split cycle losses improved cycle fidelity, but the biology-retention deficit remained.
- **R3:** a shared biological contrastive-weight grid did not materially move the candidate-control biology-retention gap, arguing against insufficient biological-pressure as the explanation.
- **R4:** donor-invariant acquisition pooling more than halved the biology-retention deficit while improving acquisition transfer and cycle fidelity, showing that donor leakage was real but incomplete as an explanation.
- **R4 gradient-conflict audit:** the predeclared rule did not support systematic negative gradient transfer as the primary explanation. Overall candidate-specific-vs-retention cosine mean was approximately +0.00196 with a negative fraction of 0.4667; crossed-vs-retention cosine mean was approximately +0.00080 with the same negative fraction.
- **R5:** constraining acquisition to a bounded FiLM-style operator further reduced the biology-retention deficit but did not reverse it.

## Scientific interpretation boundary

The v2 line supports several positive mechanism observations:

1. crossed-intervention training can improve acquisition transfer relative to an exact-capacity control;
2. donor-invariant pooling materially reduces biological leakage;
3. cycle fidelity can improve without resolving biological allocation;
4. the remaining retention deficit is not well explained by a simple systematic gradient-conflict mechanism.

But the line does **not** establish a PA-NF-specific biological-retention advantage over the exact-capacity control. The unchanged promotion gate therefore failed.

## Stop rule

Do not:

- create R6 on this synthetic development set;
- retune R5 coefficients, widths, or thresholds against these outcomes;
- relax the biology-retention gate;
- reinterpret the positive acquisition-transfer/cycle results as overall promotion.

The next development line must use a **fundamentally different representation hypothesis** and a **new synthetic benchmark**.

## Next hypothesis

The next candidate is operator-defined canonicalization rather than two independently learned content channels:

\[
\theta(x)=E_a(x), \qquad u(x)=T_{-\theta(x)}x.
\]

The biological representation is therefore defined as the observation after inverse acquisition transport. For paired acquisitions of the same region:

\[
T_{\theta_t-\theta_s}(x_s) \approx x_t,
\]

\[
u(x_s) \approx u(x_t).
\]

The acquisition family must have an explicit identity, inverse, and composition law. This is a new representation hypothesis and must be tested on fresh synthetic evidence before any new real-pathology confirmation.