# Research outreach package

This directory packages the dominant-site federated pathology work for readers who do not want to start with the full paper.

## Read order

1. [Plain-English one-page summary](plain-english-summary.md) - for recruiters, non-specialist scientists, journalists, and first-contact outreach.
2. [Technical one-page summary](technical-summary.md) - for ML researchers, computational pathology labs, and applied ML hiring teams.
3. [Demo / figure thread](demo-figure-thread.md) - short post/thread copy for LinkedIn, X, BlueSky, or email.
4. [Outreach list](outreach-list.md) - target labs, researchers, startups, companies, and journalist categories.

## Main PDF site

The public site is intentionally reduced to the paper PDF. The active deploy workflow builds the LaTeX paper and publishes a PDF-only GitHub Pages site.

Expected public site:

```text
https://matthewvaishnav.github.io/computational-pathology-research/
```

Expected PDF path:

```text
https://matthewvaishnav.github.io/computational-pathology-research/when-more-data-is-less-trustworthy.pdf
```

## Claim boundary

This package is for research and employment outreach. It must not be framed as clinical validation, diagnostic software, hospital deployment evidence, or regulatory readiness. The supported claim is narrower: in these simulated federated pathology experiments over real pathology-derived features, sample count is not equivalent to task-specific site-signal alignment, and sample-size dominance should be audited rather than assumed safe.
