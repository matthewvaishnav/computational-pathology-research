# Required provenance evidence

## Evidence hierarchy

Strongest to weakest:

1. Exact historical content hash matched to a known artifact in a verified run
   manifest.
2. Archive-specific generator configuration or deterministic run record that
   uniquely identifies the invocation and output.
3. Archive-specific run log containing output path, model, backbone, fold,
   seed, condition, variant, evaluation split, run ID, and historical output
   hash.
4. Source commit and deterministic code path combined with archive-specific
   invocation evidence that uniquely identifies the output.
5. Exact deterministic internal proof uniquely identifying the generating
   metadata.
6. Internal metadata consistency without unique invocation proof.
7. Path/family-derived expectation.
8. Filename-only inference.

Only levels 1-5 may support `corrected`. A source commit and deterministic code
path are sufficient only when an archive-specific record uniquely identifies
the invocation and output. Code that could normally produce an archive,
aggregate run completion, family membership, path reconstruction, or filename
inference is insufficient by itself.

Current-state content hashes are fingerprints of present local bytes. They are
not historical output hashes unless a verified historical record independently
binds the same digest to a producing invocation.

## Family-level reconstructive evidence

The builder verifies these concrete Git objects without checking them out or
modifying the working tree:

| In-scope archives | Commit | Generator/result unit |
|---:|---|---|
| 75 | `43520b86210d0bfed8a2869d514639af6ce8e15a` | canine pair-integrity |
| 25 | `f435bfa28c438588df5bee53bb3e5843e1d3b0d8` | SCORPION DINOv2 pair-integrity |
| 50 | `14726b13e7c0f23f9fe494399bab9fd902fecd7a` | SCORPION Phikon/ResNet50 pair-integrity |
| 75 | `e4819c42e49f9c4a1e7a652fc8bf8651a2f6b628` | canine pair-structure boundary |
| 50 | `a89bfb32977dc723ef895f150ab4ae720a345ac5` | canine acquisition bottleneck |
| 150 | `0e2af24730a0a298fbf0363dfbab7682dc65a1af` | SCORPION cross-backbone bottleneck |

The commits, generators, designs, and logs are concrete and reachable, but the
current references operate at family level. They do not contain present archive
hashes, archive IDs, row-specific evidence selectors, or verified historical
output identities. They support lineage-derived proposed values but do not
adjudicate any current conflict.

Six reused family-level reference bundles are acceptable supporting context.
They must be labeled reconstructed family-level lineage evidence and cannot be
treated as archive-specific producing-invocation proof.

## Local-untracked contextual evidence

The separately supplied file
`results/external_multiscanner_caninescc/features/fold_0_dinov2_base.summary.json`
is local-untracked context: local-only, untracked, non-adjudicating
present-workspace context whose current-state availability is checked directly.
Clean-clone availability is not assumed.

This file supports only the bounded `legacy-optional` context for
`canine_original_dinov2`. The archive has embedded extraction parameters, lacks
optional explicit backbone metadata without contradiction, and does not assert
a canonical backbone. The adjacent summary is insufficient for `confirmed`,
`corrected`, historical origin, historical byte identity, a producing
invocation, or archive-specific adjudication. Historical origin remains
unverified and archive-specific adjudication is absent.

Deterministic checking therefore requires all 426 source archives, every
reachable Git commit:path reference, and this separately supplied
local-untracked file. If the file is absent or no longer satisfies the bounded
local-context rules, checking fails closed.

## Evidence inventory

- Archives with scalar metadata JSON: 426.
- Derived archives with structured embedded configuration: 425.
- Archives with associated family-level run logs: 425.
- Archives with family-level source/result commit associations: 425.
- Archives with a producing repository commit embedded in metadata: 0.
- Archives with archive-specific historical output-identity binding: 0.
- Conflicts with archive-specific adjudication evidence: 0.
- Unresolved conflicts with lineage-derived proposed values: 350.
- Archives lacking historical cryptographic output binding: 426.
- Archives with computable current-state raw hashes: 426.

Upstream foundation-model revision values are not producing repository commit
references. Reachability of a generator/result commit does not show that the
present NPZ was produced by that exact code snapshot or invocation.

## Evidence still needed

Each unresolved row records the need for a verified artifact or run manifest
that uniquely links:

- archive path and archive-specific run ID;
- exact producing command or invocation;
- fold, seed, condition, variant, and evaluation split;
- model, backbone, and source feature identity;
- exact producing source commit and generator configuration;
- historical output hash matching the archived bytes; and
- dataset and source-feature manifest hashes.

For a cryptographically closed lineage chain, all 426 source NPZs or immutable
object locations must also be frozen. None of the 426 NPZ paths is Git-tracked,
so this package does not establish public artifact availability or clean-clone
execution without separately supplied archives, reachable Git commit:path
references, and the local-untracked context file named above.

This package does not require or run new training. Crossed preparation, site,
processing, and stain-batch evidence remains a separate future study-design
requirement.
