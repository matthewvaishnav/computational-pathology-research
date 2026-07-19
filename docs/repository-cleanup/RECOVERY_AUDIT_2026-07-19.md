# Repository recovery audit and cleanup proposal

Audit date: 2026-07-19  
Repository: `matthewvaishnav/computational-pathology-research`  
Audit branch: `cleanup/repository-recovery-audit-20260719`

## Decision summary

Treat `main` as the recovered source of truth, preserve
`rescue/pre-recovery-20260719` as immutable recovery evidence, and do not
delete paired-acquisition experiment branches until their raw result artifacts
have a permanent consolidated home.

No scientific result, manuscript, result table, child repository, or recovery
reference is changed by this audit.

## 1. Recovery boundary

| Role | Ref | Commit | Decision |
|---|---|---|---|
| Recovered source of truth | `main` | `932c743595599a5de1446e7d87bbd17050871d4f` | Keep |
| Last shared pre-divergence point | — | `b1a6a03a15c7c933d57b8fa6b94b9ae8931d7a9f` | Evidence boundary |
| Pre-recovery safety snapshot | `rescue/pre-recovery-20260719` | `8f0a223b333255f6402c666a0946c43f23294a29` | Preserve; never rewrite |

The rescue ref is three commits ahead of the shared point and one commit behind
the recovered `main`. Its 25 divergent paths comprise:

- 10 crossed-preparation-identifiability files;
- 7 preparation-workflow metadata-readiness files;
- 8 files from the abandoned broad TransnnMIL repair.

The recovered `main` contains the intentionally narrow four-file TransnnMIL
salvage from PR #46:

- `src/models/transnnmil/__init__.py`
- `src/models/transnnmil/branch_token_fusion.py`
- `src/models/transnnmil/transnnmil_branch_token.py`
- `tests/models/test_transnnmil_fusion_salvage.py`

## 2. Branch inventory

There were 26 branches at audit time, including `main`.

### Preserve as repository infrastructure

- `main`
- `gh-pages`
- `rescue/pre-recovery-20260719`
- `cleanup/repository-recovery-audit-20260719`

### Preserve as paired-acquisition evidence

These branches contain unique experiment runners or result directories that are
absent from `main` and are referenced by the claim ledger or figure/table
manifest:

- `experiment/acquisition-bottleneck-separation-frontier`
- `experiment/acquisition-branch-audit`
- `experiment/acquisition-factor-swapping-audit`
- `experiment/biological-label-preservation-audit`
- `experiment/frontier-selected-crossbackbone-validation`
- `experiment/frontier-selected-downstream-validation`
- `experiment/linear-baseline-consistency-audit`
- `experiment/linear-residual-branch-separation-audit`
- `experiment/oldstyle-residual-branch-separation-audit`
- `experiment/pair-structure-boundary-crossbackbone`
- `experiment/pair-structure-boundary-test`
- `experiment/sample-subset-disjoint-scanner-heldout-transfer-audit`
- `experiment/scanner-confounded-label-robustness-audit`
- `experiment/scanner-heldout-label-transfer-audit`
- `experiment/unified-separation-scoreboard`

### Quarantine for manual recovery review

Do not merge these into `main`, and do not delete them until their relationship
to the rescue snapshot is documented:

- `research/crossed-preparation-identifiability-design`
- `research/fix-transnnmil-branch-fusion`
- `research/panda-fusion-controlled-evaluation`
- `research/preparation-workflow-metadata-readiness`
- `research/public-dataset-provenance-discovery`

The PANDA and public-dataset branches include unique unmerged material beyond
the rescue snapshot. They require an explicit keep-or-discard decision.

### Proven redundant; approval-ready for deletion

- `experiment/claim-ledger-and-paper-skeleton`: zero commits ahead of
  `main`, eight behind.
- `fix/detector-transfer-table-overlap`: its only changed blob,
  `paper/arxiv/broader_research_program.tex`, is byte-identical to `main`.
- `research/transnnmil-fusion-salvage`: all four changed blobs are
  byte-identical to the four-file salvage currently on `main`.

No branch deletion has been performed.

## 3. Artifact classification

| Class | Treatment |
|---|---|
| Verified current code and tests on `main` | Keep |
| Paired-acquisition claim sources and raw result tables | Keep until consolidated |
| `rescue/pre-recovery-20260719` | Preserve as immutable evidence |
| Broad recovery-era crossed-preparation, preparation-readiness, and PANDA work | Quarantine; no merge |
| Three proven redundant branches | Delete only after approval |
| `paper/arxiv/build/main.tex` | Approval-ready to remove from tracking; `build/` is already ignored |
| Child repositories, releases, DOI metadata, and committed result artifacts | Do not touch |

## 4. Paired-acquisition claim trace

All 15 commits named by
`paper/claim_ledger/result_to_claim_map.csv` are reachable and contain the
expected experiment runner and result artifacts.

| Claim(s) | Commit | Evidence branch | Primary result directory |
|---|---|---|---|
| 2 | `3e5bf19e420abfcb9a95c517407cba636b9f126d` | `experiment/acquisition-branch-audit` | `results/paired_acquisition_factorization_acquisition_branch_audit` |
| 1, 2 | `e4819c42e49f9c4a1e7a652fc8bf8651a2f6b628` | `experiment/pair-structure-boundary-test` | `results/paired_acquisition_factorization_pair_structure_boundary_test` |
| 1 | `d018c924757c90a56ab5d515c4ecc02110286df6` | `experiment/pair-structure-boundary-crossbackbone` | `results/paired_acquisition_factorization_pair_structure_boundary_crossbackbone` |
| 1, 2 | `bec06eb47e14ca3317efae7b74c3399615aa0d13` | `experiment/biological-label-preservation-audit` | `results/paired_acquisition_factorization_biological_label_preservation_audit` |
| 1, 2 | `535eea183ab7471bf661dd1abf122868f1ff56ac` | `experiment/scanner-heldout-label-transfer-audit` | `results/paired_acquisition_factorization_scanner_heldout_label_transfer_audit` |
| 1, 2 | `0d7cdc927619f38006c79069b5cbb087a27241e4` | `experiment/sample-subset-disjoint-scanner-heldout-transfer-audit` | `results/paired_acquisition_factorization_sample_subset_disjoint_scanner_heldout_transfer_audit` |
| 1, 2 | `b5a9886e77776942d133929c1592d1203229aa8c` | `experiment/scanner-confounded-label-robustness-audit` | `results/paired_acquisition_factorization_scanner_confounded_label_robustness_audit` |
| 2, 3 | `ec2a509f19e5c4a1fb59c3e49486a022db03191c` | `experiment/linear-baseline-consistency-audit` | `results/paired_acquisition_factorization_linear_residual_branch_separation_audit` |
| 3 | `a325c009254bd2c220297e482203dd9a252728a6` | `experiment/linear-baseline-consistency-audit` | `results/paired_acquisition_factorization_linear_baseline_consistency_audit` |
| 2, 3 | `3450ede25d374495ea73fe2d41b1e365e8b5b5a7` | `experiment/oldstyle-residual-branch-separation-audit` | `results/paired_acquisition_factorization_oldstyle_residual_branch_separation_audit` |
| 4 | `a89bfb32977dc723ef895f150ab4ae720a345ac5` | `experiment/acquisition-bottleneck-separation-frontier` | `results/paired_acquisition_factorization_acquisition_bottleneck_separation_frontier` |
| 4 | `c29a038debf9df709d6ce10ad4c63e510556b083` | `experiment/frontier-selected-downstream-validation` | `results/paired_acquisition_factorization_frontier_selected_downstream_validation` |
| 4 | `0e2af24730a0a298fbf0363dfbab7682dc65a1af` | `experiment/frontier-selected-crossbackbone-validation` | `results/paired_acquisition_factorization_frontier_selected_crossbackbone_validation` |
| 2, 3, 4 | `1c5276978d2d5e83619c88684c9e02df3688d6e3` | `experiment/unified-separation-scoreboard` | `results/paired_acquisition_factorization_unified_separation_scoreboard` |
| 5 | `aa8d0596dfe5f4be650d034ed80fca76fc337116` | `experiment/acquisition-factor-swapping-audit` | `results/paired_acquisition_factorization_acquisition_factor_swapping_audit` |

The claim trace is intact, but it is branch-dependent. The current provenance
manifest also reports 350 unresolved metadata-lineage conflicts and no
historical output-hash binding for the 426 local archives. Therefore evidence
branches and result artifacts must not be removed as ordinary cleanup.

## 5. Exact cleanup proposal

### Approval-ready actions

1. Remove `paper/arxiv/build/main.tex` from Git tracking on this cleanup
   branch. The source `paper/arxiv/main.tex` remains; `build/` is already
   ignored.
2. Delete only these three proven redundant remote branches:
   - `experiment/claim-ledger-and-paper-skeleton`
   - `fix/detector-transfer-table-overlap`
   - `research/transnnmil-fusion-salvage`
3. Keep the recovery audit document as the decision record.

### Blocked until evidence consolidation

Do not delete any paired-acquisition evidence branch. First create a permanent
evidence archive containing the runner, design, raw metrics, summaries, reports,
run logs, full commit SHA, and checksum for every claim source. Re-run the
claim/figure validation against that archive before retiring any evidence ref.

### Explicitly excluded

- no rewrite, force-push, or merge to `main`;
- no change to `rescue/pre-recovery-20260719`;
- no result-table or manuscript-number edits;
- no child-repository changes;
- no release, tag, DOI, Zenodo, issue-state, or GitHub Pages changes;
- no merge of quarantined ChatGPT/recovery-era work.

## Approval gate

The audit is complete. The approval decision is limited to the three redundant
branch deletions and removal of the generated tracked build file. All other
refs and artifacts remain unchanged.
