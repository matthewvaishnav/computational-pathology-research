# Preparation/workflow metadata readiness audit

This package determines whether candidate datasets expose enough explicit provenance to support the next crossed-preparation research program before any download, model training, or representation analysis.

Target question:

> After controlling tissue identity, does scanner-suppressed representation structure track preparation or post-preparation workflow identity across scanners?

This is a metadata-feasibility audit, not an experimental result. It does not establish preparation effects, workflow effects, scanner invariance, causal attribution, biological validity, diagnostic performance, or clinical relevance.

## Files

- `metadata_requirements.md`: required identities and process provenance.
- `dataset_candidate_registry.csv`: fail-closed candidate inventory.
- `readiness_rules.md`: tier and contrast-specific decision rules.
- `audit_dataset_candidates.py`: deterministic standard-library audit.
- `readiness_report.md`: checked report for the registry.
- `public_dataset_discovery.md`: primary-source dataset search, candidate ranking, and bounded PLISM feasibility decision.
- `acquisition_plan.md`: minimum prospective collection plan when no existing dataset is confirmatory-ready.

## Commands

```powershell
python benchmarks\preparation_workflow_metadata_readiness\audit_dataset_candidates.py
python benchmarks\preparation_workflow_metadata_readiness\audit_dataset_candidates.py --self-test
python benchmarks\preparation_workflow_metadata_readiness\audit_dataset_candidates.py --check-report
python benchmarks\preparation_workflow_metadata_readiness\audit_dataset_candidates.py --format json
```

Custom input does not overwrite the checked report:

```powershell
python benchmarks\preparation_workflow_metadata_readiness\audit_dataset_candidates.py --input <registry.csv>
```

## Current decision

No inspected public dataset satisfies every confirmatory-readiness field. PLISM is the strongest feasibility candidate because it combines serial-section preparation variation, repeated scanner domains, original WSIs, and registered coordinates. It remains exploratory until missing batch, order, physical-device, section-hierarchy, and immutable source-event provenance is recovered.

## Boundaries

- `unknown` means the audit found no explicit evidence in the cited source; it does not mean the underlying factor was absent.
- `inferred` metadata never counts as confirmatory support.
- A site label is not a post-preparation workflow definition.
- A scanner model name is not same-section paired-scanner provenance.
- Scanner suppression is not evidence of biological validity.
- Confirmatory readiness is granted only when every contrast-specific required field is explicit, available, and supported by verified evidence.
