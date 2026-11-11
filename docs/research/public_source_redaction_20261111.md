# Public source portability and redaction note — 11 November 2026

This maintenance change removes assistant-client references, machine-specific Windows home directories, and local execution-environment paths from the public **working tree**. It replaces operational absolute paths with repository-relative or explicitly supplied paths, and redacts non-scientific path fields in previously published textual execution logs and result tables.

**Scientific scope:** Numeric metrics, statistical conclusions, dataset identity, checkpoint gates, original experiment selections, and recorded comparison decisions were not intentionally altered. Redacted historical log and CSV files are *publication-safe derived text copies*, **not byte-identical reproductions of the original files**. Their SHA-256 digests have necessarily changed. Any scientific provenance system that verifies the original raw log bytes must use the independently archived original artifacts, not these redacted working-tree copies. Do not silently replace a frozen reference hash with the redacted file's new hash.

Revised scripts accept repository-relative locations or explicit external paths rather than embedded personal directories. The independent copied-result verification path is optional and remains SHA-checked when explicitly supplied.

This change does **not** rewrite historical commits, GitHub forks, cached copies, existing releases, or earlier pull-request branches. Those surfaces are separate from the current default-branch working tree. Do not force-push or rewrite shared research history without a separately reviewed migration plan and full evidence custody inventory.

No neural-network weights, checkpoints, private TIFFs, or patient-level source data are introduced by this maintenance change.
