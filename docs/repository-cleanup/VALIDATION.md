# Cleanup validation gates

PR #7 remains draft until the cleanup branch satisfies all required checks on a normal, non-bot-authored commit.

## Required before merge

- Repository audit and Python syntax check pass.
- Quick Check core imports and fast tests pass.
- CI lint gates pass with the project-declared Black range and isort.
- Internal and standalone security scans pass.
- Full test matrix and Docker build complete successfully.
- Paired-Acquisition Neural Factorization reproducibility evidence remains present and unchanged except for mechanical formatting.
- The paper builds and deploys to the canonical URL `computational-pathology-research-platform-for-federated-oncology.pdf`; previous paper filenames may remain available only as compatibility aliases.

## Completed repairs

- Restored the truncated streaming cache and replaced executable cache deserialization.
- Corrected package identity and removed unsupported production claims.
- Separated optional foundation/integration collection from the fast test tier.
- Broke the TransnnMIL model-factory circular import and added regression coverage.
- Implemented security environment detection, including research-mode support.
- Restored the pinned model-download manager contract and dedicated security exception.
- Applied repository-wide Black and isort formatting without weakening either gate.
