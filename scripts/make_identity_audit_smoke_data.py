#!/usr/bin/env python3
"""Create a small synthetic dataset for PathoAlign Identity Audit smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Create synthetic features/metadata for identity-audit smoke testing.")
    parser.add_argument("--out", type=Path, default=Path("tmp/identity_audit_smoke"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--scanners", type=str, default="scanner_a,scanner_b")
    parser.add_argument("--n-features", type=int, default=16)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    scanners = [s.strip() for s in args.scanners.split(",") if s.strip()]
    if len(scanners) < 2:
        raise SystemExit("Need at least two scanners for the smoke test.")

    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    features = []
    for sample_idx in range(args.n_samples):
        sample_id = f"sample_{sample_idx:03d}"
        biology_label = "tumor_like" if sample_idx % 2 else "benign_like"
        biological_vector = rng.normal(0, 1.0, size=args.n_features)
        for scanner_idx, scanner_id in enumerate(scanners):
            unit_id = f"{sample_id}_{scanner_id}"
            scanner_shift = np.zeros(args.n_features)
            scanner_shift[0] = 1.5 if scanner_idx == 0 else -1.5
            scanner_shift[1] = 0.75 if scanner_idx == 0 else -0.75
            x = biological_vector + scanner_shift + rng.normal(0, 0.20, size=args.n_features)
            rows.append({
                "unit_id": unit_id,
                "sample_id": sample_id,
                "region_id": sample_id,
                "scanner_id": scanner_id,
                "site_id": "site_0" if scanner_idx == 0 else "site_1",
                "client_id": "client_0" if scanner_idx == 0 else "client_1",
                "biology_label": biology_label,
            })
            features.append({"unit_id": unit_id, **{f"f{i}": float(x[i]) for i in range(args.n_features)}})

    pd.DataFrame(features).to_csv(args.out / "features.csv", index=False)
    pd.DataFrame(rows).to_csv(args.out / "metadata.csv", index=False)
    print(f"Wrote {args.out / 'features.csv'}")
    print(f"Wrote {args.out / 'metadata.csv'}")
    print("Run:")
    print(f"python scripts/pathoalign_identity_audit.py --features {args.out / 'features.csv'} --metadata {args.out / 'metadata.csv'} --out {args.out / 'audit'} --block-column sample_id")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
