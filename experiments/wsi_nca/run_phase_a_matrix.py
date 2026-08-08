#!/usr/bin/env python3
"""Generate or execute the controlled WSI-NCA Phase A experiment matrix.

Dry-run is the default so the complete matched command set can be inspected and
recorded before compute is spent. Pass ``--execute`` only after the smoke run is
clean.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


TRAINER = Path(__file__).with_name("train_panda_phase_a.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSI-NCA Phase A matrix launcher")
    parser.add_argument("--manifest", default="results/panda_manifest/panda_phikon_manifest.csv")
    parser.add_argument("--out-dir", default="results/wsi_nca_phase_a")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-patches", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-shuffled", action="store_true")
    parser.add_argument("--include-embedding", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def common_command(args: argparse.Namespace, steps: int, seed: int) -> List[str]:
    command = [
        sys.executable,
        str(TRAINER),
        "--manifest",
        args.manifest,
        "--out-dir",
        args.out_dir,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-patches",
        str(args.max_patches),
        "--hidden-dim",
        str(args.hidden_dim),
        "--k-neighbors",
        str(args.k_neighbors),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--num-steps",
        str(steps),
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def build_commands(args: argparse.Namespace) -> List[List[str]]:
    commands: List[List[str]] = []
    for seed in args.seeds:
        for steps in args.steps:
            spatial = common_command(args, steps, seed)
            spatial.extend(["--neighbor-mode", "spatial", "--coordinate-control", "real"])
            commands.append(spatial)

            # T=0 does not use the graph, so topology controls are intentionally
            # omitted: they would be mathematically identical static readouts.
            if steps == 0:
                continue

            if args.include_shuffled:
                shuffled = common_command(args, steps, seed)
                shuffled.extend(
                    ["--neighbor-mode", "spatial", "--coordinate-control", "shuffle"]
                )
                commands.append(shuffled)

            if args.include_embedding:
                embedding = common_command(args, steps, seed)
                embedding.extend(
                    ["--neighbor-mode", "embedding", "--coordinate-control", "real"]
                )
                commands.append(embedding)

    return commands


def main() -> None:
    args = parse_args()
    commands = build_commands(args)

    print(f"Prepared {len(commands)} matched Phase A runs.")
    for index, command in enumerate(commands, start=1):
        print(f"[{index:02d}/{len(commands):02d}] {' '.join(command)}")

    if not args.execute:
        print("Dry run only. Re-run with --execute after the smoke test is clean.")
        return

    for index, command in enumerate(commands, start=1):
        print(f"\n=== Executing run {index}/{len(commands)} ===", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
