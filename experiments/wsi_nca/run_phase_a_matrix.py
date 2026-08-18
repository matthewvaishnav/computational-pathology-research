#!/usr/bin/env python3
"""Generate or execute the controlled WSI-NCA Phase A experiment matrix.

Dry-run is the default so the complete matched command set can be inspected and
recorded before compute is spent. ``--scientific-causal-ladder`` emits the
frozen post-calibration PANDA protocol: T0, T1, T4 tied with real+shuffled test
evaluation from one checkpoint, T4 shuffled-train/shuffled-test, and T4 untied.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TRAINER = Path(__file__).with_name("train_panda_phase_a.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSI-NCA Phase A matrix launcher")
    parser.add_argument(
        "--manifest",
        default="results/wsi_nca_phase_a/panda_coordinate_manifest.csv",
        help="Frozen readable coordinate manifest prepared before matched controls",
    )
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
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-shuffled", action="store_true")
    parser.add_argument("--include-embedding", action="store_true")
    parser.add_argument(
        "--include-real-shuffled-eval-t4",
        action="store_true",
        help=(
            "For tied real-topology T=4, evaluate the selected checkpoint on "
            "both real and shuffled held-out test topology without retraining."
        ),
    )
    parser.add_argument(
        "--include-untied-t4",
        action="store_true",
        help="Add the real-topology untied T=4 recurrent/GNN control",
    )
    parser.add_argument(
        "--scientific-causal-ladder",
        action="store_true",
        help=(
            "Emit the frozen post-calibration PANDA causal ladder. This mode "
            "uses T0, T1, T4 tied real->(real+shuffle), T4 tied "
            "shuffle->shuffle, and T4 untied real->real for every seed."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def common_command(
    args: argparse.Namespace,
    steps: int,
    seed: int,
    dynamics_mode: str = "tied",
) -> list[str]:
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
        "--val-fraction",
        str(getattr(args, "val_fraction", 0.2)),
        "--test-fraction",
        str(getattr(args, "test_fraction", 0.2)),
        "--split-seed",
        str(getattr(args, "split_seed", 42)),
        "--num-steps",
        str(steps),
        "--seed",
        str(seed),
        "--dynamics-mode",
        dynamics_mode,
        "--device",
        args.device,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def configured_command(
    args: argparse.Namespace,
    steps: int,
    seed: int,
    *,
    dynamics_mode: str = "tied",
    neighbor_mode: str = "spatial",
    coordinate_control: str = "real",
    eval_coordinate_control: str = "match",
) -> list[str]:
    command = common_command(args, steps, seed, dynamics_mode=dynamics_mode)
    command.extend(
        [
            "--neighbor-mode",
            neighbor_mode,
            "--coordinate-control",
            coordinate_control,
            "--eval-coordinate-control",
            eval_coordinate_control,
        ]
    )
    return command


def build_scientific_causal_ladder(args: argparse.Namespace) -> list[list[str]]:
    """Build the frozen five-training-run ladder representing six comparisons."""
    commands: list[list[str]] = []
    for seed in args.seeds:
        commands.append(
            configured_command(
                args,
                0,
                seed,
                coordinate_control="real",
                eval_coordinate_control="real",
            )
        )
        commands.append(
            configured_command(
                args,
                1,
                seed,
                coordinate_control="real",
                eval_coordinate_control="real",
            )
        )
        commands.append(
            configured_command(
                args,
                4,
                seed,
                coordinate_control="real",
                eval_coordinate_control="both",
            )
        )
        commands.append(
            configured_command(
                args,
                4,
                seed,
                coordinate_control="shuffle",
                eval_coordinate_control="shuffle",
            )
        )
        commands.append(
            configured_command(
                args,
                4,
                seed,
                dynamics_mode="untied",
                coordinate_control="real",
                eval_coordinate_control="real",
            )
        )
    return commands


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    if getattr(args, "scientific_causal_ladder", False):
        return build_scientific_causal_ladder(args)

    commands: list[list[str]] = []
    for seed in args.seeds:
        for steps in args.steps:
            eval_control = "match"
            if steps == 4 and getattr(args, "include_real_shuffled_eval_t4", False):
                eval_control = "both"

            commands.append(
                configured_command(
                    args,
                    steps,
                    seed,
                    coordinate_control="real",
                    eval_coordinate_control=eval_control,
                )
            )

            # T=0 does not use the graph, so topology controls are intentionally
            # omitted: they would be mathematically identical static readouts.
            if steps == 0:
                continue

            if args.include_shuffled:
                commands.append(
                    configured_command(
                        args,
                        steps,
                        seed,
                        coordinate_control="shuffle",
                        eval_coordinate_control="shuffle",
                    )
                )

            if args.include_embedding:
                commands.append(
                    configured_command(
                        args,
                        steps,
                        seed,
                        neighbor_mode="embedding",
                        coordinate_control="real",
                        eval_coordinate_control="real",
                    )
                )

            if steps == 4 and args.include_untied_t4:
                commands.append(
                    configured_command(
                        args,
                        steps,
                        seed,
                        dynamics_mode="untied",
                        coordinate_control="real",
                        eval_coordinate_control="real",
                    )
                )

    return commands


def main() -> None:
    args = parse_args()
    commands = build_commands(args)

    print(f"Prepared {len(commands)} matched Phase A runs.")
    if args.scientific_causal_ladder:
        print(
            "Scientific causal ladder: each tied real T=4 run evaluates the same "
            "selected checkpoint on real and shuffled held-out test topology."
        )
    for index, command in enumerate(commands, start=1):
        print(f"[{index:02d}/{len(commands):02d}] {' '.join(command)}")

    if not args.execute:
        print("Dry run only. Re-run with --execute after the command set is verified.")
        return

    for index, command in enumerate(commands, start=1):
        print(f"\n=== Executing run {index}/{len(commands)} ===", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
