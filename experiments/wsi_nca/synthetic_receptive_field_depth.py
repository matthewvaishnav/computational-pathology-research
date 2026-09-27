#!/usr/bin/env python3
"""Falsify shallow WSI-NCA dynamics with an exact depth-two task.

The two classes use the same eight-node line geometry and the same four-zero /
four-one feature multiset.  Their colored, signed-edge rooted neighborhoods have
the same slide-level multiset through depth one and differ first at depth two:

    class 0: 00101101
    class 1: 00110101

Consequently, T=0 and T=1 followed by any permutation-invariant readout cannot
separate the classes.  A T>=2 local model can in principle separate them.  The
experiment trains the repository's actual WSI-NCA implementation and includes
shuffled-topology and untied-depth controls.

This is a synthetic mechanism test, not pathology or biological evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.wsi_nca import WSINCA, build_neighbor_index  # noqa: E402

CLASS_TEMPLATES = {
    0: (0, 0, 1, 0, 1, 1, 0, 1),
    1: (0, 0, 1, 1, 0, 1, 0, 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSI-NCA receptive-field-depth falsification")
    parser.add_argument("--train-pairs", type=int, default=256)
    parser.add_argument("--val-pairs", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43])
    parser.add_argument(
        "--diagnostic-epochs",
        type=int,
        default=240,
        help="Post-hoc fixed-horizon diagnostic for depth-capable runs stuck at chance",
    )
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out",
        default="results/wsi_nca_phase_a/synthetic_receptive_field_depth.json",
    )
    return parser.parse_args()


def base_coordinates() -> Tensor:
    """Return a fixed line geometry; only relative offsets enter the model."""
    return torch.stack((torch.arange(8, dtype=torch.float32), torch.zeros(8)), dim=1)


def template_features(label: int, complement: bool = False) -> Tensor:
    cell_types = torch.tensor(CLASS_TEMPLATES[label], dtype=torch.long)
    if complement:
        cell_types = 1 - cell_types
    return torch.nn.functional.one_hot(cell_types, num_classes=2).float()


def make_dataset(num_pairs: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """Create exactly paired class samples with shared nuisance transforms."""
    if num_pairs < 1:
        raise ValueError("num_pairs must be >= 1")

    rng = np.random.default_rng(seed)
    features: list[Tensor] = []
    coordinates: list[Tensor] = []
    labels: list[int] = []

    for _ in range(num_pairs):
        complement = bool(rng.integers(0, 2))
        # Absolute position is randomized and shared by the class pair. The
        # model itself receives only relative positions inside local messages.
        translation = torch.tensor(rng.integers(-100, 101, size=2), dtype=torch.float32)
        pair_coordinates = base_coordinates() + translation
        for label in (0, 1):
            features.append(template_features(label, complement=complement))
            coordinates.append(pair_coordinates.clone())
            labels.append(label)

    permutation = torch.tensor(rng.permutation(len(labels)), dtype=torch.long)
    return (
        torch.stack(features)[permutation],
        torch.stack(coordinates)[permutation],
        torch.tensor(labels, dtype=torch.long)[permutation],
    )


def reassign_coordinates(coordinates: Tensor, seed: int) -> Tensor:
    """Destroy feature/topology correspondence while preserving each coordinate set."""
    rng = np.random.default_rng(seed)
    reassigned = coordinates.clone()
    for sample_index in range(coordinates.shape[0]):
        permutation = torch.tensor(rng.permutation(coordinates.shape[1]), dtype=torch.long)
        reassigned[sample_index] = reassigned[sample_index, permutation]
    return reassigned


def rooted_signatures(label: int, depth: int) -> list[object]:
    """Compute exact colored, signed-edge rooted signatures on the model's kNN graph."""
    features = template_features(label)
    coordinates = base_coordinates()
    mask = torch.ones(1, coordinates.shape[0], dtype=torch.bool)
    neighbor_index = build_neighbor_index(
        states=features.unsqueeze(0),
        coordinates=coordinates.unsqueeze(0),
        mask=mask,
        k=2,
        mode="spatial",
    )[0]

    signatures: list[object] = [int(value) for value in features.argmax(dim=1).tolist()]
    for _ in range(depth):
        refined: list[object] = []
        for cell_index, center_signature in enumerate(signatures):
            neighbor_signatures = []
            for neighbor in neighbor_index[cell_index].tolist():
                delta = coordinates[neighbor] - coordinates[cell_index]
                edge_label = (int(delta[0].item()), int(delta[1].item()))
                neighbor_signatures.append((edge_label, signatures[neighbor]))
            refined.append((center_signature, tuple(sorted(neighbor_signatures, key=repr))))
        signatures = refined
    return signatures


def slide_signature(label: int, depth: int) -> list[str]:
    return sorted(repr(signature) for signature in rooted_signatures(label, depth))


def signature_digest(signature: Sequence[str]) -> str:
    encoded = json.dumps(list(signature), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def construction_audit(max_depth: int = 4) -> dict[str, object]:
    depths: dict[str, object] = {}
    for depth in range(max_depth + 1):
        class_0 = slide_signature(0, depth)
        class_1 = slide_signature(1, depth)
        row: dict[str, object] = {
            "equal": class_0 == class_1,
            "class_0_sha256": signature_digest(class_0),
            "class_1_sha256": signature_digest(class_1),
        }
        if depth == 1:
            row["class_0_histogram"] = dict(sorted(Counter(class_0).items()))
            row["class_1_histogram"] = dict(sorted(Counter(class_1).items()))
        depths[str(depth)] = row

    return {
        "unordered_feature_multisets_equal": sorted(CLASS_TEMPLATES[0])
        == sorted(CLASS_TEMPLATES[1]),
        "coordinate_geometry_equal": True,
        "neighbor_count": 2,
        "depths": depths,
        "first_distinguishing_depth": next(
            depth for depth in range(max_depth + 1) if not bool(depths[str(depth)]["equal"])
        ),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: WSINCA,
    features: Tensor,
    coordinates: Tensor,
    labels: Tensor,
) -> dict[str, float]:
    model.eval()
    logits = model(features, coordinates).logits
    return {
        "loss": float(torch.nn.functional.cross_entropy(logits, labels).item()),
        "accuracy": float((logits.argmax(dim=1) == labels).float().mean().item()),
    }


def train_model(
    train_data: tuple[Tensor, Tensor, Tensor],
    val_data: tuple[Tensor, Tensor, Tensor],
    *,
    num_steps: int,
    dynamics_mode: str,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[WSINCA, dict[str, float]]:
    set_seed(seed)
    model = WSINCA(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_classes=2,
        num_steps=num_steps,
        k_neighbors=2,
        neighbor_mode="spatial",
        dynamics_mode=dynamics_mode,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_features, train_coordinates, train_labels = (item.to(device) for item in train_data)
    val_features, val_coordinates, val_labels = (item.to(device) for item in val_data)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(train_labels.shape[0], device=device)
        for start in range(0, order.numel(), batch_size):
            batch_index = order[start : start + batch_size]
            logits = model(
                train_features[batch_index],
                train_coordinates[batch_index],
            ).logits
            loss = torch.nn.functional.cross_entropy(logits, train_labels[batch_index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    train_metrics = evaluate(model, train_features, train_coordinates, train_labels)
    val_metrics = evaluate(model, val_features, val_coordinates, val_labels)
    return model, {
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
    }


def parameter_count(model: WSINCA) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def readout_leakage_audit(hidden_dim: int) -> dict[str, object]:
    """Numerically verify that the global readout cannot bypass T=0/T=1 depth."""
    features = torch.stack(
        [
            template_features(0),
            template_features(1),
            template_features(0, complement=True),
            template_features(1, complement=True),
        ]
    )
    coordinates = base_coordinates().unsqueeze(0).repeat(4, 1, 1)
    coordinates[2:] += torch.tensor([73.0, -41.0])

    rows: dict[str, object] = {}
    for steps in (0, 1):
        max_logit_difference = 0.0
        max_slide_state_difference = 0.0
        for seed in (11, 29, 47):
            set_seed(seed)
            model = WSINCA(
                input_dim=2,
                hidden_dim=hidden_dim,
                num_classes=2,
                num_steps=steps,
                k_neighbors=2,
                dynamics_mode="tied",
                dropout=0.0,
            ).eval()
            with torch.no_grad():
                output = model(features, coordinates)
            for first, second in ((0, 1), (2, 3)):
                max_logit_difference = max(
                    max_logit_difference,
                    float((output.logits[first] - output.logits[second]).abs().max().item()),
                )
                max_slide_state_difference = max(
                    max_slide_state_difference,
                    float(
                        (output.slide_state[first] - output.slide_state[second]).abs().max().item()
                    ),
                )
        rows[f"t{steps}"] = {
            "max_abs_logit_difference": max_logit_difference,
            "max_abs_slide_state_difference": max_slide_state_difference,
        }
    return rows


def depth_initialization_audit(seeds: Sequence[int], hidden_dim: int) -> list[dict[str, object]]:
    """Measure whether depth-capable initial states and gradients are non-degenerate."""
    features = torch.stack(
        [
            template_features(0),
            template_features(1),
            template_features(0, complement=True),
            template_features(1, complement=True),
        ]
    )
    coordinates = base_coordinates().unsqueeze(0).repeat(4, 1, 1)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    rows: list[dict[str, object]] = []

    for seed in seeds:
        for steps, dynamics_mode in ((2, "tied"), (4, "tied"), (4, "untied")):
            set_seed(seed)
            model = WSINCA(
                input_dim=2,
                hidden_dim=hidden_dim,
                num_classes=2,
                num_steps=steps,
                k_neighbors=2,
                dynamics_mode=dynamics_mode,
                dropout=0.0,
            )
            output = model(features, coordinates)
            loss = torch.nn.functional.cross_entropy(output.logits, labels)
            loss.backward()
            gradient_l2 = (
                sum(
                    float(parameter.grad.detach().square().sum().item())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
                ** 0.5
            )
            rows.append(
                {
                    "seed": seed,
                    "steps": steps,
                    "dynamics_mode": dynamics_mode,
                    "initial_slide_gap": float(
                        (output.slide_state[0] - output.slide_state[1]).detach().abs().max().item()
                    ),
                    "initial_logit_gap": float(
                        (output.logits[0] - output.logits[1]).detach().abs().max().item()
                    ),
                    "initial_gradient_l2": gradient_l2,
                }
            )
    return rows


def make_run_row(
    model: WSINCA,
    metrics: dict[str, float],
    *,
    steps: int,
    dynamics_mode: str,
    train_topology: str,
    eval_topology: str,
) -> dict[str, object]:
    return {
        "steps": steps,
        "dynamics_mode": dynamics_mode,
        "train_topology": train_topology,
        "eval_topology": eval_topology,
        "parameter_count": parameter_count(model),
        **metrics,
    }


def aggregate_runs(per_seed: Sequence[dict[str, object]]) -> dict[str, object]:
    run_names = sorted(per_seed[0]["runs"])
    aggregate: dict[str, object] = {}
    for run_name in run_names:
        rows = [seed_row["runs"][run_name] for seed_row in per_seed]
        val_accuracies = [float(row["val_accuracy"]) for row in rows]
        train_accuracies = [float(row["train_accuracy"]) for row in rows]
        aggregate[run_name] = {
            "val_accuracy_values": val_accuracies,
            "val_accuracy_mean": float(np.mean(val_accuracies)),
            "val_accuracy_std": float(np.std(val_accuracies)),
            "train_accuracy_values": train_accuracies,
            "train_accuracy_mean": float(np.mean(train_accuracies)),
            "parameter_count": int(rows[0]["parameter_count"]),
        }
    return aggregate


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if args.train_pairs < 1 or args.val_pairs < 1:
        raise ValueError("train-pairs and val-pairs must be >= 1")
    if args.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if args.diagnostic_epochs < args.epochs:
        raise ValueError("diagnostic-epochs must be >= epochs")

    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)

    audit = construction_audit()
    if not (
        bool(audit["depths"]["0"]["equal"])
        and bool(audit["depths"]["1"]["equal"])
        and not bool(audit["depths"]["2"]["equal"])
    ):
        raise RuntimeError("Synthetic construction no longer has the frozen depth-two boundary")

    train_real = make_dataset(args.train_pairs, seed=123)
    val_real = make_dataset(args.val_pairs, seed=456)
    train_shuffled = (
        train_real[0],
        reassign_coordinates(train_real[1], seed=8123),
        train_real[2],
    )
    val_shuffled = (
        val_real[0],
        reassign_coordinates(val_real[1], seed=8456),
        val_real[2],
    )

    per_seed: list[dict[str, object]] = []
    for seed in args.seeds:
        runs: dict[str, object] = {}
        trained_t4: WSINCA | None = None
        trained_t4_metrics: dict[str, float] | None = None

        for steps in (0, 1, 2, 4):
            model, metrics = train_model(
                train_real,
                val_real,
                num_steps=steps,
                dynamics_mode="tied",
                hidden_dim=args.hidden_dim,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=seed,
                device=device,
            )
            runs[f"t{steps}_tied_real"] = make_run_row(
                model,
                metrics,
                steps=steps,
                dynamics_mode="tied",
                train_topology="real",
                eval_topology="real",
            )
            if steps == 4:
                trained_t4 = model
                trained_t4_metrics = metrics

        if trained_t4 is None or trained_t4_metrics is None:
            raise RuntimeError("Missing trained tied T4 model")
        shuffled_eval = evaluate(
            trained_t4,
            val_shuffled[0].to(device),
            val_shuffled[1].to(device),
            val_shuffled[2].to(device),
        )
        runs["t4_tied_real_train_shuffled_eval"] = make_run_row(
            trained_t4,
            {
                "train_loss": trained_t4_metrics["train_loss"],
                "train_accuracy": trained_t4_metrics["train_accuracy"],
                "val_loss": shuffled_eval["loss"],
                "val_accuracy": shuffled_eval["accuracy"],
            },
            steps=4,
            dynamics_mode="tied",
            train_topology="real",
            eval_topology="shuffled",
        )

        shuffled_model, shuffled_metrics = train_model(
            train_shuffled,
            val_shuffled,
            num_steps=4,
            dynamics_mode="tied",
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
            device=device,
        )
        runs["t4_tied_shuffled"] = make_run_row(
            shuffled_model,
            shuffled_metrics,
            steps=4,
            dynamics_mode="tied",
            train_topology="shuffled",
            eval_topology="shuffled",
        )

        untied_model, untied_metrics = train_model(
            train_real,
            val_real,
            num_steps=4,
            dynamics_mode="untied",
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
            device=device,
        )
        runs["t4_untied_real"] = make_run_row(
            untied_model,
            untied_metrics,
            steps=4,
            dynamics_mode="untied",
            train_topology="real",
            eval_topology="real",
        )
        per_seed.append({"seed": seed, "runs": runs})

    # Preserve the locked-horizon primary outcome. A representationally capable
    # run that remains at chance on both train and validation gets one post-hoc
    # rerun where only the epoch horizon changes. It never replaces the primary
    # metric in the aggregate above.
    diagnostic_configs = {
        "t2_tied_real": (2, "tied"),
        "t4_tied_real": (4, "tied"),
        "t4_untied_real": (4, "untied"),
    }
    extended_horizon: list[dict[str, object]] = []
    if args.diagnostic_epochs > args.epochs:
        for seed_row in per_seed:
            for run_name, (steps, dynamics_mode) in diagnostic_configs.items():
                primary = seed_row["runs"][run_name]
                if float(primary["train_accuracy"]) > 0.75:
                    continue
                diagnostic_model, diagnostic_metrics = train_model(
                    train_real,
                    val_real,
                    num_steps=steps,
                    dynamics_mode=dynamics_mode,
                    hidden_dim=args.hidden_dim,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    epochs=args.diagnostic_epochs,
                    batch_size=args.batch_size,
                    seed=int(seed_row["seed"]),
                    device=device,
                )
                extended_horizon.append(
                    {
                        "seed": int(seed_row["seed"]),
                        "run": run_name,
                        "primary_epochs": args.epochs,
                        "diagnostic_epochs": args.diagnostic_epochs,
                        "parameter_count": parameter_count(diagnostic_model),
                        "primary_metrics": {
                            key: primary[key]
                            for key in (
                                "train_loss",
                                "train_accuracy",
                                "val_loss",
                                "val_accuracy",
                            )
                        },
                        "extended_metrics": diagnostic_metrics,
                    }
                )

    aggregate = aggregate_runs(per_seed)
    tied_t4_count = int(aggregate["t4_tied_real"]["parameter_count"])
    untied_t4_count = int(aggregate["t4_untied_real"]["parameter_count"])
    t1_accuracy = float(aggregate["t1_tied_real"]["val_accuracy_mean"])
    t2_accuracy = float(aggregate["t2_tied_real"]["val_accuracy_mean"])
    t4_accuracy = float(aggregate["t4_tied_real"]["val_accuracy_mean"])
    shuffled_accuracy = float(aggregate["t4_tied_shuffled"]["val_accuracy_mean"])
    successful_t4_shuffled_eval = [
        float(seed_row["runs"]["t4_tied_real_train_shuffled_eval"]["val_accuracy"])
        for seed_row in per_seed
        if float(seed_row["runs"]["t4_tied_real"]["val_accuracy"]) > 0.9
    ]

    payload = {
        "experiment": "wsi_nca_receptive_field_depth_falsification",
        "status": "synthetic mechanism evidence only; not pathology evidence",
        "scientific_question": (
            "Can repeated application of one shared local update rule extract a signal "
            "that one spatial pass cannot?"
        ),
        "predictions_before_observation": {
            "t0": "incapable because the unordered feature multisets are identical",
            "t1": (
                "incapable because the complete slide multiset of rooted one-hop "
                "colored, signed-edge neighborhoods is identical"
            ),
            "t2_t4": "representationally capable because class signatures first differ at depth two",
            "shuffled": "expected to remove the stable feature/topology correspondence",
        },
        "task_construction": {
            "class_templates": {str(key): list(value) for key, value in CLASS_TEMPLATES.items()},
            "geometry": "eight equally spaced cells on a line; k=2 spatial nearest neighbors",
            "nuisance_controls": [
                "class-paired random global feature complement",
                "class-paired random coordinate translation",
                "the model uses relative rather than absolute coordinates in messages",
            ],
            "construction_audit": audit,
            "readout_leakage_audit": readout_leakage_audit(args.hidden_dim),
            "depth_initialization_audit": depth_initialization_audit(args.seeds, args.hidden_dim),
        },
        "execution": {
            "train_pairs": args.train_pairs,
            "val_pairs": args.val_pairs,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "seeds": args.seeds,
            "device": str(device),
            "diagnostic_epochs": args.diagnostic_epochs,
        },
        "observed_results": {
            "per_seed": per_seed,
            "aggregate": aggregate,
            "parameter_counts": {
                "tied_t4": tied_t4_count,
                "untied_t4": untied_t4_count,
                "untied_minus_tied": untied_t4_count - tied_t4_count,
                "untied_over_tied": untied_t4_count / tied_t4_count,
            },
            "post_hoc_optimization_diagnostics": {
                "rule": (
                    "Only depth-capable primary runs with train accuracy <= 0.75 were "
                    "rerun with the same data, model, seed, optimizer, and learning rate; "
                    "only the epoch horizon changed."
                ),
                "extended_horizon_runs": extended_horizon,
            },
        },
        "bounded_interpretation": {
            "t0_t1_at_chance": bool(
                float(aggregate["t0_tied_real"]["val_accuracy_mean"]) == 0.5 and t1_accuracy == 0.5
            ),
            "depth_two_gate_observed": bool(t2_accuracy > t1_accuracy + 0.25),
            "t4_depth_gate_observed": bool(t4_accuracy > t1_accuracy + 0.25),
            "topology_perturbation_destroyed_t4_signal": bool(
                shuffled_accuracy < t4_accuracy - 0.25
            ),
            "successful_t4_seed_shuffled_eval_accuracy_mean": (
                float(np.mean(successful_t4_shuffled_eval)) if successful_t4_shuffled_eval else None
            ),
            "locked_horizon_optimization_failures": [
                {"seed": row["seed"], "run": row["run"]} for row in extended_horizon
            ],
            "optimization_diagnosis": (
                "A locked-horizon failure is attributable to optimization rather than "
                "task leakage or architectural incapacity only when the same initialization "
                "has nonzero class-sensitive states/gradients and succeeds in the labeled "
                "extended-horizon diagnostic."
            ),
            "claim_boundary": (
                "A positive separation supports only that this shared local update can "
                "propagate information across multiple graph hops on the constructed task. "
                "It does not establish pathology utility, self-repair, attractors, "
                "regeneration, morphogenesis, or clinical value."
            ),
        },
        "provenance": {
            "git_parent_revision": git_revision(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }

    rendered = json.dumps(payload, indent=2)
    print(rendered)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
