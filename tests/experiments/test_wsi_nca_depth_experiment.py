import argparse
import sys

import pandas as pd

from experiments.wsi_nca.run_phase_a_matrix import build_commands
from experiments.wsi_nca.synthetic_receptive_field_depth import (
    construction_audit,
    readout_leakage_audit,
)
from experiments.wsi_nca.train_panda_phase_a import (
    make_splits,
    parse_args,
    resolve_test_coordinate_modes,
)


def test_depth_task_is_equal_through_one_hop_and_differs_at_two():
    audit = construction_audit()

    assert audit["unordered_feature_multisets_equal"]
    assert audit["coordinate_geometry_equal"]
    assert audit["depths"]["0"]["equal"]
    assert audit["depths"]["1"]["equal"]
    assert not audit["depths"]["2"]["equal"]
    assert audit["first_distinguishing_depth"] == 2


def test_static_and_one_hop_readout_cannot_bypass_task_construction():
    audit = readout_leakage_audit(hidden_dim=12)

    assert audit["t0"]["max_abs_logit_difference"] < 1e-6
    assert audit["t1"]["max_abs_logit_difference"] < 1e-6
    assert audit["t0"]["max_abs_slide_state_difference"] < 1e-6
    assert audit["t1"]["max_abs_slide_state_difference"] < 1e-6


def test_panda_trainer_exposes_frozen_protocol_controls(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_panda_phase_a.py",
            "--dynamics-mode",
            "untied",
            "--split-seed",
            "42",
            "--test-fraction",
            "0.2",
            "--eval-coordinate-control",
            "both",
        ],
    )

    args = parse_args()
    assert args.dynamics_mode == "untied"
    assert args.split_seed == 42
    assert args.test_fraction == 0.2
    assert args.eval_coordinate_control == "both"


def test_fixed_split_is_180_60_60_disjoint_and_deterministic():
    frame = pd.DataFrame(
        {
            "image_id": [f"slide-{index:03d}" for index in range(300)],
            "isup_grade": [grade for grade in range(6) for _ in range(50)],
        }
    )

    first = make_splits(
        frame,
        val_fraction=0.2,
        test_fraction=0.2,
        split_seed=42,
    )
    second = make_splits(
        frame,
        val_fraction=0.2,
        test_fraction=0.2,
        split_seed=42,
    )

    train, val, test = first
    assert (len(train), len(val), len(test)) == (180, 60, 60)

    train_ids = set(train["image_id"])
    val_ids = set(val["image_id"])
    test_ids = set(test["image_id"])
    assert not train_ids & val_ids
    assert not train_ids & test_ids
    assert not val_ids & test_ids
    assert train_ids | val_ids | test_ids == set(frame["image_id"])

    for lhs, rhs in zip(first, second):
        assert lhs["image_id"].tolist() == rhs["image_id"].tolist()

    assert train["isup_grade"].value_counts().sort_index().tolist() == [30] * 6
    assert val["isup_grade"].value_counts().sort_index().tolist() == [10] * 6
    assert test["isup_grade"].value_counts().sort_index().tolist() == [10] * 6


def test_real_trained_checkpoint_can_be_evaluated_real_and_shuffled():
    assert resolve_test_coordinate_modes("real", "both") == ["real", "shuffle"]
    assert resolve_test_coordinate_modes("real", "match") == ["real"]
    assert resolve_test_coordinate_modes("shuffle", "match") == ["shuffle"]


def _matrix_args(**overrides):
    values = dict(
        manifest="manifest.csv",
        out_dir="results",
        steps=[4],
        seeds=[42],
        epochs=20,
        batch_size=4,
        max_patches=512,
        hidden_dim=256,
        k_neighbors=8,
        lr=3e-4,
        weight_decay=1e-4,
        val_fraction=0.2,
        test_fraction=0.2,
        split_seed=42,
        device="cpu",
        limit=None,
        include_shuffled=False,
        include_embedding=False,
        include_real_shuffled_eval_t4=False,
        include_untied_t4=True,
        scientific_causal_ladder=False,
        execute=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _command_value(command, flag):
    return command[command.index(flag) + 1]


def test_matrix_can_add_tied_and_untied_t4_controls():
    commands = build_commands(_matrix_args(epochs=2, max_patches=128, hidden_dim=32))

    assert len(commands) == 2
    dynamics_modes = {_command_value(command, "--dynamics-mode") for command in commands}
    assert dynamics_modes == {"tied", "untied"}
    assert all(_command_value(command, "--num-steps") == "4" for command in commands)
    assert all(_command_value(command, "--split-seed") == "42" for command in commands)


def test_scientific_causal_ladder_is_exact_and_matched():
    args = _matrix_args(
        steps=[0, 1, 4],
        seeds=[7, 19],
        include_untied_t4=False,
        scientific_causal_ladder=True,
    )

    commands = build_commands(args)

    assert len(commands) == 10
    for seed in [7, 19]:
        seed_commands = [
            command for command in commands if _command_value(command, "--seed") == str(seed)
        ]
        assert len(seed_commands) == 5

        observed = {
            (
                _command_value(command, "--num-steps"),
                _command_value(command, "--dynamics-mode"),
                _command_value(command, "--coordinate-control"),
                _command_value(command, "--eval-coordinate-control"),
            )
            for command in seed_commands
        }
        assert observed == {
            ("0", "tied", "real", "real"),
            ("1", "tied", "real", "real"),
            ("4", "tied", "real", "both"),
            ("4", "tied", "shuffle", "shuffle"),
            ("4", "untied", "real", "real"),
        }

    matched_flags = [
        "--epochs",
        "--batch-size",
        "--max-patches",
        "--hidden-dim",
        "--k-neighbors",
        "--lr",
        "--weight-decay",
        "--val-fraction",
        "--test-fraction",
        "--split-seed",
    ]
    for flag in matched_flags:
        assert len({_command_value(command, flag) for command in commands}) == 1
