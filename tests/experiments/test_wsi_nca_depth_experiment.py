import argparse
import sys

from experiments.wsi_nca.run_phase_a_matrix import build_commands
from experiments.wsi_nca.synthetic_receptive_field_depth import (
    construction_audit,
    readout_leakage_audit,
)
from experiments.wsi_nca.train_panda_phase_a import parse_args


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


def test_panda_trainer_exposes_dynamics_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_panda_phase_a.py", "--dynamics-mode", "untied"])

    assert parse_args().dynamics_mode == "untied"


def test_matrix_can_add_tied_and_untied_t4_controls():
    args = argparse.Namespace(
        manifest="manifest.csv",
        out_dir="results",
        steps=[4],
        seeds=[42],
        epochs=2,
        batch_size=4,
        max_patches=128,
        hidden_dim=32,
        k_neighbors=4,
        lr=3e-4,
        weight_decay=1e-4,
        device="cpu",
        limit=None,
        include_shuffled=False,
        include_embedding=False,
        include_untied_t4=True,
        execute=False,
    )

    commands = build_commands(args)

    assert len(commands) == 2
    dynamics_modes = {command[command.index("--dynamics-mode") + 1] for command in commands}
    assert dynamics_modes == {"tied", "untied"}
    assert all(command[command.index("--num-steps") + 1] == "4" for command in commands)
