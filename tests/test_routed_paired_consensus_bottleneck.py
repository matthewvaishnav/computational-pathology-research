from __future__ import annotations

import csv
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.paired_acquisition import run_routed_paired_consensus_bottleneck as routed


REPOSITORY = Path(__file__).resolve().parents[1]
ANCHOR_RESULT = REPOSITORY / routed.FROZEN_SPECS["auxiliary_anchor"][0]


@pytest.fixture(scope="module")
def frozen() -> dict:
    return routed.verify_frozen_chain(REPOSITORY, ANCHOR_RESULT)


@pytest.fixture(scope="module")
def config(frozen: dict):
    return routed.anchor.config_from_benchmark(frozen["task_benchmark"]["payload"])


@pytest.fixture(scope="module")
def dataset(config):
    return routed.unseen.make_unseen_identity_dataset(config, "linear")


@pytest.fixture(scope="module")
def target(dataset):
    return routed.anchor.construct_consensus_targets(dataset)


def test_auxiliary_anchor_file_and_internal_hashes_verify(frozen: dict) -> None:
    record = frozen["auxiliary_anchor"]
    assert record["file_sha256"] == routed.FROZEN_SPECS["auxiliary_anchor"][1]
    assert record["payload"]["result_sha256"] == routed.FROZEN_SPECS["auxiliary_anchor"][2]


@pytest.mark.parametrize("name", tuple(routed.FROZEN_SPECS))
def test_complete_upstream_artifact_chain_verifies(name: str, frozen: dict) -> None:
    assert frozen[name]["file_sha256"] == routed.FROZEN_SPECS[name][1]
    routed.verify_internal_hash(frozen[name]["payload"], routed.FROZEN_SPECS[name][2])


def test_consensus_target_hashes_match_frozen_anchor(target, frozen: dict) -> None:
    expected = frozen["auxiliary_anchor"]["payload"]["target_manifests"]["4301:linear"]
    assert target["manifest"] == expected


def test_preflight_reproduction_comparator_accepts_frozen_copy(frozen: dict) -> None:
    prior = frozen["auxiliary_anchor"]["payload"]["target_preflight"]
    comparison = routed.compare_preflight(prior, prior)
    assert comparison["passed"]
    assert comparison["factorizer_models_initialized_during_preflight"] == 0


def test_preflight_initializes_zero_factorizers() -> None:
    source = inspect.getsource(routed.run_experiment)
    assert source.index("anchor.run_target_preflight") < source.index("build_family_model")


def test_exactly_three_families_exist() -> None:
    assert routed.FAMILIES == (
        "crossed_target_baseline_32",
        "routed_dimension_control_64",
        "routed_consensus_bottleneck_64",
    )


def test_exactly_twenty_four_fits_are_scheduled() -> None:
    assert len(routed.scheduled_runs()) == 24
    assert len(set(routed.scheduled_runs())) == 24


@pytest.mark.parametrize(
    ("family", "dimension", "weight"),
    (
        ("crossed_target_baseline_32", 32, 0.0),
        ("routed_dimension_control_64", 64, 0.0),
        ("routed_consensus_bottleneck_64", 64, 0.25),
    ),
)
def test_family_dimensions_and_weights(family: str, dimension: int, weight: float) -> None:
    assert routed.family_dimension(family) == dimension
    assert routed.consensus_weight(family) == weight


def test_routed_family_has_no_auxiliary_head(config) -> None:
    model = routed.build_family_model(
        "routed_consensus_bottleneck_64", config, torch.device("cpu")
    )
    assert not hasattr(model, "consensus_head")


def test_decoder_receives_exact_returned_representation(config) -> None:
    model = routed.build_family_model(
        "routed_consensus_bottleneck_64", config, torch.device("cpu")
    )
    proof = routed.verify_single_biological_path(model, config.observation_dim, torch.device("cpu"))
    assert proof["decoder_received_exact_returned_tensor"]
    assert proof["single_biological_path_verified"]


def test_no_private_preconsensus_representation_exists(config) -> None:
    model = routed.build_family_model(
        "routed_consensus_bottleneck_64", config, torch.device("cpu")
    )
    assert not hasattr(model, "preconsensus_code")
    assert not hasattr(model, "decoder_content_code")


def test_no_encoder_decoder_bypass_exists() -> None:
    source = inspect.getsource(routed.RoutedFactorizer)
    assert "skip" not in source
    assert "torch.cat" not in source
    assert "detach" not in source


@pytest.mark.parametrize(
    "required_fragment",
    (
        "all_biological.index_select(0, consistency_left)",
        "biological_variance_floor(all_biological.index_select(0, train))",
        "model.decode(\n            all_biological.index_select(0, crossed_source)",
        "F.mse_loss(all_biological.index_select(0, train), targets.index_select(0, train))",
    ),
)
def test_training_consumers_use_the_routed_code(required_fragment: str) -> None:
    assert required_fragment in inspect.getsource(routed.train_family_model)


@pytest.mark.parametrize(
    "required_fragment",
    (
        "anchor.model_task_evaluation",
        "anchor.operational_diagnostics",
        "retrieval_geometry",
        "repeated_nonlinear_scanner_probe",
        "anchor.counterfactual_linear_task",
    ),
)
def test_evaluation_consumers_use_the_returned_code(required_fragment: str) -> None:
    source = inspect.getsource(routed.run_experiment) + inspect.getsource(
        routed.anchor.operational_diagnostics
    )
    assert required_fragment in source


def test_biological_latents_and_task_labels_are_absent_from_training() -> None:
    source = inspect.getsource(routed.train_family_model)
    assert "biological_latents" not in source
    assert "labels" not in inspect.signature(routed.train_family_model).parameters


def test_only_dimension_and_consensus_loss_differ_between_routed_families(config) -> None:
    control = routed.build_family_model(
        "routed_dimension_control_64", config, torch.device("cpu")
    )
    anchored = routed.build_family_model(
        "routed_consensus_bottleneck_64", config, torch.device("cpu")
    )
    assert type(control) is type(anchored)
    assert [(name, tuple(value.shape)) for name, value in control.state_dict().items()] == [
        (name, tuple(value.shape)) for name, value in anchored.state_dict().items()
    ]
    assert routed.consensus_weight("routed_dimension_control_64") == 0.0
    assert routed.consensus_weight("routed_consensus_bottleneck_64") == 0.25


def test_baseline_replication_is_mandatory() -> None:
    assert "Frozen baseline replication failed closed" in inspect.getsource(
        routed.run_experiment
    )


def test_task_budgets_count_identities() -> None:
    assert routed.task_benchmark.LABEL_BUDGETS == (8, 16, 32)


def test_both_residual_seeds_are_required() -> None:
    assert routed.task_benchmark.RESIDUAL_SEEDS == (7203, 7204)
    assert "len(biological) == 4" in inspect.getsource(routed.anchor.linear_task_flags)


def test_identity_permutation_operates_by_identity() -> None:
    assert "identity_permuted_features" in inspect.getsource(
        routed.anchor.model_task_evaluation
    )


def test_normalized_invariance_uses_between_identity_variance() -> None:
    source = inspect.getsource(routed.consensus_geometry_metrics)
    assert "within.mean() / between" not in source
    assert "ratios = within / max(between" in source


def test_legacy_and_normalized_invariance_flags_are_separate() -> None:
    source = inspect.getsource(routed.consensus_geometry_metrics)
    assert '"legacy_absolute_view_variance_passed": legacy' in source
    assert '"normalized_consensus_invariance_passed": normalized' in source


def test_normalized_invariance_thresholds_are_exact() -> None:
    assert routed.NORMALIZED_VARIANCE_RATIO_MAXIMUM == 0.01
    assert routed.MAXIMUM_IDENTITY_VARIANCE_RATIO == 0.05
    assert routed.LEGACY_VIEW_VARIANCE_TOLERANCE == 1e-4


def test_primary_scanner_seeds_are_frozen() -> None:
    assert routed.PRIMARY_SCANNER_SEEDS == (7301, 7302, 7303)
    assert tuple(routed.calibrated.SCANNER_PROBE_SEEDS) == routed.PRIMARY_SCANNER_SEEDS


def test_expanded_confirmation_only_triggers_after_primary_leakage() -> None:
    result = routed.expanded_scanner_confirmation(
        np.empty((0, 1)), None, None, None, False  # type: ignore[arg-type]
    )
    assert not result["triggered"]
    assert not result["expanded_scanner_leakage_confirmed"]


def test_every_expanded_seed_has_a_paired_null() -> None:
    source = inspect.getsource(routed.expanded_scanner_confirmation)
    assert routed.EXPANDED_SCANNER_SEEDS == tuple(range(7301, 7309))
    assert "include_permutation_null=True" in source


def test_acquisition_exclusion_requires_all_task_probes() -> None:
    source = inspect.getsource(routed.anchor.linear_task_flags)
    assert "max(acquisition_ridge) < 0.10" in source
    assert "max(acquisition_residual) < 0.10" in source


def test_counterfactual_requires_direct_eligibility() -> None:
    assert "eligible = direct_r2 >= 0.70" in inspect.getsource(
        routed.anchor.counterfactual_linear_task
    )


def test_counterfactual_uses_requested_target_scanner() -> None:
    source = inspect.getsource(routed.anchor.counterfactual_linear_task)
    assert "model.acquisition_from_scanner(target_scanner)" in source


def test_counterfactual_reencodes_after_decoding() -> None:
    source = inspect.getsource(routed.anchor.counterfactual_linear_task)
    assert source.index("generated = model.decode") < source.index(
        "model.encode_biological(generated)"
    )


def _task_result(score: float, area: float) -> dict:
    record = {
        "probes": {
            "residual_repeats": [
                {"metrics": {"r2": score}},
                {"metrics": {"r2": score}},
            ]
        }
    }
    return {
        "evaluations": {"biological_code": {"32": [record, record]}},
        "label_efficiency": {
            "biological_code": {"area_under_performance_vs_log_label_budget": area}
        },
    }


def _fake_run(family: str, score: float = 0.0, area: float = 0.0) -> dict:
    return {
        "family": family,
        "linear_task_evaluation": _task_result(score, area),
        "interpretation_flags": {
            "routed_consensus_mechanism_succeeded": False,
            "dimension_increase_succeeded": False,
            "routed_consensus_operational_tradeoff": False,
            "linear_task_sufficient": False,
            "operational_capabilities_preserved": True,
        },
    }


def test_dimension_control_success_cannot_be_attributed_to_consensus() -> None:
    source = inspect.getsource(routed.run_flags)
    assert 'family == "routed_dimension_control_64" and common' in source
    assert 'family == "routed_consensus_bottleneck_64"' in source


def test_operational_degradation_triggers_tradeoff_status() -> None:
    runs = [_fake_run(family) for family in routed.FAMILIES for _ in range(8)]
    runs[-1]["interpretation_flags"]["routed_consensus_operational_tradeoff"] = True
    assert (
        routed.family_interpretation(runs)["status"]
        == "complete_routed_consensus_operational_tradeoff"
    )


def test_poor_scientific_performance_is_not_execution_failure() -> None:
    runs = [_fake_run(family) for family in routed.FAMILIES for _ in range(8)]
    assert (
        routed.family_interpretation(runs)["status"]
        == "complete_routed_consensus_mechanism_unsupported"
    )


def test_prior_statuses_remain_immutable(frozen: dict) -> None:
    assert frozen["auxiliary_anchor"]["status"] == "complete_consensus_anchor_operational_tradeoff"
    assert frozen["task_benchmark"]["status"] == "complete_task_defined_biological_sufficiency_unsupported"


def test_existing_output_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    with pytest.raises(routed.ExperimentError, match="overwrite"):
        routed.ensure_new_output_root(tmp_path)


def test_heterogeneous_csv_output_is_supported(tmp_path: Path) -> None:
    rows = [{"a": 1}, {"a": 2, "b": 3}]
    path = tmp_path / "summary.csv"
    routed.parent.atomic_csv(path, routed.parent.summary_csv_fieldnames(rows), rows)
    with path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["b"] == ""


def test_all_frozen_artifacts_remain_unchanged(frozen: dict) -> None:
    after = routed.verify_frozen_chain(REPOSITORY, ANCHOR_RESULT)
    assert {name: row["file_sha256"] for name, row in frozen.items()} == {
        name: row["file_sha256"] for name, row in after.items()
    }


def test_one_epoch_routed_training_records_direct_gradient_diagnostics(
    config, dataset, target
) -> None:
    short = replace(config, epochs=1)
    routed.base.set_deterministic_seed(2201)
    model = routed.build_family_model(
        "routed_consensus_bottleneck_64", short, torch.device("cpu")
    )
    result = routed.train_family_model(
        model,
        "routed_consensus_bottleneck_64",
        dataset,
        target["per_view_standardized"],
        short,
        torch.device("cpu"),
    )
    row = result["history"][-1]
    assert result["optimizer_steps"] == 1
    assert row["consensus_loss"] > 0
    assert row["biological_encoder_consensus_gradient_norm"] > 0
    assert np.isfinite(row["original_consensus_gradient_cosine_similarity"])
    assert row["per_dimension_training_consensus_r2"]


def test_target_constructor_accepts_no_latents_or_labels() -> None:
    assert list(inspect.signature(routed.anchor.construct_consensus_targets).parameters) == [
        "dataset"
    ]


def test_target_fitting_uses_training_identities_once(target) -> None:
    assert target["manifest"]["training_identity_count"] == 40
    assert target["manifest"]["every_training_identity_contributes_once_to_scaler"]
    assert target["manifest"]["every_scanner_contributes_once_per_identity"]


def test_unseen_identities_do_not_affect_target_fit(target) -> None:
    assert not target["manifest"]["unseen_identities_used_for_fitting"]
    assert target["manifest"]["targets_detached_constants"]


def test_routed_result_does_not_define_path_forward_gate() -> None:
    assert "path_forward_gate_open" not in inspect.getsource(routed)
