from __future__ import annotations

import csv
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.paired_acquisition import (
    run_biological_bottleneck_capacity_allocation_factorial as capacity,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ROUTED_RESULT = REPOSITORY / (
    "results/routed_paired_consensus_bottleneck_20260803T143209/"
    "routed_paired_consensus_bottleneck_result.json"
)


@pytest.fixture(scope="module")
def frozen() -> dict:
    return capacity.verify_routed_chain(REPOSITORY, ROUTED_RESULT)


@pytest.fixture(scope="module")
def config(frozen: dict):
    return capacity.anchor.config_from_benchmark(frozen["task_benchmark"]["payload"])


@pytest.fixture(scope="module")
def parameter_audit(config):
    return capacity.verify_parameter_bands(config, torch.device("cpu"))


def test_routed_result_file_and_internal_hashes_verify(frozen: dict) -> None:
    assert frozen["routed_result"]["file_sha256"] == capacity.ROUTED_FILE_SHA256
    assert frozen["routed_result"]["internal_sha256"] == capacity.ROUTED_INTERNAL_SHA256


@pytest.mark.parametrize("name", ("routed_result", *capacity.routed.FROZEN_SPECS))
def test_complete_inherited_chain_verifies(name: str, frozen: dict) -> None:
    assert frozen[name]["file_sha256"]
    assert frozen[name]["internal_sha256"]


def test_no_consensus_target_is_constructed() -> None:
    source = inspect.getsource(capacity.run_experiment)
    assert "construct_consensus_targets" not in source
    assert '"consensus_target_constructed": False' in source


def test_no_consensus_loss_exists() -> None:
    source = inspect.getsource(capacity.run_experiment)
    assert "train_family_model" not in source
    assert '"consensus_loss_present": False' in source
    assert 'parent.train_model(\n                        "crossed_target_prototype"' in source


def test_no_latent_or_task_label_enters_factorizer_training() -> None:
    source = inspect.getsource(capacity.run_experiment)
    training_call = source[source.index("training = parent.train_model") : source.index("operational, biological")]
    assert "biological_latents" not in training_call
    assert "labels" not in training_call


def test_exactly_four_families_exist() -> None:
    assert len(capacity.FAMILIES) == 4


def test_exactly_sixty_four_fits_are_scheduled() -> None:
    assert len(capacity.scheduled_runs()) == 64
    assert len(set(capacity.scheduled_runs())) == 64


def test_four_model_seeds_are_present() -> None:
    assert capacity.MODEL_SEEDS == (2201, 2202, 2203, 2204)


def test_legacy_and_expansion_seeds_are_separate() -> None:
    assert capacity.LEGACY_MODEL_SEEDS == (2201, 2202)
    assert capacity.EXPANSION_MODEL_SEEDS == (2203, 2204)
    assert not set(capacity.LEGACY_MODEL_SEEDS) & set(capacity.EXPANSION_MODEL_SEEDS)


@pytest.mark.parametrize(
    ("biological_dimension", "hidden_width", "expected"),
    ((32, 128, 44296), (64, 112, 44184), (32, 145, 52626), (64, 128, 52520)),
)
def test_parameter_count_formula_is_exact(
    biological_dimension: int, hidden_width: int, expected: int
) -> None:
    assert capacity.parameter_count_formula(biological_dimension, hidden_width) == expected


@pytest.mark.parametrize("family", capacity.FAMILIES)
def test_expected_and_actual_parameter_counts_match(
    family: str, parameter_audit: dict
) -> None:
    row = parameter_audit["families"][family]
    assert row["actual_parameter_count"] == row["expected_parameter_count"]
    assert row["formula_parameter_count"] == row["expected_parameter_count"]


def test_matched_parameter_pairs_are_below_half_percent(parameter_audit: dict) -> None:
    assert parameter_audit["matched_pairs"]["low_budget"]["absolute_difference"] == 112
    assert parameter_audit["matched_pairs"]["high_budget"]["absolute_difference"] == 106
    assert all(
        row["relative_difference"] < 0.005
        for row in parameter_audit["matched_pairs"].values()
    )


def test_model_depth_and_operations_remain_fixed(config) -> None:
    models = [capacity.build_family_model(family, config, torch.device("cpu")) for family in capacity.FAMILIES]
    assert all(len(model.biological_encoder) == 4 for model in models)
    assert all(len(model.content_to_hidden) == 3 for model in models)
    assert all(len(model.output_head) == 4 for model in models)
    assert all(hasattr(model, "prototype_to_film") for model in models)


@pytest.mark.parametrize(
    ("family", "dimension", "hidden"),
    (
        ("b32_h128_low_budget", 32, 128),
        ("b64_h112_low_budget", 64, 112),
        ("b32_h145_high_budget", 32, 145),
        ("b64_h128_high_budget", 64, 128),
    ),
)
def test_family_dimensions_and_hidden_widths(
    family: str, dimension: int, hidden: int
) -> None:
    assert capacity.FAMILY_CONFIGS[family]["biological_dimension"] == dimension
    assert capacity.FAMILY_CONFIGS[family]["hidden_width"] == hidden


def test_exact_frozen_families_require_replication() -> None:
    source = inspect.getsource(capacity.run_experiment)
    assert 'family == "b32_h128_low_budget"' in source
    assert 'reference_family = "crossed_target_baseline_32"' in source
    assert 'family == "b64_h128_high_budget"' in source
    assert 'reference_family = "routed_dimension_control_64"' in source
    assert "Frozen architecture replication failed closed" in source


def test_replication_applies_only_to_legacy_seeds() -> None:
    source = inspect.getsource(capacity.run_experiment)
    assert "model_seed in LEGACY_MODEL_SEEDS" in source


def test_primary_endpoint_uses_only_linear_task() -> None:
    source = inspect.getsource(capacity.run_experiment)
    assert '["linear_regression"]' in source
    assert '"nonlinear_teacher"' not in source


def test_labeled_budgets_count_identities() -> None:
    assert capacity.task_benchmark.LABEL_BUDGETS == (8, 16, 32)


def test_labeled_subsets_are_nested() -> None:
    identities = np.arange(32)
    subsets = capacity.task_benchmark.nested_identity_subsets(identities, 8101)
    assert set(subsets[8]) < set(subsets[16]) < set(subsets[32])


def test_task_scalers_use_training_identities_only() -> None:
    source = inspect.getsource(capacity.anchor.model_task_evaluation)
    assert "make_task_split" in source
    assert "training_indices" in source


def test_both_residual_seeds_are_required() -> None:
    assert capacity.task_benchmark.RESIDUAL_SEEDS == (7203, 7204)
    assert "len(biological) == 4" in inspect.getsource(capacity.anchor.linear_task_flags)


def test_identity_permutation_operates_by_identity() -> None:
    assert "identity_permuted_features" in inspect.getsource(
        capacity.anchor.model_task_evaluation
    )


def test_spectral_pca_fits_training_identities_only() -> None:
    source = inspect.getsource(capacity.task_accessibility_spectral_audit)
    assert "pca.fit_transform(training_x)" in source
    assert '"pca_fit_uses_training_identities_only": True' in source


def test_pca_curve_never_selects_components_using_test_results() -> None:
    source = inspect.getsource(capacity.task_accessibility_spectral_audit)
    assert '"test_performance_used_for_component_selection": False' in source
    assert "max(component_curve" not in source


def test_regularization_selection_uses_validation_only() -> None:
    source = inspect.getsource(capacity.task_accessibility_spectral_audit)
    assert 'max(alpha_curve, key=lambda row: row["validation_r2"])' in source
    assert '"selected_by_validation_only": True' in source


def test_operational_evaluation_uses_encoder_biological_code() -> None:
    source = inspect.getsource(capacity.run_experiment)
    assert "operational, biological, acquisition = anchor.operational_diagnostics" in source
    assert "model_task_evaluation(\n                        model,\n                        family,\n                        biological" in source


def test_scanner_seed_policies_are_frozen() -> None:
    assert capacity.PRIMARY_SCANNER_SEEDS == (7301, 7302, 7303)
    assert capacity.EXPANDED_SCANNER_SEEDS == tuple(range(7301, 7309))


def test_expanded_scanner_runs_only_after_primary_leakage() -> None:
    source = inspect.getsource(capacity.run_experiment)
    assert "primary_leakage = operational" in source
    assert "expanded_scanner_confirmation" in source


def test_every_scanner_confirmation_has_a_paired_null() -> None:
    assert "include_permutation_null=True" in inspect.getsource(
        capacity.routed.expanded_scanner_confirmation
    )


def test_acquisition_exclusion_uses_all_frozen_requirements() -> None:
    source = inspect.getsource(capacity.task_flags)
    assert 'flags["acquisition_linear_task_excluded"]' in source
    assert 'operational["calibrated_flags"]["acquisition_biology_exclusion"]' in source
    assert 'operational["acquisition_prototype_invariance_verified"]' in source


def test_counterfactual_requires_direct_eligibility_and_requested_scanner() -> None:
    source = inspect.getsource(capacity.anchor.counterfactual_linear_task)
    assert "eligible = direct_r2 >= 0.70" in source
    assert "model.acquisition_from_scanner(target_scanner)" in source
    assert source.index("generated = model.decode") < source.index(
        "model.encode_biological(generated)"
    )


def _task_result(score: float, area: float) -> dict:
    repeats = [{"metrics": {"r2": score}}, {"metrics": {"r2": score}}]
    return {
        "evaluations": {
            "biological_code": {
                "32": [{"probes": {"residual_repeats": repeats}}] * 2
            }
        },
        "label_efficiency": {
            "biological_code": {"area_under_performance_vs_log_label_budget": area}
        },
    }


def _fake_runs(
    family_values: dict[str, tuple[float, float]],
    expanded_family: str | None = None,
) -> list[dict]:
    runs = []
    for dataset_seed in capacity.DATASET_SEEDS:
        for renderer in capacity.RENDERERS:
            for family in capacity.FAMILIES:
                for model_seed in capacity.MODEL_SEEDS:
                    score, area = family_values[family]
                    runs.append(
                        {
                            "dataset_seed": dataset_seed,
                            "renderer": renderer,
                            "family": family,
                            "model_seed": model_seed,
                            "linear_task_evaluation": _task_result(score, area),
                            "operational_diagnostics": {
                                "calibrated_flags": {
                                    "hidden_scanner_leakage_detected": False
                                }
                            },
                            "interpretation_flags": {
                                "expanded_scanner_leakage_confirmed": bool(
                                    expanded_family == family
                                    and dataset_seed == 4301
                                    and renderer == "linear"
                                    and model_seed == 2201
                                ),
                                "operational_capabilities_preserved": True,
                            },
                        }
                    )
    return runs


def test_factorial_pairs_match_condition_keys() -> None:
    runs = _fake_runs({family: (index / 10, index / 10) for index, family in enumerate(capacity.FAMILIES)})
    effects = capacity.paired_factorial_effects(runs)
    for effect in effects.values():
        assert len(effect["full_budget_r2"]["paired_differences"]) == 16
        assert all(
            {"dataset_seed", "renderer", "model_seed"} <= set(row)
            for row in effect["full_budget_r2"]["paired_differences"]
        )


def test_dimension_and_parameter_contrasts_are_computed_correctly() -> None:
    values = {
        "b32_h128_low_budget": (0.1, 0.1),
        "b64_h112_low_budget": (0.2, 0.2),
        "b32_h145_high_budget": (0.3, 0.3),
        "b64_h128_high_budget": (0.4, 0.4),
    }
    effects = capacity.paired_factorial_effects(_fake_runs(values))
    assert effects["dimension_effect_low_budget"]["full_budget_r2"]["median"] == pytest.approx(0.1)
    assert effects["dimension_effect_high_budget"]["full_budget_r2"]["median"] == pytest.approx(0.1)
    assert effects["parameter_budget_effect_dimension_32"]["full_budget_r2"]["median"] == pytest.approx(0.2)
    assert effects["parameter_budget_effect_dimension_64"]["full_budget_r2"]["median"] == pytest.approx(0.2)


def test_material_effect_thresholds_are_fixed() -> None:
    assert capacity.MATERIAL_GAIN == 0.05
    assert capacity.MATERIAL_POSITIVE_COUNT == 12
    assert capacity.BOOTSTRAP_REPLICATES == 10_000


def test_one_favorable_seed_cannot_establish_factor_effect() -> None:
    rows = [
        {"difference": 0.1 if index == 0 else -0.01} for index in range(16)
    ]
    summary = capacity.paired_difference_summary(rows, 1)
    assert summary["positive_effect_count"] == 1
    assert summary["positive_effect_count"] < capacity.MATERIAL_POSITIVE_COUNT


def test_scanner_tradeoff_logic_is_predeclared() -> None:
    values = {
        "b32_h128_low_budget": (0.1, 0.1),
        "b64_h112_low_budget": (0.2, 0.2),
        "b32_h145_high_budget": (0.1, 0.1),
        "b64_h128_high_budget": (0.2, 0.2),
    }
    effects = capacity.paired_factorial_effects(
        _fake_runs(values, expanded_family="b64_h112_low_budget")
    )
    assert effects["dimension_effect_low_budget"]["materially_positive"]
    assert effects["dimension_effect_low_budget"]["scanner_tradeoff"][
        "scanner_tradeoff_detected"
    ]


def test_poor_scientific_performance_is_not_execution_failure() -> None:
    effects = capacity.paired_factorial_effects(
        _fake_runs({family: (0.1, 0.1) for family in capacity.FAMILIES})
    )
    status = capacity.factor_conclusions(effects)["status"]
    assert status == "complete_capacity_allocation_mechanism_unsupported"


def test_previous_statuses_remain_immutable(frozen: dict) -> None:
    assert frozen["routed_result"]["status"] == capacity.ROUTED_STATUS
    assert frozen["auxiliary_anchor"]["status"] == "complete_consensus_anchor_operational_tradeoff"


def test_existing_output_directories_cannot_be_overwritten(tmp_path: Path) -> None:
    with pytest.raises(capacity.ExperimentError, match="overwrite"):
        capacity.ensure_new_output_root(tmp_path)


def test_heterogeneous_csv_output_is_supported(tmp_path: Path) -> None:
    rows = [{"a": 1}, {"a": 2, "b": 3}]
    path = tmp_path / "summary.csv"
    capacity.parent.atomic_csv(path, capacity.parent.summary_csv_fieldnames(rows), rows)
    with path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["b"] == ""


def test_all_frozen_artifacts_remain_unchanged(frozen: dict) -> None:
    after = capacity.verify_routed_chain(REPOSITORY, ROUTED_RESULT)
    assert {name: row["file_sha256"] for name, row in frozen.items()} == {
        name: row["file_sha256"] for name, row in after.items()
    }


def test_no_path_forward_gate_is_defined() -> None:
    assert "path_forward_gate_open" not in inspect.getsource(capacity)
