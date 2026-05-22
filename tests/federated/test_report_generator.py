from src.features.federated.pathology_fl.weighting.report_generator import (
    generate_canonical_experiment_report,
)


def test_report_contains_expected_sections():
    report = generate_canonical_experiment_report()
    assert "# FAIR-WEIGHTS-H Synthetic Perturbation Report" in report
    assert "Synthetic engineering check" in report
    assert "fair_weights_h" in report
    assert "Interpretation Guardrail" in report
