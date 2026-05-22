from src.features.federated.pathology_fl.weighting.benchmark import (
    compare_weighting_strategies,
)
from src.features.federated.pathology_fl.weighting.synthetic_federation import (
    default_synthetic_federation,
)


def test_compare_weighting_strategies_returns_expected_reports():
    reports = compare_weighting_strategies(default_synthetic_federation())

    strategies = {report.strategy for report in reports}
    assert strategies == {
        "equal",
        "volume",
        "prestige",
        "fair_weights_h",
    }

    for report in reports:
        assert 0.0 <= report.normalized_entropy <= 1.0
        assert report.effective_institution_count >= 1.0
        assert report.max_weight <= 1.0
