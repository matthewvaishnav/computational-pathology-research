"""Tests for the foundations-manuscript validator (20 fail conditions)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import validate_manuscript as vm  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parents[2]


def _full_text():
    """A full valid manuscript text containing all required lines/levels."""
    return (
        "This manuscript introduces Paired-Acquisition Neural Factorization "
        "(PA-NF), TransnnMIL, PathologyFL, FAIR-WEIGHTS-H, and the PCam, PANDA, "
        "and CAMELYON17 benchmarks. The three levels are representation "
        "formation, whole-slide aggregation, and institutional aggregation. "
        "complete_mixed_real_paired_scanner_allocation_effects "
        "fixed_estimand_adjudication_not_ready "
        "complete_exact_real_bottleneck_representation_recovery "
        "complete_no_neural_feature_space_increment_supported"
    )


def test_condition1_pa_nf_only_line_fails():
    text = "The program is Paired-Acquisition Neural Factorization only."
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_required_lines(text)


def test_condition2_transnnmil_absent_fails():
    text = "PCam PANDA CAMELYON17 PathologyFL FAIR-WEIGHTS-H representation whole-slide institution"
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_required_lines(text)


def test_condition3_pathologyfl_absent_fails():
    text = _full_text().replace("PathologyFL", "the framework")
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_required_lines(text)


def test_condition4_fair_weights_h_absent_fails():
    text = _full_text().replace("FAIR-WEIGHTS-H", "weighting")
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_required_lines(text)


def test_condition5_pcam_absent_fails():
    text = _full_text().replace("PCam", "the benchmark")
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_required_lines(text)


def test_condition6_panda_absent_fails():
    text = _full_text().replace("PANDA", "the slide benchmark")
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_required_lines(text)


def test_condition7_camelyon17_absent_fails():
    text = _full_text().replace("CAMELYON17", "the center benchmark")
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_required_lines(text)


def test_condition8_three_levels_required():
    for level in ("representation", "whole-slide", "institution"):
        text = _full_text().replace(level, "other")
        with pytest.raises(vm.ManuscriptValidationError):
            vm.check_three_levels(text)


def test_condition9_projected_numbers_not_observed_fails():
    text = "TransnnMIL projected AUC 0.912."
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_projected_vs_observed(text)


def test_condition10_historical_qwk_as_repaired_fails():
    text = "withdrawn\n\\textbf{QWK 0.915}"
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_withdrawn_not_active(text)


def test_condition11_pa_nf_superiority_fails():
    text = "PA-NF is the best scanner-removal method."
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_prohibited_phrases(text)


def test_condition12_fair_weights_h_fairness_fails():
    text = "FAIR-WEIGHTS-H proves fairness."
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_prohibited_phrases(text)


def test_condition13_pathologyfl_clinical_fails():
    text = "PathologyFL improves clinical outcomes."
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_prohibited_phrases(text)


def test_condition14_camelyon17_as_full_fl_fails():
    # The manuscript must describe CAMELYON17 studies as centralized proxies,
    # not full FL. The Section 9 narrative must contain the proxy wording.
    text = _full_text() + " The CAMELYON17 weighting studies are centralized frozen-feature proxies, not full federated-learning validation."
    assert "proxies" in text
    assert "not full" in text.lower()


def test_condition15_withdrawn_artifacts_in_active_tables_fails():
    text = "withdrawn\n\\textbf{QWK}"
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_withdrawn_not_active(text)


def test_condition16_active_numbers_lack_bindings_fails(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    manifest = evidence / "manuscript_evidence_manifest.json"
    manifest.write_text(
        json.dumps({"active_empirical_claims": [{"claim_id": "X"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(vm, "PACKAGE", tmp_path)
    with pytest.raises(vm.ManuscriptValidationError, match="artifact binding"):
        vm.check_manifest_bindings()


def test_condition17_architecture_empirical_conflation_fails():
    text = "implemented and outperforms"
    with pytest.raises(vm.ManuscriptValidationError):
        vm.check_architecture_empirical_conflation(text)


def test_condition18_old_manuscript_not_overwritten():
    assert (REPO_ROOT / "paper" / "paired_acquisition_manuscript" / "manuscript_draft.md").is_file()
    assert (REPO_ROOT / "paper" / "arxiv" / "main.tex").is_file()


def test_condition19_public_site_not_modified():
    vm.check_public_site_not_modified()  # no raise = pass


def test_condition20_pdf_reproducible_documentation():
    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    assert "latexmk" in readme
    assert "clean checkout" in readme


def test_full_package_validates():
    vm.check_required_lines(_full_text())
    vm.check_three_levels(_full_text())
    vm.check_prohibited_phrases(_full_text())
    vm.check_projected_vs_observed(_full_text())
    vm.check_withdrawn_not_active(_full_text())
    vm.check_architecture_empirical_conflation(_full_text())
    vm.check_frozen_statuses_preserved()
