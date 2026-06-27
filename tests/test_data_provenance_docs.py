from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_data_provenance_document_exists():
    text = (ROOT / "docs" / "DATA_PROVENANCE.md").read_text(encoding="utf-8")
    assert "real human histopathology data" in text
    assert "CAMELYON16/17" in text
    assert "PatchCamelyon (PCam)" in text
    assert "PANDA" in text
    assert "test-only" in text


def test_camelyon_status_does_not_misstate_research_data():
    text = (ROOT / "docs" / "CAMELYON_TRAINING_STATUS.md").read_text(encoding="utf-8").lower()
    forbidden = (
        "current results are on synthetic data",
        "camelyon results are synthetic",
        "real camelyon data has not been integrated",
        "synthetic-only",
    )
    for phrase in forbidden:
        assert phrase not in text


def test_active_overview_has_data_provenance_boundary():
    text = (ROOT / "docs" / "overview" / "index.md").read_text(encoding="utf-8")
    assert "real human histopathology" in text
    assert "software-test fixtures" in text
