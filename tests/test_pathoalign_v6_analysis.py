import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/pathoalign_identifiability_v6/analyze_two_resource_law.py"


def load():
    name = "pathoalign_v6_analysis_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def test_sustained_crossing_rejects_transient_pass():
    m = load()
    frame = pd.DataFrame(
        {"pair_count": [0, 10, 20, 30], "universal_biological_score_mean": [0.4, 0.52, 0.48, 0.55]}
    )
    r = m.boundary_metrics(frame, 0.5)
    assert r["first_crossing"] == 10
    assert r["sustained_crossing"] == 30
    assert r["post_crossing_failures"] == 1


def test_no_recovery_is_reported_above_grid():
    m = load()
    frame = pd.DataFrame(
        {"pair_count": [0, 10, 20], "universal_biological_score_mean": [0.2, 0.3, 0.4]}
    )
    r = m.boundary_metrics(frame, 0.5)
    assert pd.isna(r["sustained_crossing"])
    assert r["sustained_boundary_interval"] == "> 20"
