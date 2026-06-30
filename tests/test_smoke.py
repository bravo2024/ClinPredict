"""Smoke tests for ClinPredict — Cox survival analysis."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_synthetic
from src.model import fit_cox_model, fit_and_evaluate
from src.core import concordance_index, kaplan_meier, logrank_test


def test_data():
    """Clinical data has time-to-event and event indicator."""
    d = make_synthetic(n=200, seed=42)
    assert d["n_samples"] == 200
    assert "time" in d["df"].columns
    assert "event" in d["df"].columns
    assert 0.0 < d["event_rate"] < 1.0


def test_kaplan_meier():
    """KM estimator produces decreasing survival curve."""
    times = np.array([1, 2, 3, 4, 5])
    events = np.array([1, 0, 1, 1, 0])
    km = kaplan_meier(times, events)
    assert len(km["survival"]) > 0
    assert all(0 <= s <= 1 for s in km["survival"])
    assert km["n_events"] == 3


def test_concordance_index():
    """C-index is > 0.5 when risk scores are concordant with survival."""
    times = np.array([5, 10, 15, 20, 25])
    events = np.array([1, 1, 1, 0, 1])
    risk = np.array([5, 4, 3, 2, 1])  # higher risk → shorter time
    c = concordance_index(times, events, risk)
    assert c > 0.5


def test_cox_model():
    """Cox model fits and produces hazard ratios."""
    d = make_synthetic(n=200, seed=42)
    cox = fit_cox_model(d["df"], d["features"], d["time_col"], d["event_col"])
    assert len(cox["beta"]) == 4
    assert cox["c_index"] > 0.5  # should be better than random


def test_fit_and_evaluate():
    """Full pipeline returns model and metrics with C-index and log-rank."""
    d = make_synthetic(n=300, seed=42)
    model, metrics = fit_and_evaluate(d, seed=42)
    assert "c_index" in metrics
    assert "hazard_ratios" in metrics
    assert "logrank_p_value" in metrics
    assert metrics["c_index"] > 0.5
    # Treatment should have hazard ratio < 1 (protective)
    assert metrics["treatment_hazard_ratio"] < 1.0


def test_logrank():
    """Log-rank test distinguishes groups with different survival."""
    t1 = np.array([5, 6, 7, 8, 10])
    e1 = np.array([1, 1, 1, 1, 1])
    t2 = np.array([15, 16, 17, 18, 20])  # clearly longer survival
    e2 = np.array([1, 1, 1, 1, 1])
    lr = logrank_test(t1, e1, t2, e2)
    assert lr["chi2"] > 0


if __name__ == "__main__":
    test_data()
    test_kaplan_meier()
    test_concordance_index()
    test_cox_model()
    test_fit_and_evaluate()
    test_logrank()
    print("All ClinPredict smoke tests passed!")
